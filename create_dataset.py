"""
Dataset creation for CPU-1: BPE-to-byte alignment and dataset pipeline.

Section 1 - Core algorithms: BPE-to-byte alignment, first-byte marginalization,
            document packing, and dynamic patch marginalization.
Section 2 - Pipeline: teacher logprob extraction (Qwen-7B), byte-level and
            BPE dataset construction, and HuggingFace Hub upload.

Usage:
    # Full pipeline
    python create_dataset.py --n_docs 50000

    # Build byte marginalization dataset from logprobs
    python create_dataset.py --logprobs-dir ./logprobs --output-dir ./dataset
"""

# MUST be set before any HuggingFace tokenizer import.
# ProcessPoolExecutor uses fork() on Linux; the tokenizer's internal Rust threads
# hold locks that get cloned in a locked state, deadlocking all CPU workers.
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import enum
import gc
import json
import math
import random
import shutil
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from queue import Queue, Empty
from typing import NamedTuple

import numpy as np
import torch

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it


# =============================================================================
# Section 1: BPE-to-Byte Alignment Core
# =============================================================================

# CPU-1 procesa texto como patches de 4 bytes cada uno
PATCH_SIZE = 4
# Vocabulario de bytes: 0-255 (un byte = un valor posible)
VOCAB_SIZE = 256


# =============================================================================
# 1. Tipos
# =============================================================================

class AlignmentCase(enum.Enum):
    """
    Tipo de alineamiento entre patch boundary y BPE boundary.

    El problema central: los patches son de 4 bytes fijos, pero los tokens BPE
    tienen largo variable (1-15 bytes). Los boundaries casi nunca coinciden.
    Este enum clasifica cada posicion para saber si hay senal del teacher.
    """
    ALIGNED = "aligned"           # Caso 1: boundary patch = boundary BPE -> hay senal teacher
    INSIDE_TOKEN = "inside"       # Caso 2: dentro de un token BPE -> byte deterministico, solo CE
    MULTI_TOKEN = "multi"         # Caso 3: multiples tokens BPE en un patch -> senal en el primero


class TokenInfo(NamedTuple):
    """Info de un token BPE con su span en bytes (inicio inclusive, fin exclusive)."""
    token_str: str          # texto del token decodificado
    token_bytes: list[int]  # bytes UTF-8 del token
    byte_start: int         # posicion del primer byte en el texto completo
    byte_end: int           # posicion despues del ultimo byte (exclusive)


class PatchTeacherSignal(NamedTuple):
    """Senal del teacher para un patch target (4 bytes)."""
    has_teacher: list[bool]                          # [4] True por cada byte con senal teacher
    byte_distributions: list[dict[int, float] | None]  # [4] P(byte) marginalizada o None


# =============================================================================
# 2. BPEToByteAligner
# =============================================================================

class BPEToByteAligner:
    """
    Alinea tokens BPE del teacher con patches de bytes del student.
    Determina que posiciones tienen senal del teacher y cuales no.
    """

    def __init__(self, patch_size: int = PATCH_SIZE):
        """
        Args:
            patch_size: bytes por patch del student (4 para CPU-1)
        """
        self.patch_size = patch_size

    def tokens_to_byte_spans(self, token_infos: list[dict]) -> list[TokenInfo]:
        """
        Convierte lista de tokens con bytes en TokenInfo con byte spans.

        Args:
            token_infos: lista de dicts con 'token_str'/'token' y
                         'token_bytes'/'bytes' (list[int])
        Returns:
            lista de TokenInfo con byte_start, byte_end
        """
        spans = []
        pos = 0
        for info in token_infos:
            token_str = info.get('token_str', info.get('token', ''))
            token_bytes = info.get('token_bytes', info.get('bytes', []))
            n = len(token_bytes)
            spans.append(TokenInfo(
                token_str=token_str,
                token_bytes=token_bytes,
                byte_start=pos,
                byte_end=pos + n,
            ))
            pos += n
        return spans

    def find_alignment_cases(
        self,
        token_spans: list[TokenInfo],
        total_bytes: int,
    ) -> list[tuple[int, AlignmentCase, int | None]]:
        """
        Para cada byte de cada patch target, determina el caso de alineamiento.

        Returns:
            Lista de (byte_position, AlignmentCase, token_index_or_None)
            donde token_index es el indice del token BPE que empieza en esa posicion
            (solo para ALIGNED), None para los demas.
        """
        # Indice rapido: posicion_byte -> indice_token (O(1) lookup)
        bpe_starts: dict[int, int] = {}
        for i, span in enumerate(token_spans):
            bpe_starts[span.byte_start] = i

        # Mapear cada byte a su token BPE contenedor.
        # Necesario para saber si un byte esta "dentro" de un token.
        byte_to_token = [0] * total_bytes
        for i, span in enumerate(token_spans):
            for b in range(span.byte_start, min(span.byte_end, total_bytes)):
                byte_to_token[b] = i

        results = []
        n_patches = (total_bytes + self.patch_size - 1) // self.patch_size

        for p in range(n_patches):
            for offset in range(self.patch_size):
                byte_pos = p * self.patch_size + offset
                if byte_pos >= total_bytes:
                    results.append((byte_pos, AlignmentCase.INSIDE_TOKEN, None))
                    continue

                if byte_pos in bpe_starts:
                    results.append((byte_pos, AlignmentCase.ALIGNED, bpe_starts[byte_pos]))
                else:
                    results.append((byte_pos, AlignmentCase.INSIDE_TOKEN, None))

        return results

    def build_teacher_targets(
        self,
        logprobs_positions: list[dict],
        token_spans: list[TokenInfo],
        text_bytes: bytes,
    ) -> list[PatchTeacherSignal]:
        """
        Construye senales del teacher para cada patch target.

        El patch target[i] contiene los bytes del proximo patch (i+1).
        Para cada byte del patch target, determina:
        - Si hay senal teacher (ALIGNED con BPE boundary)
        - La distribucion marginalizada del teacher (o None)

        Args:
            logprobs_positions: output de extract_logprobs (lista de dicts por posicion)
            token_spans: tokens BPE con byte spans
            text_bytes: texto original en bytes
        Returns:
            Lista de PatchTeacherSignal, uno por patch target
        """
        total_bytes = len(text_bytes)
        n_patches = (total_bytes + self.patch_size - 1) // self.patch_size

        # Mapa token_index -> logprobs de prediccion.
        # En modelo autoregresivo: logprobs_positions[i] predice token[i+1].
        # Entonces lo indexamos por i+1 (el token que se esta prediciendo).
        token_logprobs: dict[int, list[dict]] = {}
        for i, pos_data in enumerate(logprobs_positions):
            token_logprobs[i + 1] = pos_data.get('top_tokens', [])

        # BPE boundaries
        bpe_starts: dict[int, int] = {}
        for i, span in enumerate(token_spans):
            bpe_starts[span.byte_start] = i

        marginalizer = FirstByteMarginalization()
        signals = []

        for p in range(n_patches):
            has_teacher = [False] * self.patch_size
            byte_dists: list[dict[int, float] | None] = [None] * self.patch_size

            for offset in range(self.patch_size):
                byte_pos = p * self.patch_size + offset
                if byte_pos >= total_bytes:
                    continue

                # Solo hay senal si este byte es el inicio de un token BPE
                # Y tenemos logprobs para la prediccion de ese token
                if byte_pos in bpe_starts:
                    token_idx = bpe_starts[byte_pos]
                    if token_idx in token_logprobs:
                        top_tokens = token_logprobs[token_idx]
                        if top_tokens:
                            dist = marginalizer.marginalize(top_tokens)
                            if dist:
                                has_teacher[offset] = True
                                byte_dists[offset] = dist

            signals.append(PatchTeacherSignal(
                has_teacher=has_teacher,
                byte_distributions=byte_dists,
            ))

        return signals


# =============================================================================
# 3. FirstByteMarginalization
# =============================================================================

class FirstByteMarginalization:
    """
    Convierte P(next_token) del teacher en P(next_byte) para el student.

    Para cada token candidato, toma SOLO su primer byte y acumula probabilidad.
    Normaliza sobre bytes observados.
    """

    def marginalize(self, top_tokens: list[dict]) -> dict[int, float]:
        """
        Marginalizacion basica: normalizar sobre top-k.

        Args:
            top_tokens: lista de {token, bytes, logprob}
        Returns:
            dict byte_value (0-255) -> probabilidad normalizada
        """
        byte_probs: dict[int, float] = {}
        for item in top_tokens:
            token_bytes = item.get('bytes', item.get('token_bytes', []))
            if not token_bytes:
                continue
            first_byte = token_bytes[0]
            prob = math.exp(item['logprob'])
            byte_probs[first_byte] = byte_probs.get(first_byte, 0.0) + prob

        total = sum(byte_probs.values())
        if total > 0:
            byte_probs = {k: v / total for k, v in byte_probs.items()}

        return byte_probs

    def marginalize_with_smoothing(
        self,
        top_tokens: list[dict],
        alpha: float = 0.01,
    ) -> dict[int, float]:
        """
        Marginalizacion con smoothing uniforme para bytes no observados.

        Args:
            top_tokens: lista de {token, bytes, logprob} del teacher
            alpha:      fraccion de masa reservada para bytes no vistos (0.01 = 1%)

        Returns:
            dict con 256 entries (todas las posibles), byte_value -> prob

        Reserva alpha de masa de probabilidad para distribuir uniformemente
        entre bytes no vistos en el top-k. Evita que el student asigne prob=0
        a bytes validos que el top-k del teacher no cubrio.
        """
        raw = self.marginalize(top_tokens)
        if not raw:
            return {}

        smoothed: dict[int, float] = {}
        for k, v in raw.items():
            smoothed[k] = v * (1.0 - alpha)

        unseen = VOCAB_SIZE - len(smoothed)
        if unseen > 0:
            uniform = alpha / unseen
            for b in range(VOCAB_SIZE):
                if b not in smoothed:
                    smoothed[b] = uniform

        return smoothed

    def to_256_array(self, byte_probs: dict[int, float]) -> np.ndarray:
        """
        Convierte dict sparse a array denso [256] para el dataset.

        Args:
            byte_probs: dict byte -> prob (claves pueden ser int o str tras JSON roundtrip)
        Returns:
            np.ndarray float32 [256]
        """
        arr = np.zeros(VOCAB_SIZE, dtype=np.float32)
        for byte_val, prob in byte_probs.items():
            byte_val = int(byte_val)  # JSON deserializa claves int como str
            if 0 <= byte_val < VOCAB_SIZE:
                arr[byte_val] = prob
        return arr


# =============================================================================
# 4. DistillationDatasetBuilder
# =============================================================================

class DistillationDatasetBuilder:
    """
    Construye el dataset final de distillation a partir de logprobs crudos.

    Input: directorio con Parquet de logprobs (de extract_logprobs.py)
    Output: dataset HuggingFace-compatible con patches, targets, teacher signals

    Formato de salida por sample:
        patches:       [seq_len, 4] uint8 -- input patches
        targets:       [seq_len, 4] uint8 -- next patches (shifted)
        teacher_probs: [seq_len, 256] float32 -- distribucion byte 0 del target
        teacher_mask:  [seq_len] bool -- True donde hay senal teacher para byte 0
    """

    def __init__(
        self,
        logprobs_dir: str,
        output_dir: str,
        patch_size: int = PATCH_SIZE,
        seq_len: int = 2048,
    ):
        """
        Args:
            logprobs_dir: directorio con logprobs_*.parquet (output de extract_logprobs.py)
            output_dir:   directorio para guardar distillation_dataset.parquet
            patch_size:   bytes por patch (4 para CPU-1)
            seq_len:      patches por secuencia despues del packing (2048 = 8KB de contexto)
        """
        self.logprobs_dir = Path(logprobs_dir)
        self.output_dir = Path(output_dir)
        self.patch_size = patch_size
        self.seq_len = seq_len
        self.aligner = BPEToByteAligner(patch_size)
        self.marginalizer = FirstByteMarginalization()

    def process_document(self, record: dict) -> dict | None:
        """
        Procesa un documento de logprobs en patches + teacher signals.

        Args:
            record: dict con doc_id, text_bytes, positions
        Returns:
            dict con patches, targets, teacher_probs, teacher_mask o None
        """
        text_bytes = record.get('text_bytes', b'')
        if isinstance(text_bytes, str):
            text_bytes = text_bytes.encode('utf-8')

        positions = record.get('positions', [])
        if not text_bytes or not positions:
            return None

        n_bytes = len(text_bytes)

        # Pad a multiplo de patch_size con bytes nulos.
        # Ej: 11 bytes -> 12 bytes (3 patches de 4)
        pad = (self.patch_size - n_bytes % self.patch_size) % self.patch_size
        padded = text_bytes + b'\x00' * pad
        n_patches = len(padded) // self.patch_size

        if n_patches < 2:
            return None

        # Construir array de patches [n_patches, 4]
        patches = np.zeros((n_patches, self.patch_size), dtype=np.uint8)
        for i in range(n_patches):
            start = i * self.patch_size
            for j in range(self.patch_size):
                patches[i, j] = padded[start + j]

        # Targets: siguiente patch (shifted by 1).
        # El modelo predice el patch siguiente dado el actual:
        # patches[i] -> targets[i] = patches[i+1].
        # El ultimo target queda en ceros (no hay patch siguiente).
        targets = np.zeros_like(patches)
        targets[:-1] = patches[1:]

        # Teacher signals
        # Para cada posicion de patch target, verificar si el primer byte
        # del target patch coincide con un boundary BPE
        teacher_probs = np.zeros((n_patches, VOCAB_SIZE), dtype=np.float32)
        teacher_mask = np.zeros(n_patches, dtype=bool)

        # Reconstruir token spans desde las posiciones de logprobs
        for pos_data in positions:
            first_byte_probs = pos_data.get('first_byte_probs', {})
            if not first_byte_probs:
                continue

            byte_position = pos_data.get('byte_position', -1)
            if byte_position < 0:
                continue

            # Este byte_position corresponde al inicio de un token BPE
            # Lo mapeamos al patch que lo contiene
            patch_idx = byte_position // self.patch_size
            byte_offset = byte_position % self.patch_size

            # Logica de alineamiento:
            # target[i] predice patches[i+1]. Si byte_position es el primer byte
            # de patches[k] (byte_offset=0), la senal teacher aplica a target[k-1].
            # Solo usamos senal en byte_offset=0 (primer byte del patch target)
            # porque la first-byte marginalization solo da P(primer byte del siguiente token).
            if byte_offset == 0 and patch_idx > 0:
                target_idx = patch_idx - 1
                if 0 <= target_idx < n_patches:
                    probs_array = self.marginalizer.to_256_array(first_byte_probs)
                    teacher_probs[target_idx] = probs_array
                    teacher_mask[target_idx] = True

        return {
            'doc_id': record.get('doc_id', ''),
            'patches': patches.tolist(),
            'targets': targets.tolist(),
            'teacher_probs': teacher_probs.tolist(),
            'teacher_mask': teacher_mask.tolist(),
        }

    def pack_documents(
        self,
        documents: list[dict],
    ) -> list[dict]:
        """
        Document packing: concatena documentos hasta llenar seq_len patches.
        Inserta patch separador [0,0,0,0] entre documentos.

        Args:
            documents: lista de dicts procesados (output de process_document)

        Returns:
            Lista de samples empaquetados, cada uno con exactamente seq_len patches.
            Secuencias incompletas (< seq_len) se descartan.

        Nota: el packing maximiza utilizacion de GPU al evitar padding excesivo.
        Sin packing, documentos cortos desperdician la mayor parte de la secuencia.
        """
        packed = []
        current_patches = []
        current_targets = []
        current_teacher_probs = []
        current_teacher_mask = []

        separator_patch = [0] * self.patch_size
        separator_probs = [0.0] * VOCAB_SIZE

        for doc in documents:
            doc_patches = doc['patches']
            doc_targets = doc['targets']
            doc_teacher_probs = doc['teacher_probs']
            doc_teacher_mask = doc['teacher_mask']

            # Agregar separador si hay contenido previo
            if current_patches:
                current_patches.append(separator_patch)
                current_targets.append(separator_patch)
                current_teacher_probs.append(separator_probs)
                current_teacher_mask.append(False)

            # Agregar documento
            for i in range(len(doc_patches)):
                current_patches.append(doc_patches[i])
                current_targets.append(doc_targets[i])
                current_teacher_probs.append(doc_teacher_probs[i])
                current_teacher_mask.append(doc_teacher_mask[i])

                # Emitir sample cuando llegamos a seq_len
                if len(current_patches) >= self.seq_len:
                    packed.append({
                        'patches': current_patches[:self.seq_len],
                        'targets': current_targets[:self.seq_len],
                        'teacher_probs': current_teacher_probs[:self.seq_len],
                        'teacher_mask': current_teacher_mask[:self.seq_len],
                    })
                    current_patches = []
                    current_targets = []
                    current_teacher_probs = []
                    current_teacher_mask = []

        # Descartar sobrante incompleto (no padear)
        return packed

    def build(self, n_workers: int = 0):
        """
        Pipeline completo: lee logprobs -> procesa docs -> empaqueta -> guarda Parquet.
        Paralelizado con ProcessPoolExecutor para saturar CPU multi-core.

        Flujo:
            1. Lee todos los logprobs_*.parquet de logprobs_dir
            2. Procesa cada documento en paralelo: bytes -> patches + teacher signals
            3. Empaqueta documentos en secuencias de seq_len patches
            4. Guarda como distillation_dataset.parquet con zstd
        """
        if pq is None:
            raise ImportError("pyarrow requerido: pip install pyarrow")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        parquet_files = sorted(self.logprobs_dir.glob("logprobs_*.parquet"))
        if not parquet_files:
            print(f"No hay archivos logprobs_*.parquet en {self.logprobs_dir}")
            return

        if n_workers <= 0:
            n_workers = max(1, (os.cpu_count() or 4) - 1)

        print(f"Procesando {len(parquet_files)} archivos de logprobs ({n_workers} workers)...")

        # Cargar todos los records
        all_records = []
        for pf in parquet_files:
            table = pq.read_table(pf)
            for i in range(len(table)):
                all_records.append({
                    'doc_id': table['doc_id'][i].as_py(),
                    'text_bytes': table['text_bytes'][i].as_py(),
                    'positions': json.loads(table['positions'][i].as_py()),
                })

        # Descomposición de datos: cada worker procesa un documento independiente
        if n_workers > 1 and len(all_records) > 10:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(
                    self.process_document, all_records,
                    chunksize=max(1, len(all_records) // (n_workers * 4)),
                ))
            all_docs = [r for r in results if r is not None]
        else:
            all_docs = [r for r in (self.process_document(rec) for rec in all_records) if r]

        print(f"Documentos procesados: {len(all_docs)}")

        # Document packing
        packed = self.pack_documents(all_docs)
        print(f"Samples empaquetados (seq_len={self.seq_len}): {len(packed)}")

        if not packed:
            print("Sin samples para guardar")
            return

        # Guardar como Parquet
        output_file = self.output_dir / "distillation_dataset.parquet"
        arrays = {
            'patches': pa.array([json.dumps(s['patches']) for s in packed], type=pa.large_string()),
            'targets': pa.array([json.dumps(s['targets']) for s in packed], type=pa.large_string()),
            'teacher_probs': pa.array([json.dumps(s['teacher_probs']) for s in packed], type=pa.large_string()),
            'teacher_mask': pa.array([json.dumps(s['teacher_mask']) for s in packed], type=pa.large_string()),
        }
        table = pa.table(arrays)
        pq.write_table(table, str(output_file), compression='zstd')

        print(f"Dataset guardado: {output_file}")
        print(f"  Samples: {len(packed)}")
        print(f"  Patches/sample: {self.seq_len}")

    def validate_sample(self, n: int = 10):
        """
        Valida consistencia del dataset construido.

        Args:
            n: cantidad de samples a validar

        Verifica shapes y reporta teacher coverage promedio (~20-30% esperado).
        """
        output_file = self.output_dir / "distillation_dataset.parquet"
        if not output_file.exists():
            print(f"Dataset no encontrado: {output_file}")
            return

        table = pq.read_table(str(output_file))
        n = min(n, len(table))
        total_coverage = 0.0

        for i in range(n):
            patches = json.loads(table['patches'][i].as_py())
            targets = json.loads(table['targets'][i].as_py())
            teacher_mask = json.loads(table['teacher_mask'][i].as_py())

            assert len(patches) == self.seq_len
            assert len(targets) == self.seq_len
            assert len(teacher_mask) == self.seq_len

            coverage = sum(teacher_mask) / len(teacher_mask)
            total_coverage += coverage

        avg_coverage = total_coverage / n
        print(f"Validacion ({n} samples):")
        print(f"  Teacher coverage promedio: {avg_coverage:.1%}")
        print(f"  Patches/sample: {self.seq_len}")


# =============================================================================
# 5. DynamicPatchMarginalization — Parches dinámicos (Bolmo)
# =============================================================================

class DynamicPatchMarginalization:
    """
    Versión de byte_marginalization que soporta parches dinámicos (Bolmo).

    La diferencia vs la versión estática:
    - Los boundaries no son fijos cada 4 bytes
    - Los boundaries vienen del BoundaryPredictor o de las reglas naturales
    - La marginalización BPE→byte sigue siendo igual para los primeros bytes

    COMPATIBILIDAD:
    - Si boundaries=None → comportamiento idéntico a la versión estática
    - Los runs 01-11 usan la versión estática (boundaries=None)
    - Los runs 12-15 usan la versión dinámica
    """

    def __init__(self, patch_size_max: int = 8, patch_size_static: int = 4):
        self.patch_size_max    = patch_size_max
        self.patch_size_static = patch_size_static

    def build_teacher_targets_dynamic(
        self,
        text_bytes: bytes,
        byte_to_teacher: dict,   # byte_offset → {byte_val: prob}
        boundaries: list[bool],  # del BoundaryPredictor o reglas naturales
    ) -> list[dict]:
        """
        Construir targets de distilación para parches dinámicos.

        Para cada parche:
          - Si el primer byte del parche coincide con un boundary BPE
            → tiene señal del teacher (marginalización)
          - Si no → solo CE

        La señal del teacher siempre es para el primer byte del parche
        (igual que en la versión estática).

        Returns: lista de dicts por parche:
            {
                'patch':         list[int],  # bytes del parche (paddeados)
                'patch_len':     int,        # longitud real sin padding
                'has_teacher':   bool,
                'teacher_probs': list[float] | None,  # [256] o None
            }
        """
        patches = _bytes_to_dynamic_patches(
            text_bytes, boundaries, self.patch_size_max
        )

        results = []
        byte_offset = 0

        for patch in patches:
            # Longitud real (sin padding)
            patch_len = sum(1 for b in patch if b != 0) or 1

            # ¿Hay señal del teacher para el primer byte de este parche?
            has_teacher = byte_offset in byte_to_teacher
            teacher_probs = None

            if has_teacher:
                bp = byte_to_teacher[byte_offset]
                teacher_probs = [bp.get(b, 0.0) for b in range(256)]

            results.append({
                'patch':         patch,
                'patch_len':     patch_len,
                'has_teacher':   has_teacher,
                'teacher_probs': teacher_probs,
            })

            byte_offset += patch_len

        return results


def _bytes_to_dynamic_patches(
    byte_seq: bytes,
    boundaries: list[bool],
    max_patch_size: int = 8,
    pad_byte: int = 0,
) -> list[list[int]]:
    """
    Convertir secuencia de bytes a parches de longitud variable.

    Los parches tienen longitud entre 1 y max_patch_size bytes.
    Se paddean a max_patch_size con pad_byte para batch processing.

    Returns: lista de parches, cada uno de longitud max_patch_size (paddeado)
    """
    patches = []
    current_patch = []

    for i, (b, is_boundary) in enumerate(zip(byte_seq, boundaries)):
        current_patch.append(b)
        if is_boundary or len(current_patch) >= max_patch_size:
            padded = current_patch + [pad_byte] * (max_patch_size - len(current_patch))
            patches.append(padded[:max_patch_size])
            current_patch = []

    if current_patch:
        padded = current_patch + [pad_byte] * (max_patch_size - len(current_patch))
        patches.append(padded[:max_patch_size])

    return patches


# =============================================================================
# 6. Tests
# =============================================================================

def test_alignment_cases():
    """Verifica los 3 casos de alineamiento BPE <-> byte patches."""
    aligner = BPEToByteAligner(patch_size=4)

    # "Hello world" (11 bytes)
    # BPE: ["Hello"(5 bytes), " world"(6 bytes)]
    # Patches: [0:4], [4:8], [8:12] (con padding)
    tokens = [
        {'token_str': 'Hello', 'token_bytes': [72, 101, 108, 108, 111]},
        {'token_str': ' world', 'token_bytes': [32, 119, 111, 114, 108, 100]},
    ]
    spans = aligner.tokens_to_byte_spans(tokens)

    assert spans[0].byte_start == 0
    assert spans[0].byte_end == 5
    assert spans[1].byte_start == 5
    assert spans[1].byte_end == 11

    # Verificar alineamiento
    cases = aligner.find_alignment_cases(spans, total_bytes=11)

    # Byte 0: inicio patch 0 Y inicio token 0 -> ALIGNED
    assert cases[0][1] == AlignmentCase.ALIGNED

    # Byte 4: inicio patch 1, pero DENTRO de token "Hello" (que va 0-4) -> INSIDE
    assert cases[4][1] == AlignmentCase.INSIDE_TOKEN

    # Byte 5: NO inicio de patch (offset 1 del patch 1), pero inicio de token 1 -> ALIGNED
    assert cases[5][1] == AlignmentCase.ALIGNED

    # Byte 8: inicio patch 2, dentro de token " world" (5-10) -> INSIDE
    assert cases[8][1] == AlignmentCase.INSIDE_TOKEN

    print("[OK] Caso 1 (ALIGNED): byte 0 coincide con BPE boundary")
    print("[OK] Caso 2 (INSIDE_TOKEN): byte 4 cae dentro de 'Hello'")
    print("[OK] Caso 3 (ALIGNED en offset): byte 5 es BPE boundary aunque no es patch boundary")


def test_first_byte_marginalization():
    """Verifica marginalizacion con el ejemplo del spec."""
    marginalizer = FirstByteMarginalization()

    top_tokens = [
        {'token': 'the', 'bytes': [116, 104, 101], 'logprob': -0.5},
        {'token': 'The', 'bytes': [84, 104, 101], 'logprob': -1.2},
        {'token': 'to', 'bytes': [116, 111], 'logprob': -1.5},
    ]

    probs = marginalizer.marginalize(top_tokens)
    assert abs(probs[116] - 0.7336) < 0.001
    assert abs(probs[84] - 0.2664) < 0.001
    assert len(probs) == 2
    print("[OK] Marginalization basica: valores correctos")

    # Smoothing
    smoothed = marginalizer.marginalize_with_smoothing(top_tokens, alpha=0.01)
    assert len(smoothed) == VOCAB_SIZE
    assert abs(sum(smoothed.values()) - 1.0) < 0.001
    assert smoothed[116] > smoothed[0]  # byte 116 deberia ser mayor que uniform
    print("[OK] Marginalization con smoothing: 256 entries, suman 1.0")

    # to_256_array
    arr = marginalizer.to_256_array(probs)
    assert arr.shape == (256,)
    assert abs(arr[116] - 0.7336) < 0.001
    assert arr[0] == 0.0  # no observado
    print("[OK] to_256_array: shape y valores correctos")


def test_document_processing():
    """Verifica procesamiento end-to-end de un documento."""
    builder = DistillationDatasetBuilder(
        logprobs_dir=".",
        output_dir=".",
        seq_len=8,
    )

    # Documento simple: "ABCDEFGH" (8 bytes = 2 patches)
    text = "ABCDEFGH"
    text_bytes = text.encode('utf-8')

    record = {
        'doc_id': 'test',
        'text_bytes': text_bytes,
        'positions': [
            {
                'byte_position': 0,
                'top_tokens': [
                    {'token': 'AB', 'bytes': [65, 66], 'logprob': -0.5},
                    {'token': 'AC', 'bytes': [65, 67], 'logprob': -1.0},
                ],
                'first_byte_probs': {65: 1.0},  # 100% byte 'A'
            },
            {
                'byte_position': 4,
                'top_tokens': [
                    {'token': 'EF', 'bytes': [69, 70], 'logprob': -0.3},
                    {'token': 'EG', 'bytes': [69, 71], 'logprob': -2.0},
                ],
                'first_byte_probs': {69: 1.0},  # 100% byte 'E'
            },
        ],
    }

    result = builder.process_document(record)
    assert result is not None
    assert len(result['patches']) == 2
    assert len(result['targets']) == 2
    assert len(result['teacher_mask']) == 2

    # El byte_position=4 es el inicio del patch 1.
    # Esto da senal al target[0] (que predice patch[1]).
    assert result['teacher_mask'][0] == True, "Target 0 deberia tener senal (byte 4 = inicio patch 1)"

    print("[OK] Document processing: patches, targets, teacher signals correctos")


def test_document_packing():
    """Verifica document packing con separador."""
    builder = DistillationDatasetBuilder(
        logprobs_dir=".",
        output_dir=".",
        seq_len=8,
    )

    doc1 = {
        'patches': [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
        'targets': [[5, 6, 7, 8], [9, 10, 11, 12], [0, 0, 0, 0]],
        'teacher_probs': [[0.0] * 256] * 3,
        'teacher_mask': [False, True, False],
    }
    doc2 = {
        'patches': [[13, 14, 15, 16], [17, 18, 19, 20]],
        'targets': [[17, 18, 19, 20], [0, 0, 0, 0]],
        'teacher_probs': [[0.0] * 256] * 2,
        'teacher_mask': [True, False],
    }

    packed = builder.pack_documents([doc1, doc2])

    # doc1 (3) + separator (1) + doc2 (2) = 6 patches
    # 6 < seq_len=8, asi que no se emite sample completo
    assert len(packed) == 0, "No deberia haber sample completo (6 < 8)"

    # Con seq_len=6
    builder.seq_len = 6
    packed = builder.pack_documents([doc1, doc2])
    assert len(packed) == 1
    assert len(packed[0]['patches']) == 6

    # Verificar separador en posicion 3
    assert packed[0]['patches'][3] == [0, 0, 0, 0], "Posicion 3 debe ser separador"
    assert packed[0]['teacher_mask'][3] == False, "Separador no tiene teacher"

    print("[OK] Document packing: separador y truncamiento correctos")


# =============================================================================
# 6. CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Byte marginalization y dataset builder")

    p.add_argument('--logprobs-dir', type=str, default=None,
                   help="Directorio con Parquet de logprobs")
    p.add_argument('--output-dir', type=str, default='./dataset',
                   help="Directorio de salida para dataset")
    p.add_argument('--seq-len', type=int, default=2048,
                   help="Patches por secuencia (para document packing)")
    p.add_argument('--validate', action='store_true',
                   help="Validar dataset existente")

    return p.parse_args()


def main():
    import sys

    if len(sys.argv) == 1:
        print("Sin argumentos -> corriendo tests\n")
        print("=" * 60)
        test_alignment_cases()
        print()
        test_first_byte_marginalization()
        print()
        test_document_processing()
        print()
        test_document_packing()
        print("\n" + "=" * 60)
        print("Todos los tests pasaron.")
        return

    args = parse_args()

    if args.validate:
        builder = DistillationDatasetBuilder(
            logprobs_dir=args.logprobs_dir or ".",
            output_dir=args.output_dir,
            seq_len=args.seq_len,
        )
        builder.validate_sample()
        return

    if args.logprobs_dir is None:
        print("Error: --logprobs-dir requerido")
        return

    builder = DistillationDatasetBuilder(
        logprobs_dir=args.logprobs_dir,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
    )
    builder.build()




# =============================================================================
# Section 2: Dataset Pipeline
# =============================================================================

# ============================================================
# Configuration
# ============================================================

TEACHER_MODEL = "Qwen/Qwen2.5-3B"

# Module-level cache: tokenizer id → {token_id: bytes}
_TOKEN_BYTES_CACHE: dict = {}
DATASET_REPO = "Cukinator/cpu1-ablation-dataset"
DATASET_BUCKET = "Cukinator/cpu1-ablation-dataset"  # HF Bucket (read-write via hf-mount)
OUTPUT_DIR = Path("./shared_dataset")
TOP_K_LOGPROBS = 128
PATCH_SIZE = 4
SEQ_LEN_PATCHES = 5000
SEQ_LEN_TOKENS = 5000
MAX_EXTRACTION_TOKENS = 700  # 700 max tokens per doc → shorter seqs = bigger batch
TEACHER_HIDDEN_DIM = 2048   # Qwen2.5-3B hidden size (2048); -7B seria 3584
ALIGN_DIM = 128             # Project teacher hidden 2048→128 (Johnson-Lindenstrauss)
SEQS_PER_SHARD = 50         # Align with 500-doc chunks to upload immediately
SEED = 42
BATCH_SIZE_TEACHER = 29    # T4 optimal: avoids OOM cascade (was 120→57→42→31)
BATCH_SIZE_TEACHER_L4 = 100   # L4 24GB: FP8 model ~7GB + attn@4096. Restored to 100.
MAX_EXTRACTION_TOKENS_L4 = 4096  # L4: 4k seq fits at batch 14 (bucket batching cuts avg)

# --- Concurrency & Resource Control ---
TARGET_VRAM_GB = 13.0              # Hard cap: 13GB per T4 GPU (15GB total, 2GB headroom)
VRAM_SAFETY_FACTOR = 0.96          # Aggressive: OOM-halving fallback is the real safety net
RAM_HARD_LIMIT_GB = 25.0           # Hard cap: never exceed 25GB RAM total
CALIBRATION_SAMPLES = 30           # More samples = accurate per-doc VRAM (overhead diluted)
CALIBRATION_WARMUP = 2             # Warm-up passes before measuring (eliminates GPU cold-start bias)
# GPU buffer: how many chunks GPU can run ahead of CPU before blocking.
# Each queued chunk is just a tuple — actual data is on /tmp disk (57GB available).
EXTRACTION_QUEUE_DEPTH = 10        # GPU can queue 10 chunks ahead (was 5)
UPLOAD_QUEUE_DEPTH = 10            # CPU can queue 10 shards for upload (was 6)
MAX_ACCUMULATED_PATCHES = 50_000   # ~10× SEQ_LEN_PATCHES, larger buffer with flattened reads
MAX_ACCUMULATED_TOKENS = 50_000    # ~10× SEQ_LEN_TOKENS, larger buffer with flattened reads
MODEL_RESET_EVERY_CHUNKS = 30      # Reload models to defragment VRAM (was 20, extended)
CORPUS_PROGRESS_INTERVAL = 10_000  # Print progress every N docs during download
RAM_HEADROOM_FRACTION = 0.17       # Keep 17% RAM free (25/30 = 83% usable)
CHECKPOINT_EVERY_SHARDS = 3        # Checkpoint+upload every 3 shards (batch uploads for efficiency)
UPLOAD_EVERY_SHARDS = 3            # Upload byte+BPE shard files to HF every N complete shards


def _get_ram_usage():
    """Return (used_gb, total_gb, fraction_used)."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.used / 1024**3, mem.total / 1024**3, mem.percent / 100.0
    except ImportError:
        try:
            with open('/proc/meminfo') as f:
                lines = f.readlines()
            info = {}
            for line in lines:
                parts = line.split()
                info[parts[0].rstrip(':')] = int(parts[1]) * 1024
            total = info.get('MemTotal', 0)
            avail = info.get('MemAvailable', info.get('MemFree', 0))
            used = total - avail
            return used / 1024**3, total / 1024**3, used / max(total, 1)
        except Exception:
            return 0, 0, 0


def _check_ram_pressure(label=""):
    """Force GC if RAM headroom < RAM_HEADROOM_FRACTION. Returns True if pressure high."""
    used_gb, total_gb, frac = _get_ram_usage()
    headroom = 1.0 - frac
    if headroom < RAM_HEADROOM_FRACTION:
        gc.collect()
        used_gb2, _, frac2 = _get_ram_usage()
        print(f"  [RAM {label}] Pressure! {frac*100:.0f}%→{frac2*100:.0f}% "
              f"({used_gb:.1f}→{used_gb2:.1f}/{total_gb:.1f}GB) — forced GC", flush=True)
        return True
    return False


def detect_gpu_profile() -> dict:
    """Detect GPU and return optimal extraction config.

    Profiles:
        L4  (24GB, Ada Lovelace): FP8 W8A8 dynamic, batch 14, seq 4096, bucket sorted
        T4  (15GB, Turing):       4-bit NF4,        batch 2,  seq 2048
        other:                    fallback based on VRAM

    Returns dict with keys: name, vram_gb, batch_size, max_tokens, fp8, dtype
    """
    if not torch.cuda.is_available():
        return {
            'name': 'cpu', 'vram_gb': 0,
            'batch_size': 1, 'max_tokens': MAX_EXTRACTION_TOKENS,
            'fp8': False, 'dtype': torch.float32,
        }

    props = torch.cuda.get_device_properties(0)
    name = props.name.lower()
    vram_gb = props.total_memory / (1024 ** 3)

    if 'l4' in name or (vram_gb >= 22 and vram_gb < 28):
        # L4: Ada Lovelace, FP8 W8A8 via torch._scaled_mm, 24 GB GDDR6, 4 CPU cores
        # logit_chunk_size=512: 14×512×3584 BF16 = ~49 MB/chunk
        return {
            'name': props.name, 'vram_gb': vram_gb,
            'batch_size': BATCH_SIZE_TEACHER_L4,
            'max_tokens': MAX_EXTRACTION_TOKENS_L4,
            'fp8': True, 'dtype': torch.bfloat16,
            'logit_chunk_size': 512,
            'cpu_workers': 4,
        }
    elif 't4' in name or (vram_gb >= 13 and vram_gb < 18):
        # T4: 15GB, cap at 13GB. With use_cache=False, 700 tok max:
        #   NF4 model: ~3 GB
        #   Attention [B,H,S,S] FP16: 120×28×700²×2B = ~3.3 GB
        #   Hidden hooks: 120×700×2048×2B×2 = ~0.7 GB
        #   logit chunk [120,64,151643] FP32: 120×64×151643×4B = ~4.4 GB
        #   Total peak: ~11.4 GB → under 13GB cap (calibrator will adjust)
        return {
            'name': props.name, 'vram_gb': vram_gb,
            'batch_size': BATCH_SIZE_TEACHER,
            'max_tokens': MAX_EXTRACTION_TOKENS,
            'fp8': False, 'dtype': torch.float16,
            'logit_chunk_size': 64,  # 4x larger chunks → fewer kernel launches, better occupancy
            'cpu_workers': 2,
        }
    elif 'a100' in name or 'h100' in name:
        return {
            'name': props.name, 'vram_gb': vram_gb,
            'batch_size': 16, 'max_tokens': 4096,
            'fp8': False, 'dtype': torch.bfloat16,
            'logit_chunk_size': 512,
            'cpu_workers': 4,
        }
    else:
        batch = max(1, int(vram_gb / 8))
        dtype = torch.bfloat16 if vram_gb >= 20 else torch.float16
        cpu_w = min(4, max(1, os.cpu_count() or 2))
        return {
            'name': props.name, 'vram_gb': vram_gb,
            'batch_size': batch, 'max_tokens': MAX_EXTRACTION_TOKENS,
            'fp8': False, 'dtype': dtype,
            'logit_chunk_size': 256,
            'cpu_workers': cpu_w,
        }


# Per-GPU calibration cache: avoids re-calibrating every chunk (~10s idle per call)
_CALIBRATED_BATCH_CACHE: dict[int, int] = {}  # device_index → batch_size


def _calibrate_batch_size(model, tokenizer, profile: dict,
                          target_vram_gb: float = TARGET_VRAM_GB) -> int:
    """Profile VRAM with a small calibration batch and calculate optimal batch size.

    Uses per-GPU cache: first call runs actual calibration (~5-10s),
    subsequent calls return cached result instantly. Cache is cleared
    by _reset_models() so batch sizes are recalibrated after defrag.

    Returns:
        Optimal batch_size that keeps peak VRAM at target_vram_gb.
    """
    device = next(model.parameters()).device
    dev_idx = device.index if (hasattr(device, 'index') and device.index is not None) else 0

    # Return cached result if available (saves ~10s per call)
    if dev_idx in _CALIBRATED_BATCH_CACHE:
        cached = _CALIBRATED_BATCH_CACHE[dev_idx]
        print(f"[Calibrate] GPU {dev_idx}: using cached batch_size={cached}")
        return cached

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    gc.collect()

    # Use texts that match the actual workload: ~max_tokens length (not ~50 tokens).
    # Short texts massively underestimate per-doc VRAM (no KV cache pressure, tiny
    # activations). A 768-token doc uses ~10× more VRAM than a 50-token doc.
    _chars_per_token = 4  # conservative for English text
    _cal_len = profile.get('max_tokens', MAX_EXTRACTION_TOKENS) * _chars_per_token
    cal_text = ("The quick brown fox jumps over the lazy dog. " * 200)[:_cal_len]
    cal_texts = [cal_text] * CALIBRATION_SAMPLES

    seq_len = profile.get('max_tokens', MAX_EXTRACTION_TOKENS)
    logit_chunk = profile.get('logit_chunk_size', 16)

    # Warm-up pass: eliminate CUDA cold-start bias (first run always allocates more)
    for _ in range(CALIBRATION_WARMUP):
        try:
            _ = extract_logprobs_and_hidden_batch(
                model, tokenizer, cal_texts[:1],
                top_k=TOP_K_LOGPROBS, seq_len=seq_len,
                logit_chunk_size=logit_chunk,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect(); break
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    gc.collect()

    baseline_vram = torch.cuda.memory_allocated(device) / (1024**3)

    try:
        _ = extract_logprobs_and_hidden_batch(
            model, tokenizer, cal_texts,
            top_k=TOP_K_LOGPROBS, seq_len=seq_len,
            logit_chunk_size=logit_chunk,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        print(f"[Calibrate] WARNING: OOM on {CALIBRATION_SAMPLES} docs! Using batch_size=1")
        _CALIBRATED_BATCH_CACHE[dev_idx] = 1
        return 1

    peak_vram = torch.cuda.max_memory_allocated(device) / (1024**3)
    vram_per_doc = (peak_vram - baseline_vram) / CALIBRATION_SAMPLES

    # (target - model_baseline) * safety / per_doc
    available = (target_vram_gb - baseline_vram) * VRAM_SAFETY_FACTOR
    if vram_per_doc > 0:
        optimal_batch = max(1, int(available / vram_per_doc))
    else:
        optimal_batch = profile['batch_size']

    # Cap at 4× profile default to allow aggressive VRAM usage
    optimal_batch = min(optimal_batch, profile['batch_size'] * 4)

    torch.cuda.empty_cache()
    gc.collect()

    _CALIBRATED_BATCH_CACHE[dev_idx] = optimal_batch
    print(f"[Calibrate] GPU {dev_idx}: baseline={baseline_vram:.2f}GB  "
          f"peak={peak_vram:.2f}GB  per_doc={vram_per_doc*1000:.1f}MB  "
          f"→ batch_size={optimal_batch}")
    return optimal_batch


def _fast_download_corpus(corpus_name: str, n_docs: int,
                          hf_token: str = None) -> list[str]:
    """Download corpus with optimized streaming for fast cold start.

    Optimizations vs naive iteration:
    - Authenticated requests via HF_TOKEN (higher rate limits, priority CDN)
    - No shuffle during download (shuffle in-memory after — much faster)
    - Progress tracking with ETA
    - Filter empty docs during download to avoid wasted GPU work
    """
    from datasets import load_dataset

    t0 = time.time()
    print(f"[Corpus] Downloading {n_docs:,} docs from {corpus_name}...")

    kwargs: dict = {'streaming': True}
    if hf_token:
        kwargs['token'] = hf_token
        print(f"[Corpus] Using HF_TOKEN for authenticated download (faster CDN)")

    fw = load_dataset(corpus_name, name="sample-10BT", split="train", **kwargs)

    texts: list[str] = []
    skipped = 0
    for i, doc in enumerate(fw):
        if i >= n_docs:
            break
        text = doc.get('text', '')
        if text.strip():
            texts.append(text)
        else:
            skipped += 1
        if (i + 1) % CORPUS_PROGRESS_INTERVAL == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_docs - i - 1) / rate if rate > 0 else 0
            print(f"  [Corpus] {i+1:,}/{n_docs:,} docs "
                  f"({rate:.0f} docs/s, ETA {eta:.0f}s)", flush=True)

    # Shuffle in-memory (instantaneous vs streaming shuffle's buffer overhead)
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(texts))
    texts = [texts[j] for j in idx]

    elapsed = time.time() - t0
    print(f"[Corpus] {len(texts):,} docs in {elapsed:.1f}s "
          f"({len(texts)/max(elapsed,0.1):.0f} docs/s, {skipped} empty skipped)")
    return texts


def _reset_models(models: list, profile: dict) -> list:
    """Reload all teacher models to defragment CUDA memory.

    After thousands of batches, CUDA memory becomes fragmented —
    allocated+freed blocks leave gaps that can't satisfy large contiguous
    allocations (like logit tensors). Reloading the model forces a clean
    CUDA memory layout.

    Takes ~30-60 seconds per GPU. Call every MODEL_RESET_EVERY_CHUNKS chunks.
    Clears calibration cache so batch sizes are recalculated with fresh VRAM.

    Args:
        models: list of (model, tokenizer) tuples
        profile: GPU profile dict

    Returns:
        New list of (model, tokenizer) tuples with fresh CUDA allocations.
    """
    global _CALIBRATED_BATCH_CACHE
    _CALIBRATED_BATCH_CACHE.clear()  # Force recalibration after defrag
    print(f"\n[Reset] Reloading {len(models)} model(s) to defragment VRAM...")
    t0 = time.time()

    # Phase 1: Delete all models and free CUDA memory
    for i, (model, _tok) in enumerate(models):
        dev = next(model.parameters()).device
        del model
    models.clear()
    gc.collect()
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(i)

    # Phase 2: Reload fresh models
    new_models = []
    for gpu_id in range(torch.cuda.device_count()):
        dev = f"cuda:{gpu_id}"
        m, tok = load_teacher(device=dev, profile=profile)
        torch.cuda.empty_cache()
        new_models.append((m, tok))

    elapsed = time.time() - t0
    vram_info = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        vram_info.append(f"GPU{i}: {(total-free)/1024**3:.1f}/{total/1024**3:.1f}GB")
    print(f"[Reset] Done in {elapsed:.1f}s. VRAM: {', '.join(vram_info)}")
    return new_models


def _check_vram_fragmentation(device: int = 0) -> float:
    """Return VRAM fragmentation ratio (0.0 = clean, 1.0 = fully fragmented).

    Fragmentation = (reserved - allocated) / reserved.
    High fragmentation means the allocator has many small free gaps
    between allocated blocks, making large contiguous allocations fail.
    """
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    if reserved == 0:
        return 0.0
    return (reserved - allocated) / reserved


# Lazy-initialized random orthogonal projection matrix (CPU) and per-device GPU cache
_PROJECTION_MATRIX = None
_PROJECTION_MATRIX_GPU: dict = {}  # device_str → GPU tensor, avoids re-transfer every batch

def _get_projection_matrix(teacher_dim: int, align_dim: int = ALIGN_DIM):
    """Random orthogonal projection, seeded for reproducibility."""
    global _PROJECTION_MATRIX
    if _PROJECTION_MATRIX is None:
        rng = np.random.RandomState(SEED)
        rand_mat = rng.randn(teacher_dim, align_dim).astype(np.float32)
        q, _ = np.linalg.qr(rand_mat)
        _PROJECTION_MATRIX = torch.from_numpy(q[:, :align_dim])
    return _PROJECTION_MATRIX


def _get_projection_matrix_gpu(device, dtype):
    """Return projection matrix cached on the given device+dtype. Transfers once per run."""
    key = (str(device), dtype)
    if key not in _PROJECTION_MATRIX_GPU:
        cpu_proj = _get_projection_matrix(TEACHER_HIDDEN_DIM)
        _PROJECTION_MATRIX_GPU[key] = cpu_proj.to(device=device, dtype=dtype)
    return _PROJECTION_MATRIX_GPU[key]


# ============================================================
# STEP 1: Extract teacher logprobs
# ============================================================

def _load_teacher_fp8(model_name: str, device: str, dtype=torch.bfloat16):
    """Load teacher in BF16 then apply FP8 dynamic-activation quantization via torchao.

    Uses Float8DynamicActivationFloat8WeightConfig (W8A8 FP8):
      - Weights AND activations in FP8 E4M3 during matmuls
      - Activations quantized dynamically per-token at runtime
      - Fires L4 Ada Lovelace FP8 tensor cores on every linear layer (attention + MLP)
      - Weights ~7 GB; activations quantized on-the-fly (no extra VRAM)
      - Requires torchao >= 0.4 and PyTorch >= 2.5 with CUDA 12.x

    Fallback chain:
      torchao W8A8  →  torchao W8A16 (weight-only, no tensor cores)  →  plain BF16
    """
    from transformers import AutoModelForCausalLM

    print(f"[Teacher] Loading {model_name} in BF16 on {device} for FP8 quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map={"": device},
    )

    try:
        # Validate runtime prerequisites for FP8 tensor core path
        if torch.version.cuda and tuple(int(x) for x in torch.version.cuda.split('.')) < (12, 0):
            print(f"[Teacher] WARNING: CUDA {torch.version.cuda} < 12.0 — "
                  "FP8 tensor cores require CUDA 12.x. Falling back to BF16.")
            model.eval()
            return model
        from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
        print("[Teacher] Applying FP8 W8A8 dynamic quantization (torchao) ...")
        quantize_(model, Float8DynamicActivationFloat8WeightConfig())
        print("[Teacher] FP8 W8A8 applied — L4 Ada FP8 tensor cores active on all linears. "
              "Weights ~7 GB, activations quantized per-token at runtime.")
    except (ImportError, AttributeError):
        # Float8DynamicActivationFloat8WeightConfig requires torchao >= 0.4
        try:
            from torchao.quantization import quantize_, Float8WeightOnlyConfig
            print("[Teacher] torchao < 0.4 detected — falling back to FP8 W8A16 "
                  "(weight-only, NO tensor cores, memory savings only). "
                  "Upgrade with: pip install -U torchao")
            quantize_(model, Float8WeightOnlyConfig())
        except ImportError:
            print("[Teacher] torchao not found — running BF16 (~14 GB). "
                  "Install: pip install torchao")

    model.eval()
    return model


def load_teacher(model_name: str = TEACHER_MODEL, device: str = "cuda:0",
                 profile: dict = None):
    """Load Qwen-7B with the optimal quantization for the detected GPU.

    - T4  (15 GB): 4-bit NF4 (~4 GB VRAM) — original Kaggle path
    - L4  (24 GB): BF16 + FP8 weights via torchao (~7 GB VRAM, batch×4 throughput)
    - CPU:         4-bit NF4 via bitsandbytes CPU path
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if profile is None:
        profile = detect_gpu_profile()

    print(f"[Teacher] GPU: {profile['name']}  ({profile['vram_gb']:.1f} GB VRAM)")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if profile['fp8'] and device != 'cpu':
        model = _load_teacher_fp8(model_name, device, profile['dtype'])
    else:
        # Original 4-bit NF4 path (T4 / CPU / unknown GPU)
        compute_dtype = profile.get('dtype', torch.float16)
        print(f"[Teacher] Loading {model_name} in 4-bit NF4 on {device}...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map={"": device},
        )
        model.eval()

    # Cache tokenizer locally so BPE ProcessPoolExecutor workers don't hit network
    local_cache = Path("/tmp/_bpe_tokenizer_cache")
    if not local_cache.exists():
        tokenizer.save_pretrained(str(local_cache))
    print(f"[Teacher] Ready on {device}. Vocab size: {tokenizer.vocab_size}")
    return model, tokenizer


def extract_logprobs_single(
    model,
    tokenizer,
    text: str,
    top_k: int = TOP_K_LOGPROBS,
    seq_len: int = SEQ_LEN_TOKENS,
) -> dict:
    """
    Extract top-k logprobs for every position of a single text.

    Returns dict with:
        text_bytes:  list[int]   — UTF-8 bytes of original text
        positions:   list[dict]  — per-position top-k logprobs
        n_tokens:    int
    """
    enc = tokenizer(
        text, return_tensors='pt',
        truncation=True, max_length=seq_len,
        return_offsets_mapping=True,
    )
    input_ids = enc['input_ids'].to(model.device)

    with torch.no_grad():
        logits = model(input_ids).logits[0]  # [seq_len, vocab_size]

    log_probs = torch.log_softmax(logits, dim=-1)
    top_logprobs, top_indices = torch.topk(log_probs, k=top_k, dim=-1)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    offsets = enc['offset_mapping'][0].tolist()

    for pos_idx, (token_id, token_str) in enumerate(
        zip(input_ids[0].tolist(), tokens)
    ):
        start_char, end_char = offsets[pos_idx]
        byte_offset = len(text[:start_char].encode('utf-8'))

        pos_top = []
        for k_idx in range(top_k):
            cand_id = top_indices[pos_idx, k_idx].item()
            cand_lp = top_logprobs[pos_idx, k_idx].item()
            cand_tok = tokenizer.convert_ids_to_tokens([cand_id])[0]
            try:
                cand_bytes = list(
                    tokenizer.convert_tokens_to_string(
                        [cand_tok]
                    ).encode('utf-8')
                )
            except Exception:
                cand_bytes = [0]

            pos_top.append({
                'token_id': cand_id,
                'token': cand_tok,
                'bytes': cand_bytes,
                'logprob': cand_lp,
            })

        positions.append({
            'byte_offset': byte_offset,
            'top_tokens': pos_top,
        })

    return {
        'text_bytes': list(text.encode('utf-8')),
        'positions': positions,
        'n_tokens': len(tokens),
    }


def extract_logprobs_and_hidden(
    model,
    tokenizer,
    text: str,
    top_k: int = TOP_K_LOGPROBS,
    seq_len: int = SEQ_LEN_TOKENS,
    hidden_layer: int = -1,  # -1 = capa del medio, o específica
) -> dict:
    """
    Versión extendida de extract_logprobs_single que también guarda
    los hidden states del teacher en la capa del medio.

    El hidden state se usa para EmbeddingAligner en Fase 1.

    Costo adicional: ~50% más memoria por batch (activaciones intermedias).
    Workaround si la memoria es justa: extraer en un segundo pass separado.

    Output adicional por posición:
        'teacher_hidden': list[list[float]]  # [n_tokens, teacher_dim] FP16
    """
    enc = tokenizer(
        text, return_tensors='pt',
        truncation=True, max_length=seq_len,
        return_offsets_mapping=True,
    )
    input_ids = enc['input_ids'].to(model.device)

    with torch.no_grad():
        outputs = model(
            input_ids,
            output_hidden_states=True,
        )
        logits = outputs.logits[0]  # [seq_len, vocab_size]

        # Hidden state de la capa del medio
        n_layers = len(outputs.hidden_states)
        mid_layer = hidden_layer if hidden_layer >= 0 else n_layers // 2
        teacher_hidden = outputs.hidden_states[mid_layer][0]  # [S, D]

        # Project 5120→ALIGN_DIM and compress to FP16
        proj = _get_projection_matrix(teacher_hidden.shape[-1])
        teacher_hidden_proj = (teacher_hidden.float().cpu() @ proj).half()

    log_probs = torch.log_softmax(logits, dim=-1)
    top_logprobs, top_indices = torch.topk(log_probs, k=top_k, dim=-1)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    offsets = enc['offset_mapping'][0].tolist()

    for pos_idx, (token_id, token_str) in enumerate(
        zip(input_ids[0].tolist(), tokens)
    ):
        start_char, end_char = offsets[pos_idx]
        byte_offset = len(text[:start_char].encode('utf-8'))

        pos_top = []
        for k_idx in range(top_k):
            cand_id = top_indices[pos_idx, k_idx].item()
            cand_lp = top_logprobs[pos_idx, k_idx].item()
            cand_tok = tokenizer.convert_ids_to_tokens([cand_id])[0]
            try:
                cand_bytes = list(
                    tokenizer.convert_tokens_to_string(
                        [cand_tok]
                    ).encode('utf-8')
                )
            except Exception:
                cand_bytes = [0]

            pos_top.append({
                'token_id': cand_id,
                'token': cand_tok,
                'bytes': cand_bytes,
                'logprob': cand_lp,
            })

        positions.append({
            'byte_offset': byte_offset,
            'top_tokens': pos_top,
        })

    return {
        'text_bytes': list(text.encode('utf-8')),
        'positions': positions,
        'n_tokens': len(tokens),
        'teacher_hidden': teacher_hidden_proj.tolist(),  # [n_tokens, ALIGN_DIM]
        'teacher_dim': teacher_hidden_proj.shape[-1],
    }


def _get_token_bytes_cache(tokenizer) -> dict:
    """Build (once per tokenizer) a full vocab id → UTF-8 bytes lookup table."""
    key = id(tokenizer)
    if key not in _TOKEN_BYTES_CACHE:
        print("[BatchExtract] Building token→bytes cache (~1 s)...")
        vocab_size = tokenizer.vocab_size
        all_tokens = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
        cache: dict = {}
        for tid, tok in enumerate(all_tokens):
            if tok is None:
                cache[tid] = b'\x00'
            else:
                try:
                    cache[tid] = tokenizer.convert_tokens_to_string([tok]).encode('utf-8')
                except Exception:
                    cache[tid] = b'\x00'
        _TOKEN_BYTES_CACHE[key] = cache
        print(f"[BatchExtract] Cache ready: {vocab_size} tokens")
    return _TOKEN_BYTES_CACHE[key]


def extract_logprobs_and_hidden_batch(
    model,
    tokenizer,
    texts: list[str],
    top_k: int = TOP_K_LOGPROBS,
    seq_len: int = MAX_EXTRACTION_TOKENS,
    hidden_layer: int = -1,
    logit_chunk_size: int = 256,
) -> list[dict]:
    """
    Batched version of extract_logprobs_and_hidden.

    Processes a list of texts in one forward pass for better GPU utilisation.

    Memory optimisations vs naïve batch:
    - Forward hooks instead of output_hidden_states=True → stores 2 layers, not 28
    - Chunked lm_head (logit_chunk_size positions per step) → avoids B×S×151k tensor
    - Pre-built token→bytes cache → no per-token tokenizer calls at build time

    Returns list[dict] with the same schema as extract_logprobs_and_hidden.
    """
    if not texts:
        return []

    bytes_cache = _get_token_bytes_cache(tokenizer)
    n_layers    = model.config.num_hidden_layers
    mid_idx     = hidden_layer if hidden_layer >= 0 else n_layers // 2

    # ── Register hooks to capture mid-layer and last-layer hidden states ──
    # IMPORTANT: .clone() is required — without it the detached tensor still
    # holds a reference to the full forward-pass memory (activations, KV cache)
    # preventing CUDA from freeing those buffers after the forward call.
    captured: dict = {}

    def _mid_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured['mid'] = h.detach().clone()   # [B, S, D] — own allocation, no lingering refs

    def _last_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured['last'] = h.detach().clone()  # [B, S, D] — own allocation, no lingering refs

    hook_mid  = model.model.layers[mid_idx].register_forward_hook(_mid_hook)
    hook_last = model.model.layers[-1].register_forward_hook(_last_hook)

    try:
        enc = tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=seq_len,
            return_offsets_mapping=True,
        )
        input_ids = enc['input_ids'].to(model.device)        # [B, S_pad]
        attn_mask = enc['attention_mask'].to(model.device)   # [B, S_pad]
        offset_mapping_cpu = enc['offset_mapping']

        with torch.no_grad():
            # use_cache=False prevents the model from retaining the KV cache in VRAM
            # after the forward pass. Without this, Qwen2.5-3B at batch=8, seq=2048
            # allocates ~3-4 GB of KV cache that is never freed until the next call.
            model(input_ids, attention_mask=attn_mask, use_cache=False)

    finally:
        hook_mid.remove()
        hook_last.remove()

    B, S_pad = input_ids.shape

    # ── Project mid-layer hidden → ALIGN_DIM (on GPU to avoid CPU bottleneck) ─
    # Projection on GPU: BF16 matmul [B,S,3584]×[3584,128] → ~14 MB result.
    # Old path moved 800 MB to CPU for a float32 matmul; this is 50× less data
    # and fires BF16 tensor cores instead of CPU SIMD.
    proj_gpu = _get_projection_matrix_gpu(captured['mid'].device, captured['mid'].dtype)
    with torch.no_grad():
        mid_proj = (captured['mid'] @ proj_gpu).cpu()  # [B, S_pad, ALIGN_DIM] — stays BF16
    del captured['mid']
    torch.cuda.empty_cache()  # free mid-layer GPU buffer before logit computation

    # ── Chunked logit computation from last-layer hidden + final norm ─────
    # Apply norm on GPU then immediately move to CPU to free VRAM before chunked logits.
    with torch.no_grad():
        last_normed = model.model.norm(captured['last'])   # [B, S_pad, D]
    del captured['last']
    torch.cuda.empty_cache()  # free last-layer raw buffer before lm_head chunks

    all_top_idx: list = []
    all_top_lp:  list = []
    with torch.no_grad():
        for cs in range(0, S_pad, logit_chunk_size):
            ce = min(cs + logit_chunk_size, S_pad)
            chunk  = last_normed[:, cs:ce, :].contiguous()   # [B, C, D]
            logits = model.lm_head(chunk).float()             # [B, C, V]
            log_p  = torch.log_softmax(logits, dim=-1)
            top_lp, top_idx = torch.topk(log_p, k=top_k, dim=-1)
            del logits, log_p
            # .cpu() implicitly synchronizes — no explicit sync needed
            all_top_idx.append(top_idx.cpu())
            all_top_lp.append(top_lp.cpu())

    top_idx_all = torch.cat(all_top_idx, dim=1)   # [B, S_pad, K]
    top_lp_all  = torch.cat(all_top_lp,  dim=1)   # [B, S_pad, K]

    del last_normed, all_top_idx, all_top_lp

    input_ids_cpu = input_ids.cpu()
    attn_mask_cpu = attn_mask.cpu()
    del input_ids, attn_mask

    # ── Build one result dict per sample ──────────────────────────────────
    results = []
    for b in range(B):
        actual_len  = int(attn_mask_cpu[b].sum().item())
        ids         = input_ids_cpu[b, :actual_len].tolist()
        offsets     = offset_mapping_cpu[b].tolist()

        # Pre-compute char→byte offset map: O(n) instead of O(n²) per-position encoding
        _txt = texts[b]
        _txt_bytes = _txt.encode('utf-8')
        _cum_byte = [0] * (len(_txt) + 1)
        _bi = 0
        for _ci, _ch in enumerate(_txt):
            _cum_byte[_ci] = _bi
            _bi += len(_ch.encode('utf-8'))
        _cum_byte[len(_txt)] = _bi

        positions   = []
        for pos_idx, token_id in enumerate(ids):
            start_char, end_char = offsets[pos_idx]
            byte_offset = _cum_byte[start_char] if start_char < len(_cum_byte) else len(_txt_bytes)

            pos_top = []
            for k_idx in range(top_k):
                cand_id = top_idx_all[b, pos_idx, k_idx].item()
                cand_lp = top_lp_all[b,  pos_idx, k_idx].item()
                pos_top.append({
                    'token_id': cand_id,
                    'token':    '',   # not used downstream
                    'bytes':    list(bytes_cache.get(cand_id, b'\x00')),
                    'logprob':  cand_lp,
                })

            positions.append({
                'byte_offset': byte_offset,
                'top_tokens':  pos_top,
            })

        hidden_proj = mid_proj[b, :actual_len, :]   # [L, ALIGN_DIM]
        results.append({
            'text_bytes':    list(_txt_bytes),
            'positions':     positions,
            'n_tokens':      actual_len,
            'teacher_hidden': hidden_proj.tolist(),
            'teacher_dim':   int(hidden_proj.shape[-1]),
        })

    return results


def _append_to_parquet(records: list[dict], path: Path) -> None:
    """Append records to a Parquet file (legacy, used by monkey-patch)."""
    columns = {
        'doc_id':     pa.array([r['doc_id'] for r in records], type=pa.int32()),
        'text_bytes': pa.array([r['text_bytes'] for r in records]),
        'positions':  pa.array([r['positions'] for r in records], type=pa.string()),
        'n_tokens':   pa.array([r['n_tokens'] for r in records], type=pa.int32()),
    }
    if records and 'teacher_hidden' in records[0]:
        columns['teacher_hidden'] = pa.array(
            [r['teacher_hidden'] for r in records],
            type=pa.list_(pa.float32()),
        )
        columns['teacher_dim'] = pa.array(
            [r['teacher_dim'] for r in records], type=pa.int32(),
        )
    table = pa.table(columns)
    if path.exists():
        existing = pq.read_table(str(path))
        combined = pa.concat_tables([existing, table])
        pq.write_table(combined, str(path))
    else:
        pq.write_table(table, str(path))


def _write_parquet_part(records: list[dict], part_path: Path) -> None:
    """Write records to a NEW parquet part file — O(1), no read-merge.

    Used by extraction workers instead of _append_to_parquet to avoid
    O(n²) read-concat-write and to eliminate lock contention between GPUs.
    """
    columns = {
        'doc_id':     pa.array([r['doc_id'] for r in records], type=pa.int32()),
        'text_bytes': pa.array([r['text_bytes'] for r in records]),
        'positions':  pa.array([r['positions'] for r in records], type=pa.string()),
        'n_tokens':   pa.array([r['n_tokens'] for r in records], type=pa.int32()),
    }
    if records and 'teacher_hidden' in records[0]:
        columns['teacher_hidden'] = pa.array(
            [r['teacher_hidden'] for r in records],
            type=pa.large_binary(),
        )
        columns['teacher_dim'] = pa.array(
            [r['teacher_dim'] for r in records], type=pa.int32(),
        )
    pq.write_table(pa.table(columns), str(part_path))


def _merge_parquet_parts(output_path: Path) -> None:
    """Merge all _partNNNN.parquet files into a single output parquet."""
    parts = sorted(output_path.parent.glob(f"{output_path.stem}_part*.parquet"))
    if not parts:
        return
    tables = [pq.read_table(str(p)) for p in parts]
    combined = pa.concat_tables(tables)
    pq.write_table(combined, str(output_path))
    for p in parts:
        p.unlink()
    print(f"  [merge] {len(parts)} parts → {output_path.name} ({len(combined)} docs)")


def run_logprob_extraction(
    corpus_texts: list[str],
    output_path: str,
    resume: bool = True,
    preloaded_models: list = None,
) -> None:
    """Extract logprobs + projected hidden states for entire corpus.

    GPU configs:
      Kaggle  2×T4 (15 GB each): 4-bit NF4, batch 2 per GPU, 2× parallel throughput
      Lightning AI 1×L4 (24 GB): FP8 W8A8 via torchao, batch 14, Ada FP8 tensor cores

    Always extracts teacher hidden states (projected 5120→ALIGN_DIM)
    for EmbeddingAligner (runs 13, 15).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_idx = 0
    if resume and output_path.exists():
        existing = pq.read_table(str(output_path))
        start_idx = len(existing)
        print(f"[Extraction] Resuming from doc {start_idx}/{len(corpus_texts)}")

    if start_idx >= len(corpus_texts):
        print("[Extraction] All docs already extracted")
        return

    profile = detect_gpu_profile()
    batch_size       = profile['batch_size']
    seq_len          = profile['max_tokens']
    logit_chunk_size = profile.get('logit_chunk_size', 256)
    print(f"[Extraction] Profile: {profile['name']}  "
          f"batch={batch_size}  seq={seq_len}  "
          f"logit_chunk={logit_chunk_size}  fp8={profile['fp8']}")

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if n_gpus >= 2:
        print(f"[Extraction] Using {n_gpus} GPUs in parallel "
              f"({', '.join(torch.cuda.get_device_name(i) for i in range(n_gpus))})")
        _run_extraction_multi_gpu(corpus_texts, output_path, start_idx, n_gpus,
                                  preloaded_models, batch_size=batch_size,
                                  seq_len=seq_len, logit_chunk_size=logit_chunk_size)
    else:
        _run_extraction_single_gpu(corpus_texts, output_path, start_idx,
                                   batch_size=batch_size, seq_len=seq_len,
                                   logit_chunk_size=logit_chunk_size,
                                   preloaded_models=preloaded_models)

    # Only free VRAM if models were loaded locally (not pre-loaded for pipeline)
    if preloaded_models is None:
        gc.collect()
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
            vram_free = [torch.cuda.mem_get_info(i)[0] / (1024**3) for i in range(torch.cuda.device_count())]
            print(f"[Extraction] VRAM freed. Available: {['%.1fG' % v for v in vram_free]}")

    print(f"[Extraction] Complete: {output_path}")


def _extract_batch_with_oom_retry(model, tokenizer, batch_texts, gpu_tag="GPU 0",
                                   seq_len: int = MAX_EXTRACTION_TOKENS,
                                   logit_chunk_size: int = 256):
    """Try extraction, halve batch on OOM until batch_size=1."""
    import traceback
    sub_batches = [batch_texts]
    while sub_batches:
        texts = sub_batches.pop(0)
        try:
            return extract_logprobs_and_hidden_batch(
                model, tokenizer, texts,
                top_k=TOP_K_LOGPROBS, seq_len=seq_len,
                logit_chunk_size=logit_chunk_size,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            if len(texts) > 1:
                mid = len(texts) // 2
                print(f"\n  [{gpu_tag}] OOM with {len(texts)} docs, splitting to {mid}+{len(texts)-mid}")
                r1 = _extract_batch_with_oom_retry(model, tokenizer, texts[:mid], gpu_tag,
                                                    seq_len, logit_chunk_size)
                r2 = _extract_batch_with_oom_retry(model, tokenizer, texts[mid:], gpu_tag,
                                                    seq_len, logit_chunk_size)
                return (r1 or []) + (r2 or [])
            else:
                print(f"\n  [{gpu_tag}] OOM on single doc, skipping")
                traceback.print_exc()
                return []
        except Exception:
            print(f"\n  [{gpu_tag}] ERROR:")
            traceback.print_exc()
            torch.cuda.empty_cache()
            return []
    return []


def _run_extraction_single_gpu(corpus_texts, output_path, start_idx,
                                batch_size: int = BATCH_SIZE_TEACHER,
                                seq_len: int = MAX_EXTRACTION_TOKENS,
                                logit_chunk_size: int = 256,
                                preloaded_models=None):
    """Single-GPU extraction with bucket batching and adaptive OOM recovery.

    Bucket batching:
        Docs sorted by estimated token length so each batch has similar-length docs.
        Average FineWeb doc is ~613 bytes ≈ 175 tokens → most batches pad to ~200
        instead of the max 2048+, cutting attention compute by ~10×.

    Adaptive OOM:
        On CUDA OOM the effective batch size is permanently halved and extraction
        resumes from the exact position that failed.  It never stops.

    VRAM hygiene:
        empty_cache + gc after every flush to keep VRAM stable and avoid
        fragmentation accumulating over thousands of batches.
    """
    if preloaded_models:
        model, tokenizer = preloaded_models[0]
    else:
        model, tokenizer = load_teacher(profile=detect_gpu_profile())

    batch_results  = []
    _FLUSH_EVERY   = 50
    _part_counter  = [0]
    _eff_batch     = [batch_size]   # mutable: permanently reduced on OOM
    output_path    = Path(output_path)

    valid_indices = [i for i in range(start_idx, len(corpus_texts))
                     if corpus_texts[i].strip()]

    # ── Bucket sort: group docs by estimated token length to minimise padding ──
    # Qwen2.5 BPE: ~3.5 bytes/token for English.  +1 to avoid zero for empty.
    def _est_tok(idx):
        return min(len(corpus_texts[idx].encode('utf-8')) // 3 + 1, seq_len)

    sorted_indices = sorted(valid_indices, key=_est_tok)
    n_valid        = len(sorted_indices)

    pbar = tqdm(total=n_valid, desc="Extracting logprobs", unit="doc")

    i = 0  # pointer into sorted_indices — only advances on success
    while i < n_valid:
        n              = _eff_batch[0]
        batch_indices  = sorted_indices[i:i + n]
        batch_texts    = [corpus_texts[j] for j in batch_indices]

        # Cap seq_len to the actual max in this bucket + 128 token margin.
        # Avoids allocating a 4096-length KV cache for a batch of 150-token docs.
        bucket_max_tok = max(_est_tok(j) for j in batch_indices)
        bucket_seq_len = min(seq_len, bucket_max_tok + 128)

        try:
            batch_out = extract_logprobs_and_hidden_batch(
                model, tokenizer, batch_texts,
                top_k=TOP_K_LOGPROBS,
                seq_len=bucket_seq_len,
                logit_chunk_size=logit_chunk_size,
            )

            for j, result in zip(batch_indices, batch_out):
                _h = result.get('teacher_hidden')
                if _h is not None:
                    _hb = np.array(_h, dtype=np.float16).tobytes()
                else:
                    _hb = b''
                batch_results.append({
                    'doc_id':         j,
                    'text_bytes':     result['text_bytes'],
                    'positions':      json.dumps(result['positions']),
                    'n_tokens':       result['n_tokens'],
                    'teacher_hidden': _hb,
                    'teacher_dim':    result['teacher_dim'],
                })
                del _h
            del batch_out

            i += len(batch_indices)
            pbar.update(len(batch_indices))

            if len(batch_results) >= _FLUSH_EVERY:
                part_path = (output_path.parent
                             / f"{output_path.stem}_part{_part_counter[0]:04d}.parquet")
                _part_counter[0] += 1
                _write_parquet_part(batch_results, part_path)
                batch_results = []
                gc.collect()
                torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            new_n = max(1, _eff_batch[0] * 3 // 4)
            pbar.write(f"[OOM] batch {_eff_batch[0]} → {new_n} (permanent, 75%). "
                       f"Resuming from doc {i}/{n_valid}")
            _eff_batch[0] = new_n
            # i is NOT advanced — retries the same position with a smaller batch

        except Exception:
            import traceback
            pbar.write(f"[ERROR] skipping batch at {i}")
            traceback.print_exc()
            torch.cuda.empty_cache()
            i += len(batch_indices)
            pbar.update(len(batch_indices))

    pbar.close()

    if batch_results:
        part_path = (output_path.parent
                     / f"{output_path.stem}_part{_part_counter[0]:04d}.parquet")
        _write_parquet_part(batch_results, part_path)

    _merge_parquet_parts(output_path)


def _run_extraction_multi_gpu(corpus_texts, output_path, start_idx, n_gpus,
                              preloaded_models=None, batch_size: int = BATCH_SIZE_TEACHER,
                              seq_len: int = MAX_EXTRACTION_TOKENS,
                              logit_chunk_size: int = 256):
    """Multi-GPU extraction with VRAM-calibrated batches and work-stealing.

    Architecture:
    - VRAM calibration: profile each GPU at startup → exact batch size for 13GB target
    - Work-stealing queue: both GPUs pull from shared queue (load-balanced)
    - Bucket seq_len: cap KV length to actual max in batch + margin (less wasted compute)
    - OOM → permanent batch halving (no recursive split)
    - Async parquet writes: GPU never waits for I/O
    - VRAM monitoring: report fragmentation every N batches
    """
    output_path = Path(output_path)
    lock = threading.Lock()  # only for part counter (microseconds)

    if preloaded_models is not None:
        models = preloaded_models
        print(f"[Extraction] Reusing {len(models)} pre-loaded models")
    else:
        print("[Extraction] Loading models sequentially...")
        profile = detect_gpu_profile()
        models = []
        for gpu_id in range(n_gpus):
            m, tok = load_teacher(device=f"cuda:{gpu_id}", profile=profile)
            torch.cuda.empty_cache()
            models.append((m, tok))
        print("[Extraction] All models loaded.")

    # --- VRAM Calibration: profile each GPU to find optimal batch size ---
    calibrated_bs = {}
    for gpu_id in range(len(models)):
        model, tokenizer = models[gpu_id]
        cal_profile = detect_gpu_profile()
        cal_profile['logit_chunk_size'] = logit_chunk_size
        cal_profile['max_tokens'] = seq_len
        cal_bs = _calibrate_batch_size(model, tokenizer, cal_profile)
        calibrated_bs[gpu_id] = cal_bs

    # Work-stealing queue: all valid doc indices, bucket-sorted by length
    valid_indices = [i for i in range(start_idx, len(corpus_texts))
                     if corpus_texts[i].strip()]

    def _est_tok(idx):
        return min(len(corpus_texts[idx].encode('utf-8')) // 3 + 1, seq_len)

    sorted_indices = sorted(valid_indices, key=_est_tok)
    _work_q = Queue()
    for idx in sorted_indices:
        _work_q.put(idx)
    total_docs = len(sorted_indices)
    print(f"[Extraction] {total_docs} docs → work-stealing queue, {n_gpus} GPUs")
    print(f"[Extraction] Calibrated batch sizes: "
          f"{', '.join(f'GPU{g}={calibrated_bs[g]}' for g in range(len(models)))}")

    _part_counter = [0]
    _async_write_threads: list[threading.Thread] = []  # track for cleanup

    def worker(gpu_id, model, tokenizer):
        bs = calibrated_bs[gpu_id]
        batch_results = []
        processed = 0
        _batch_count = 0

        while True:
            # Peek first doc to estimate length for dynamic batch sizing
            try:
                first_idx = _work_q.get_nowait()
            except Empty:
                break
            est_len = _est_tok(first_idx)
            bucket_seq_len = min(seq_len, est_len + 128)

            # Dynamic batch size: shorter docs → bigger batch
            # Attention VRAM is O(S²), so shorter seqs allow much bigger batches.
            # Docs are sorted by length, so consecutive docs are similar size.
            if bucket_seq_len < seq_len * 0.9:
                vram_scale = (bucket_seq_len / seq_len) ** 1.5
                dynamic_bs = min(int(bs / max(vram_scale, 0.1)), bs * 5)
            else:
                dynamic_bs = bs

            batch_indices = [first_idx]
            for _ in range(dynamic_bs - 1):
                try:
                    batch_indices.append(_work_q.get_nowait())
                except Empty:
                    break

            batch_texts = [corpus_texts[i] for i in batch_indices]

            # Recompute actual bucket_seq_len with full batch
            bucket_max_tok = max(_est_tok(j) for j in batch_indices)
            bucket_seq_len = min(seq_len, bucket_max_tok + 128)

            try:
                batch_out = extract_logprobs_and_hidden_batch(
                    model, tokenizer, batch_texts,
                    top_k=TOP_K_LOGPROBS, seq_len=bucket_seq_len,
                    logit_chunk_size=logit_chunk_size,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                new_bs = max(1, bs * 3 // 4)
                print(f"\n  [GPU {gpu_id}] OOM at batch={bs}, reducing to {new_bs} (permanent, 75%)")
                calibrated_bs[gpu_id] = new_bs
                bs = new_bs
                # Put docs back for retry with smaller batch
                for idx in reversed(batch_indices):
                    _work_q.put(idx)
                continue
            except Exception:
                print(f"\n  [GPU {gpu_id}] ERROR, skipping batch")
                traceback.print_exc()
                torch.cuda.empty_cache()
                processed += len(batch_indices)
                continue

            for i, result in zip(batch_indices, batch_out):
                # Store hidden states as compact float16 bytes instead of Python list.
                # Python list of 768*2048 floats = ~12MB per doc (float64 objects + overhead).
                # numpy float16 bytes = 768*2048*2 = 3.1MB per doc. ~4x smaller, no overhead.
                _h = result['teacher_hidden']
                if _h is not None:
                    import numpy as _np
                    _hb = _np.array(_h, dtype=_np.float16).tobytes()
                else:
                    _hb = b''
                batch_results.append({
                    'doc_id':         i,
                    'text_bytes':     result['text_bytes'],
                    'positions':      json.dumps(result['positions']),
                    'n_tokens':       result['n_tokens'],
                    'teacher_hidden': _hb,
                    'teacher_dim':    result['teacher_dim'],
                })
                del _h, _hb
            del batch_out
            processed += len(batch_indices)
            _batch_count += 1

            # Flush after every batch — async write so GPU never waits for I/O
            if batch_results:
                with lock:
                    pidx = _part_counter[0]; _part_counter[0] += 1
                part_path = output_path.parent / f"{output_path.stem}_part{pidx:04d}.parquet"

                _to_write = batch_results
                wt = threading.Thread(
                    target=_write_parquet_part,
                    args=(_to_write, part_path),
                    daemon=True
                )
                wt.start()
                _async_write_threads.append(wt)

                # Cap concurrent write threads at 2: join oldest to prevent
                # unbounded RAM accumulation (each holds ~100MB of batch data)
                if len(_async_write_threads) > 2:
                    _async_write_threads[0].join()
                    _async_write_threads.pop(0)

                batch_results = []
                pct = min(100, int((total_docs - _work_q.qsize()) / total_docs * 100))

                # VRAM monitoring every 10 batches
                vram_msg = ""
                if _batch_count % 10 == 0:
                    dev_idx = gpu_id
                    frag = _check_vram_fragmentation(dev_idx)
                    free_gb, total_gb = torch.cuda.mem_get_info(dev_idx)
                    used_gb = (total_gb - free_gb) / 1024**3
                    vram_msg = f"  VRAM={used_gb:.1f}GB frag={frag:.0%}"

                print(f"  [GPU {gpu_id}] {processed} docs ({pct}%) "
                      f"bs={bs}{vram_msg}", flush=True)

        # Flush remaining
        if batch_results:
            with lock:
                pidx = _part_counter[0]; _part_counter[0] += 1
            part_path = output_path.parent / f"{output_path.stem}_part{pidx:04d}.parquet"
            _write_parquet_part(batch_results, part_path)
            del batch_results
        torch.cuda.empty_cache()
        print(f"  [GPU {gpu_id}] Done — {processed} docs total", flush=True)

    threads = []
    for gpu_id in range(n_gpus):
        model, tokenizer = models[gpu_id]
        t = threading.Thread(target=worker, args=(gpu_id, model, tokenizer))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Wait for all async parquet writes to complete
    for wt in _async_write_threads:
        wt.join(timeout=30)

    # NO merge — caller reads part files directly via output_path.parent glob
    # This eliminates the GPU-blocking read-concat-write of ~1GB of data.
    print(f"  [Extraction] Complete: {output_path} ({_part_counter[0]} parts)", flush=True)


# ============================================================
# STEP 2a: Marginalize BPE→Byte
# ============================================================

def _marginalize_one_row(row: dict) -> dict:
    """Marginalizar un doc BPE→byte. Diseñado para multiprocessing (sin estado compartido).

    RAM-optimized: teacher_hidden decoded from float16 bytes, teacher_probs
    returned as numpy arrays instead of Python lists (~7x less RAM per patch).
    """
    text_bytes = bytes(row['text_bytes'])
    positions = json.loads(row['positions'])

    # Teacher hidden alignment (for EmbeddingAligner, runs 13/15)
    has_hidden = 'teacher_hidden' in row and row['teacher_hidden'] is not None
    teacher_dim = row.get('teacher_dim', 0) if has_hidden else 0
    if has_hidden:
        _raw = row['teacher_hidden']
        if isinstance(_raw, (bytes, bytearray)):
            hidden_flat = np.frombuffer(_raw, dtype=np.float16).astype(np.float32)
        elif isinstance(_raw, np.ndarray):
            hidden_flat = _raw.astype(np.float32)
        else:
            hidden_flat = np.array(_raw, dtype=np.float32)
        del _raw

    # Autoregressive: pos[i] predicts what comes at positions[i+1]['byte_offset'].
    byte_to_teacher = {}
    byte_offset_to_pos = {}
    for idx, pos in enumerate(positions):
        if idx + 1 >= len(positions):
            break
        top_toks = pos['top_tokens']
        byte_probs = {}
        for item in top_toks:
            if not item['bytes']:
                continue
            first_byte = item['bytes'][0]
            prob = math.exp(item['logprob'])
            byte_probs[first_byte] = byte_probs.get(first_byte, 0.0) + prob
        if not byte_probs:
            continue
        total = sum(byte_probs.values())
        byte_probs = {k: v / total for k, v in byte_probs.items()}
        next_byte_off = positions[idx + 1]['byte_offset']
        byte_to_teacher[next_byte_off] = byte_probs
        if has_hidden:
            byte_offset_to_pos[next_byte_off] = idx
    del positions  # free JSON memory early

    n_patches = len(text_bytes) // PATCH_SIZE
    n_out = max(0, n_patches - 1)

    # Pre-allocate numpy arrays instead of Python lists for large fields
    out_patches = []
    out_targets = []
    out_teacher_probs = np.zeros((n_out, 256), dtype=np.float32)
    out_teacher_mask = np.zeros(n_out, dtype=bool)
    out_teacher_hidden = np.zeros((n_out, teacher_dim), dtype=np.float32) if has_hidden else None
    n_with_teacher = 0

    for p_idx in range(n_out):
        patch_start = p_idx * PATCH_SIZE
        target_start = (p_idx + 1) * PATCH_SIZE
        out_patches.append(list(text_bytes[patch_start:patch_start + PATCH_SIZE]))
        out_targets.append(list(text_bytes[target_start:target_start + PATCH_SIZE]))
        if target_start in byte_to_teacher:
            bp = byte_to_teacher[target_start]
            for b_val, prob in bp.items():
                out_teacher_probs[p_idx, b_val] = prob
            out_teacher_mask[p_idx] = True
            n_with_teacher += 1

            if has_hidden and target_start in byte_offset_to_pos:
                pos_idx = byte_offset_to_pos[target_start]
                start = pos_idx * teacher_dim
                end = start + teacher_dim
                if end <= len(hidden_flat):
                    out_teacher_hidden[p_idx] = hidden_flat[start:end]

    result = {
        'patches': out_patches, 'targets': out_targets,
        'teacher_probs': out_teacher_probs,   # numpy [n_out, 256] float32
        'teacher_mask': out_teacher_mask,       # numpy [n_out] bool
        'n_with_teacher': n_with_teacher,
    }
    if has_hidden:
        result['teacher_hidden'] = out_teacher_hidden  # numpy [n_out, teacher_dim]
        result['teacher_dim'] = teacher_dim
    return result


def build_byte_level_dataset(
    logprobs_parquet: str,
    output_dir: str,
    seq_len: int = SEQ_LEN_PATCHES,
    n_workers: int = 0,
    seq_offset: int = 0,
    carry_state: dict = None,
    shard_offset: int = 0,
    executor=None,              # optional persistent ProcessPoolExecutor
) -> tuple:
    """
    Convert BPE logprobs to byte-level distributions.
    Parallelizado con ProcessPoolExecutor para saturar CPU multi-core.

    RAM-optimized: numpy arrays for teacher_probs/mask/hidden,
    persistent ProcessPoolExecutor (no per-batch fork overhead),
    shard writes without read-concat-write, RAM pressure monitoring.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Support single file or glob of part files
    _p = Path(logprobs_parquet)
    if _p.exists() and _p.is_file():
        part_files = [_p]
    else:
        stem = _p.stem
        part_files = sorted(_p.parent.glob(f"{stem}_part*.parquet"))
        if not part_files:
            print(f"[Byte dataset] No part files found for {logprobs_parquet}")
            empty_carry = {'patches': [], 'targets': [],
                           'teacher_probs_np': np.zeros((0, 256), dtype=np.float32),
                           'teacher_mask_np': np.zeros(0, dtype=bool),
                           'teacher_hidden_np': np.zeros((0, 0), dtype=np.float32),
                           'has_hidden': False, 'teacher_dim': 0}
            return {'n_sequences': 0, 'n_patches_total': 0, 'n_with_teacher': 0, 'coverage_pct': 0.0}, empty_carry

    n_total_rows = sum(pq.ParquetFile(str(f)).metadata.num_rows for f in part_files)

    if n_workers <= 0:
        n_workers = max(1, (os.cpu_count() or 4) - 1)

    # --- Restore carry state ---
    if carry_state is not None:
        all_patches = carry_state.get('patches', [])
        all_targets = carry_state.get('targets', [])
        # Support both old (list) and new (numpy) carry formats
        _tp = carry_state.get('teacher_probs_np', carry_state.get('teacher_probs'))
        if isinstance(_tp, np.ndarray):
            carry_tp = _tp
        elif _tp:
            carry_tp = np.array(_tp, dtype=np.float32)
        else:
            carry_tp = np.zeros((0, 256), dtype=np.float32)
        _tm = carry_state.get('teacher_mask_np', carry_state.get('teacher_mask'))
        if isinstance(_tm, np.ndarray):
            carry_tm = _tm
        elif _tm:
            carry_tm = np.array(_tm, dtype=bool)
        else:
            carry_tm = np.zeros(0, dtype=bool)
        _th = carry_state.get('teacher_hidden_np', carry_state.get('teacher_hidden'))
        has_hidden = carry_state.get('has_hidden', False)
        teacher_dim = carry_state.get('teacher_dim', 0)
        if isinstance(_th, np.ndarray) and _th.size > 0:
            carry_th = _th
        elif _th and has_hidden:
            carry_th = np.array(_th, dtype=np.float32).reshape(-1, max(teacher_dim, 1))
        else:
            carry_th = np.zeros((0, max(teacher_dim, 1)), dtype=np.float32)
    else:
        all_patches = []
        all_targets = []
        carry_tp = np.zeros((0, 256), dtype=np.float32)
        carry_tm = np.zeros(0, dtype=bool)
        carry_th = np.zeros((0, 1), dtype=np.float32)
        has_hidden = False
        teacher_dim = 0

    stats = {'n_sequences': 0, 'n_patches_total': 0, 'n_with_teacher': 0}

    _MARGIN_BATCH = 500  # Read batch from parquet (larger = fewer I/O calls)
    print(f"[Byte dataset] Marginalizing {n_total_rows} docs from {len(part_files)} part file(s)...")

    # Shard part-file tracking: write incremental parts to avoid read-concat-write
    _shard_part_counters: dict[int, int] = {}  # shard_idx -> part counter

    def _flush_records_to_disk(records: list[dict], shard_idx: int):
        """Write records as a NEW part file — no read-merge. O(write_size) only."""
        if not records:
            return
        cnt = _shard_part_counters.get(shard_idx, 0)
        part_path = output_dir / f"shard_{shard_idx:04d}_part{cnt:02d}.parquet"
        _shard_part_counters[shard_idx] = cnt + 1
        _write_byte_shard(records, part_path)

    shard_records: dict[int, list[dict]] = {}

    # ---- S1: Flatten all parquet rows upfront, single map() call ----
    # This eliminates idle gaps between batches. Workers are ALWAYS busy.
    # RAM cost: ~1500 docs × 400KB ≈ 600MB (safe under 25GB cap)
    _read_t0 = time.time()
    all_rows = []
    for part_file in part_files:
        for batch in pq.ParquetFile(str(part_file)).iter_batches(batch_size=_MARGIN_BATCH):
            all_rows.extend(batch.to_pylist())
    print(f"  [Byte] Read {len(all_rows)} rows in {time.time()-_read_t0:.2f}s")

    # RAM guard: check before processing
    _ram_used, _ram_total, _ram_frac = _get_ram_usage()
    if _ram_used > RAM_HARD_LIMIT_GB:
        print(f"  [Byte] WARNING: RAM {_ram_used:.1f}GB > {RAM_HARD_LIMIT_GB}GB limit, forcing GC")
        gc.collect()

    # Single map() — workers pull items continuously with chunksize=1 (no straggler effect)
    if n_workers > 1 and len(all_rows) > 4:
        _exec = executor if executor is not None else ProcessPoolExecutor(max_workers=n_workers)
        try:
            _map_iter = _exec.map(
                _marginalize_one_row, all_rows,
                chunksize=1,  # S2: workers get items immediately, zero idle
            )
            if executor is None:
                all_results = list(_map_iter)
            else:
                all_results = _map_iter  # lazy — stream as workers finish
        finally:
            if executor is None:
                _exec.shutdown(wait=False)
    else:
        all_results = [_marginalize_one_row(r) for r in all_rows]
    del all_rows  # free input memory

    # ---- Stream results into accumulators ----
    _tp_parts = [carry_tp] if carry_tp.shape[0] > 0 else []
    _tm_parts = [carry_tm] if carry_tm.shape[0] > 0 else []
    _th_parts = [carry_th] if carry_th.shape[0] > 0 else []
    _result_count = 0
    for result in all_results:
        all_patches.extend(result['patches'])
        all_targets.extend(result['targets'])
        n_res = len(result['patches'])
        _tp_parts.append(result['teacher_probs'])
        _tm_parts.append(result['teacher_mask'])
        stats['n_with_teacher'] += result['n_with_teacher']
        stats['n_patches_total'] += n_res
        if 'teacher_hidden' in result:
            has_hidden = True
            teacher_dim = result['teacher_dim']
            _th_parts.append(result['teacher_hidden'])
        del result
        _result_count += 1

        # Emit complete sequences inline (streaming)
        while len(all_patches) >= seq_len:
            hidden_slice = carry_th[:seq_len] if has_hidden and carry_th.shape[0] >= seq_len else None
            # Need to concat accumulated parts to check carry sizes
            if _tp_parts and (len(_tp_parts) > 50 or len(all_patches) >= seq_len):
                carry_tp = np.concatenate(_tp_parts, axis=0) if _tp_parts else carry_tp
                carry_tm = np.concatenate(_tm_parts) if _tm_parts else carry_tm
                if has_hidden:
                    valid_th = [p for p in _th_parts if p.shape[1] == teacher_dim] if _th_parts else []
                    carry_th = np.concatenate(valid_th, axis=0) if valid_th else np.zeros((0, teacher_dim), dtype=np.float32)
                _tp_parts = []
                _tm_parts = []
                _th_parts = []
                hidden_slice = carry_th[:seq_len] if has_hidden and carry_th.shape[0] >= seq_len else None

            if carry_tp.shape[0] < seq_len:
                break  # not enough teacher data yet

            record = _make_byte_record(
                all_patches[:seq_len], all_targets[:seq_len],
                carry_tp[:seq_len], carry_tm[:seq_len],
                teacher_hidden=hidden_slice, teacher_dim=teacher_dim,
            )
            shard = shard_offset + stats['n_sequences'] // SEQS_PER_SHARD
            shard_records.setdefault(shard, []).append(record)
            all_patches = all_patches[seq_len:]
            all_targets = all_targets[seq_len:]
            carry_tp = carry_tp[seq_len:]
            carry_tm = carry_tm[seq_len:]
            if has_hidden:
                carry_th = carry_th[seq_len:]
            stats['n_sequences'] += 1

        # Periodic flush to disk
        for shard_idx in list(shard_records.keys()):
            if len(shard_records[shard_idx]) >= 10:
                _flush_records_to_disk(shard_records[shard_idx], shard_idx)
                shard_records[shard_idx] = []

        # RAM pressure check every 200 rows
        if _result_count % 200 == 0:
            _ram_used, _, _ = _get_ram_usage()
            if _ram_used > RAM_HARD_LIMIT_GB:
                print(f"  [Byte] RAM {_ram_used:.1f}GB > limit, forcing flush+GC")
                for _si in list(shard_records.keys()):
                    if shard_records[_si]:
                        _flush_records_to_disk(shard_records[_si], _si)
                        shard_records[_si] = []
                gc.collect()
            if len(all_patches) > MAX_ACCUMULATED_PATCHES:
                print(f"  [Byte] WARNING: {len(all_patches)} patches accumulated "
                      f"(limit {MAX_ACCUMULATED_PATCHES}), forcing GC")
                gc.collect()
            _check_ram_pressure("byte-margin")

    # Final concat of any remaining parts
    if _tp_parts:
        carry_tp = np.concatenate(_tp_parts, axis=0)
        carry_tm = np.concatenate(_tm_parts)
        if has_hidden:
            valid_th = [p for p in _th_parts if p.shape[1] == teacher_dim] if _th_parts else []
            carry_th = np.concatenate(valid_th, axis=0) if valid_th else np.zeros((0, teacher_dim), dtype=np.float32)
        del _tp_parts, _tm_parts, _th_parts

    # Emit any remaining complete sequences
    while len(all_patches) >= seq_len:
        hidden_slice = carry_th[:seq_len] if has_hidden else None
        record = _make_byte_record(
            all_patches[:seq_len], all_targets[:seq_len],
            carry_tp[:seq_len], carry_tm[:seq_len],
            teacher_hidden=hidden_slice, teacher_dim=teacher_dim,
        )
        shard = shard_offset + stats['n_sequences'] // SEQS_PER_SHARD
        shard_records.setdefault(shard, []).append(record)
        all_patches = all_patches[seq_len:]
        all_targets = all_targets[seq_len:]
        carry_tp = carry_tp[seq_len:]
        carry_tm = carry_tm[seq_len:]
        if has_hidden:
            carry_th = carry_th[seq_len:]
        stats['n_sequences'] += 1

    for shard_idx in list(shard_records.keys()):
        if len(shard_records[shard_idx]) >= 10:
            _flush_records_to_disk(shard_records[shard_idx], shard_idx)
            shard_records[shard_idx] = []

    del all_results

    # Final flush
    for shard_idx, records in shard_records.items():
        if records:
            _flush_records_to_disk(records, shard_idx)
    shard_records.clear()

    # Merge shard part files into final shard files
    for shard_idx in _shard_part_counters:
        shard_path = output_dir / f"shard_{shard_idx:04d}.parquet"
        parts = sorted(output_dir.glob(f"shard_{shard_idx:04d}_part*.parquet"))
        if parts:
            tables = [pq.read_table(str(p)) for p in parts]
            combined = pa.concat_tables(tables)
            pq.write_table(combined, str(shard_path))
            for p in parts:
                p.unlink()
            del tables, combined
    _shard_part_counters.clear()
    gc.collect()

    coverage = stats['n_with_teacher'] / max(stats['n_patches_total'], 1) * 100
    stats['coverage_pct'] = coverage

    print(f"[Byte dataset] {stats['n_sequences']} sequences (chunk), total offset {seq_offset + stats['n_sequences']}")
    print(f"[Byte dataset] Teacher coverage: {coverage:.1f}%")

    next_carry = {
        'patches': all_patches, 'targets': all_targets,
        'teacher_probs_np': carry_tp, 'teacher_mask_np': carry_tm,
        'teacher_hidden_np': carry_th,
        'has_hidden': has_hidden, 'teacher_dim': teacher_dim,
    }
    return stats, next_carry


def _make_byte_record(patches, targets, teacher_probs, teacher_mask,
                      teacher_hidden=None, teacher_dim: int = 0) -> dict:
    """Build a flat record dict for one byte-level sequence.

    Accepts numpy arrays or Python lists for teacher_probs/mask/hidden.
    Flattens using numpy for efficiency.
    """
    record = {
        'patches':       [b for p in patches for b in p],
        'targets':       [b for t in targets for b in t],
    }
    # teacher_probs: numpy [seq_len, 256] or list of [256] lists
    if isinstance(teacher_probs, np.ndarray):
        record['teacher_probs'] = teacher_probs.ravel().tolist()
    else:
        record['teacher_probs'] = [v for row in teacher_probs for v in row]
    # teacher_mask: numpy [seq_len] bool or list of bools
    if isinstance(teacher_mask, np.ndarray):
        record['teacher_mask'] = teacher_mask.tolist()
    else:
        record['teacher_mask'] = list(teacher_mask)
    if teacher_hidden is not None:
        if isinstance(teacher_hidden, np.ndarray):
            record['teacher_hidden'] = teacher_hidden.ravel().tolist()
        else:
            record['teacher_hidden'] = [v for row in teacher_hidden for v in row]
        record['teacher_dim'] = teacher_dim
    return record


def _write_byte_shard(records: list[dict], shard_path: Path) -> None:
    """Write byte-level records to a shard parquet. No read-concat-write."""
    if not records:
        return
    columns = {
        'patches':       pa.array([r['patches'] for r in records], type=pa.list_(pa.uint8())),
        'targets':       pa.array([r['targets'] for r in records], type=pa.list_(pa.uint8())),
        'teacher_probs': pa.array([r['teacher_probs'] for r in records], type=pa.list_(pa.float32())),
        'teacher_mask':  pa.array([r['teacher_mask'] for r in records], type=pa.list_(pa.bool_())),
    }
    if 'teacher_hidden' in records[0]:
        columns['teacher_hidden'] = pa.array(
            [r['teacher_hidden'] for r in records], type=pa.list_(pa.float32()),
        )
        columns['teacher_dim'] = pa.array(
            [r['teacher_dim'] for r in records], type=pa.int32(),
        )
    table = pa.table(columns)
    # Write directly — no read-concat-write. Caller uses part files for incremental writes.
    pq.write_table(table, str(shard_path))


# ============================================================
# STEP 2b: Build BPE dataset
# ============================================================

def _tokenize_bpe_row(args: tuple) -> dict:
    """Procesar un documento BPE -> tokens + teacher signals.
    Diseñado para ProcessPoolExecutor (sin estado compartido).

    Args:
        args: (row_dict, tokenizer_name, seq_len, top_k)
    Returns:
        dict con ids, probs, top_ids, mask
    """
    row, tokenizer_name, seq_len_max, top_k = args

    # Lazy import para workers (cada worker importa una vez)
    # Use local cache path if available to avoid network timeouts
    from transformers import AutoTokenizer
    if not hasattr(_tokenize_bpe_row, '_tokenizer'):
        local_cache = Path("/tmp/_bpe_tokenizer_cache")
        if local_cache.exists():
            _tokenize_bpe_row._tokenizer = AutoTokenizer.from_pretrained(str(local_cache))
        else:
            _tokenize_bpe_row._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer = _tokenize_bpe_row._tokenizer

    text_bytes = bytes(row['text_bytes'])
    positions = json.loads(row['positions'])
    text = text_bytes.decode('utf-8', errors='replace')
    enc = tokenizer(text, truncation=True, max_length=seq_len_max * 2)
    ids = enc['input_ids']

    result_ids = []
    result_probs = []
    result_top_ids = []
    result_mask = []

    for pos_idx, (token_id, pos_data) in enumerate(
        zip(ids, positions[:len(ids)])
    ):
        top_toks = pos_data.get('top_tokens', [])
        probs = [0.0] * top_k
        top_ids = [0] * top_k
        has_teacher = len(top_toks) > 0

        if has_teacher:
            for k_idx, item in enumerate(top_toks[:top_k]):
                probs[k_idx] = math.exp(item['logprob'])
                top_ids[k_idx] = item.get('token_id', 0)

        result_ids.append(token_id)
        result_probs.append(probs)
        result_top_ids.append(top_ids)
        result_mask.append(has_teacher)

    return {
        'ids': result_ids, 'probs': result_probs,
        'top_ids': result_top_ids, 'mask': result_mask,
    }


def build_bpe_dataset(
    logprobs_parquet: str,
    output_dir: str,
    seq_len: int = SEQ_LEN_TOKENS,
    n_workers: int = 0,
    seq_offset: int = 0,
    carry_state: dict = None,
    executor=None,             # optional persistent ProcessPoolExecutor
    shard_offset: int = 0,
) -> tuple:
    """
    Build BPE dataset from the same teacher logprobs.
    Produces dataset for runs 01-07 (BPE).
    Paralelizado con ProcessPoolExecutor: cada worker tokeniza un doc independiente.

    Teacher logprobs are kept in BPE space (no marginalization).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Support single file or glob of part files
    _p = Path(logprobs_parquet)
    if _p.exists() and _p.is_file():
        part_files = [_p]
    else:
        stem = _p.stem
        part_files = sorted(_p.parent.glob(f"{stem}_part*.parquet"))
        if not part_files:
            print(f"[BPE dataset] No part files found for {logprobs_parquet}")
            empty_carry = {'ids': [], 'probs': [], 'top_ids': [], 'mask': []}
            return {'n_sequences': 0, 'n_tokens_total': 0}, empty_carry

    n_total_rows = sum(pq.ParquetFile(str(f)).metadata.num_rows for f in part_files)

    stats = {'n_sequences': 0, 'n_tokens_total': 0}
    if carry_state is not None:
        buffer_ids = carry_state['ids']
        buffer_probs = carry_state['probs']
        buffer_top_ids = carry_state['top_ids']
        buffer_mask = carry_state['mask']
    else:
        buffer_ids = []
        buffer_probs = []
        buffer_top_ids = []
        buffer_mask = []

    global _BPE_TOKENIZER_CACHE
    if '_BPE_TOKENIZER_CACHE' not in globals() or _BPE_TOKENIZER_CACHE is None:
        from transformers import AutoTokenizer
        _BPE_TOKENIZER_CACHE = AutoTokenizer.from_pretrained(TEACHER_MODEL)

    _BPE_BATCH = 500  # Read batch from parquet (larger = fewer I/O calls)
    print(f"[BPE dataset] Tokenizing {n_total_rows} docs from {len(part_files)} part file(s)...")

    shard_records: dict[int, list[dict]] = {}

    if n_workers <= 0:
        n_workers = max(1, (os.cpu_count() or 4) - 1)

    # ---- S1: Flatten all parquet rows upfront, single map() call ----
    # Eliminates idle gaps between batches. Workers are ALWAYS busy.
    _read_t0 = time.time()
    all_args = []
    for part_file in part_files:
        for batch in pq.ParquetFile(str(part_file)).iter_batches(batch_size=_BPE_BATCH):
            for r in batch.to_pylist():
                all_args.append((r, TEACHER_MODEL, seq_len, TOP_K_LOGPROBS))
    print(f"  [BPE] Read {len(all_args)} rows in {time.time()-_read_t0:.2f}s")

    # RAM guard: check before processing
    _ram_used, _, _ = _get_ram_usage()
    if _ram_used > RAM_HARD_LIMIT_GB:
        print(f"  [BPE] WARNING: RAM {_ram_used:.1f}GB > {RAM_HARD_LIMIT_GB}GB limit, forcing GC")
        gc.collect()

    # Single map() — workers pull items continuously with chunksize=1 (no straggler effect)
    if n_workers > 1 and len(all_args) > 4:
        _exec = executor if executor is not None else ProcessPoolExecutor(max_workers=n_workers)
        try:
            _map_iter = _exec.map(
                _tokenize_bpe_row, all_args,
                chunksize=1,  # S2: workers get items immediately, zero idle
            )
            if executor is None:
                all_results = list(_map_iter)
            else:
                all_results = _map_iter  # lazy — stream as workers finish
        finally:
            if executor is None:
                _exec.shutdown(wait=False)
    else:
        all_results = [_tokenize_bpe_row(a) for a in all_args]
    del all_args  # free input memory

    # ---- Stream results into accumulators ----
    _result_count = 0
    for result in all_results:
        buffer_ids.extend(result['ids'])
        buffer_probs.extend(result['probs'])
        buffer_top_ids.extend(result['top_ids'])
        buffer_mask.extend(result['mask'])
        stats['n_tokens_total'] += len(result['ids'])
        del result
        _result_count += 1

        # Emit complete sequences inline (streaming)
        while len(buffer_ids) >= seq_len + 1:
            record = _make_bpe_record(
                buffer_ids[:seq_len],
                buffer_ids[1:seq_len + 1],
                buffer_probs[:seq_len],
                buffer_top_ids[:seq_len],
                buffer_mask[:seq_len],
            )
            shard = shard_offset + stats['n_sequences'] // SEQS_PER_SHARD
            shard_records.setdefault(shard, []).append(record)
            buffer_ids = buffer_ids[seq_len:]
            buffer_probs = buffer_probs[seq_len:]
            buffer_top_ids = buffer_top_ids[seq_len:]
            buffer_mask = buffer_mask[seq_len:]
            stats['n_sequences'] += 1

        # Periodic flush to save RAM
        for shard_idx in list(shard_records.keys()):
            if len(shard_records[shard_idx]) >= 10:
                shard_path = output_dir / f"shard_{shard_idx:04d}.parquet"
                _write_bpe_shard(shard_records[shard_idx], shard_path)
                shard_records[shard_idx] = []

        # RAM pressure check every 200 rows
        if _result_count % 200 == 0:
            _ram_used, _, _ = _get_ram_usage()
            if _ram_used > RAM_HARD_LIMIT_GB:
                print(f"  [BPE] RAM {_ram_used:.1f}GB > limit, forcing flush+GC")
                for _si in list(shard_records.keys()):
                    if shard_records[_si]:
                        shard_path = output_dir / f"shard_{_si:04d}.parquet"
                        _write_bpe_shard(shard_records[_si], shard_path)
                        shard_records[_si] = []
                gc.collect()
            if len(buffer_ids) > MAX_ACCUMULATED_TOKENS:
                print(f"  [BPE] WARNING: {len(buffer_ids)} tokens accumulated "
                      f"(limit {MAX_ACCUMULATED_TOKENS}), forcing GC")
                gc.collect()
            _check_ram_pressure("bpe-tok")

    # Batch write all remaining records per shard
    for shard, records in shard_records.items():
        if records:
            shard_path = output_dir / f"shard_{shard:04d}.parquet"
            _write_bpe_shard(records, shard_path)

    print(f"[BPE dataset] {stats['n_sequences']} sequences (chunk), total offset {seq_offset + stats['n_sequences']}")
    next_carry = {
        'ids': buffer_ids, 'probs': buffer_probs,
        'top_ids': buffer_top_ids, 'mask': buffer_mask,
    }
    return stats, next_carry


def _make_bpe_record(input_ids, labels, teacher_probs, teacher_ids_top,
                     teacher_mask) -> dict:
    """Build a flat record dict for one BPE sequence."""
    return {
        'input_ids':  input_ids,
        'labels':     labels,
        'teacher_probs_bpe': [v for row in teacher_probs for v in row],
        'teacher_ids_bpe':   [v for row in teacher_ids_top for v in row],
        'teacher_mask_bpe':  list(teacher_mask),
    }


def _write_bpe_shard(records: list[dict], shard_path: Path) -> None:
    """Write multiple BPE records to a shard parquet in one shot."""
    if not records:
        return
    table = pa.table({
        'input_ids':         pa.array([r['input_ids'] for r in records], type=pa.list_(pa.int32())),
        'labels':            pa.array([r['labels'] for r in records], type=pa.list_(pa.int32())),
        'teacher_probs_bpe': pa.array([r['teacher_probs_bpe'] for r in records], type=pa.list_(pa.float32())),
        'teacher_ids_bpe':   pa.array([r['teacher_ids_bpe'] for r in records], type=pa.list_(pa.int32())),
        'teacher_mask_bpe':  pa.array([r['teacher_mask_bpe'] for r in records], type=pa.list_(pa.bool_())),
    })
    if shard_path.exists():
        existing = pq.read_table(str(shard_path))
        table = pa.concat_tables([existing, table])
    pq.write_table(table, str(shard_path))


# ============================================================
# STEP 3: Split info & upload
# ============================================================

def _save_stream_checkpoint(api, repo_id: str, chunk_idx: int,
                            byte_seq_offset: int, bpe_seq_offset: int,
                            byte_carry: dict, bpe_carry: dict,
                            shard_offset: int = 0) -> None:
    """Guarda checkpoint en HF Hub para poder retomar si se corta la sesion."""
    import pickle, io
    checkpoint = {
        'last_completed_chunk': chunk_idx,
        'byte_seq_offset': byte_seq_offset,
        'bpe_seq_offset': bpe_seq_offset,
        'shard_offset': shard_offset,  # persisted so resume regenerates correct shard names
    }
    # Metadata JSON
    buf = io.BytesIO(json.dumps(checkpoint).encode())
    _hf_upload_with_retry(api, buf, 'stream_checkpoint.json', repo_id)
    # Carry states (pickle)
    carry_buf = io.BytesIO(pickle.dumps({'byte': byte_carry, 'bpe': bpe_carry}))
    _hf_upload_with_retry(api, carry_buf, 'stream_carry.pkl', repo_id)
    print(f"  [checkpoint] Chunk {chunk_idx} saved to HF Hub")


def _load_stream_checkpoint(api, repo_id: str):
    """Carga checkpoint de HF Hub. Retorna (checkpoint_dict, carry_dict) o (None, None)."""
    import pickle
    from huggingface_hub import hf_hub_download
    try:
        files = list(api.list_repo_files(repo_id, repo_type='dataset'))
        if 'stream_checkpoint.json' not in files:
            return None, None
        ckpt_path = hf_hub_download(repo_id=repo_id, filename='stream_checkpoint.json',
                                    repo_type='dataset')
        with open(ckpt_path) as f:
            checkpoint = json.load(f)
        carry_path = hf_hub_download(repo_id=repo_id, filename='stream_carry.pkl',
                                     repo_type='dataset')
        with open(carry_path, 'rb') as f:
            carry = pickle.load(f)
        print(f"  [checkpoint] Resuming from chunk {checkpoint['last_completed_chunk'] + 1}")
        return checkpoint, carry
    except Exception as e:
        print(f"  [checkpoint] No checkpoint found or error loading: {e}")
        return None, None


def _save_split_info(n_sequences: int, output_dir: Path) -> None:
    """Save train/val/test split with fixed seed."""
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(n_sequences).tolist()

    n_train = int(0.95 * n_sequences)
    n_val = int(0.025 * n_sequences)

    split = {
        'train': indices[:n_train],
        'val': indices[n_train:n_train + n_val],
        'test': indices[n_train + n_val:],
        'seed': SEED,
        'n_sequences': n_sequences,
    }
    with open(output_dir / 'split_info.json', 'w') as f:
        json.dump(split, f)


def _hf_upload_with_retry(api, path_or_fileobj, path_in_repo, repo_id, max_retries=3):
    """Upload to HF Hub with retry on timeout/network errors."""
    for attempt in range(max_retries):
        try:
            api.upload_file(
                path_or_fileobj=path_or_fileobj,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
            )
            return
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"  [hub] Upload failed ({e}), retry in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [hub] Upload failed after {max_retries} attempts: {e}")
                raise


def _upload_sealed_shards(output_dir: Path, api, repo_id: str,
                          repo_subdir: str, total_seqs: int) -> None:
    """Upload and delete shards that are complete (won't receive more sequences).

    A shard N is sealed when total_seqs > (N+1)*SEQS_PER_SHARD.
    """
    max_sealed = total_seqs // SEQS_PER_SHARD  # shards 0..max_sealed-1 are complete
    for shard_idx in range(max_sealed):
        shard_path = output_dir / f"shard_{shard_idx:04d}.parquet"
        if shard_path.exists():
            _hf_upload_with_retry(api, str(shard_path),
                                  f"{repo_subdir}/shard_{shard_idx:04d}.parquet", repo_id)
            shard_path.unlink()
            print(f"  [hub] Uploaded + deleted {repo_subdir}/shard_{shard_idx:04d}.parquet")


def _upload_all_shards(output_dir: Path, api, repo_id: str, repo_subdir: str) -> int:
    """Upload ALL shards (sealed + incomplete). Returns count uploaded.

    Per-shard error isolation: one failed shard does NOT block the rest.
    Failed shards stay on disk and will be retried on the next upload cycle.
    """
    count = 0
    failed = 0
    for shard_path in sorted(output_dir.glob("shard_*.parquet")):
        try:
            _hf_upload_with_retry(api, str(shard_path),
                                  f"{repo_subdir}/{shard_path.name}", repo_id)
            shard_path.unlink()
            count += 1
        except Exception as e:
            failed += 1
            print(f"  [hub] FAILED {repo_subdir}/{shard_path.name}: {e}", flush=True)
    if count:
        print(f"  [hub] Uploaded {count} shards from {repo_subdir}/", flush=True)
    if failed:
        print(f"  [hub] WARNING: {failed} shards FAILED in {repo_subdir}/ "
              f"(will retry next cycle)", flush=True)
    return count


def _clear_hf_dataset(api, repo_id: str) -> None:
    """Delete ALL files from the HF dataset repo for a fresh start.

    Called when --clear-dataset is passed. Removes all shards, checkpoints,
    and metadata so the pipeline starts completely fresh.
    """
    try:
        files = list(api.list_repo_files(repo_id, repo_type='dataset'))
        # Filter out .gitattributes and README
        to_delete = [f for f in files
                     if not f.startswith('.git') and f != 'README.md']
        if not to_delete:
            print(f"  [hub] Repo {repo_id} is already empty.")
            return
        print(f"  [hub] Clearing {len(to_delete)} files from {repo_id}...")
        for f in to_delete:
            try:
                api.delete_file(f, repo_id=repo_id, repo_type='dataset')
            except Exception as e:
                print(f"    [hub] Could not delete {f}: {e}")
        print(f"  [hub] Cleared {len(to_delete)} files from {repo_id}")
    except Exception as e:
        print(f"  [hub] Error clearing repo: {e}")


def upload_dataset_to_hub(local_dir: str = str(OUTPUT_DIR)) -> None:
    """Upload complete dataset to HuggingFace Hub (fallback si no se usa hf-mount)."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=local_dir,
        repo_id=DATASET_REPO,
        repo_type="dataset",
        ignore_patterns=["*.tmp", "__pycache__"],
    )
    print(f"\n[Upload] Dataset available at:")
    print(f"  https://huggingface.co/datasets/{DATASET_REPO}")


# ============================================================
# MAIN: Full pipeline
# ============================================================

def build_full_dataset(corpus_name: str = "HuggingFaceFW/fineweb",
                       n_docs: int = 500_000,
                       skip_extraction: bool = False,
                       use_hf_mount: bool = False,
                       stream_to_hub: bool = False,
                       chunk_size: int = 1000,
                       gpu_profile_override: dict = None,
                       shard_offset: int = 0,
                       **_kwargs) -> None:
    """
    Complete dataset construction pipeline.

    Step 1: Download FineWeb corpus
    Step 2: Extract logprobs + projected hidden states with Qwen-7B
    Step 3a: Marginalize BPE→byte (with teacher_hidden for runs 13, 15)
    Step 3b: Build BPE dataset
    Step 4: Upload (hf-mount: ya esta en el bucket; sin mount: upload via API)

    Only needs to run ONCE. Covers all 15 ablation runs.

    Args:
        use_hf_mount: si True, monta un HF Bucket y escribe directo ahi.
                      Los archivos van a HF storage en tiempo real, sin paso
                      de upload separado. Ahorra disco local en Kaggle.
    """
    global OUTPUT_DIR
    from datasets import load_dataset

    # Global seed for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # chunk_size=500: ~250 docs/GPU, ~63 batches at batch_size=4.
    # RAM for 500-doc logprobs: ~1-2GB. Safe on Kaggle's 13GB.
    chunk_size = min(chunk_size, 2500)
    print(f"  [config] chunk_size = {chunk_size}")

    # Si hf-mount esta habilitado, montar bucket y escribir directo ahi
    mount_path = None
    if use_hf_mount:
        from hf_mount import mount_bucket
        mount_path = mount_bucket(DATASET_BUCKET)
        OUTPUT_DIR = mount_path
        print(f"[hf-mount] Output dir -> {OUTPUT_DIR} (bucket: {DATASET_BUCKET})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logprobs_path = OUTPUT_DIR / "teacher_logprobs" / "logprobs_raw.parquet"

    if not skip_extraction:
        # Step 1: Download corpus
        print(f"\n{'=' * 60}")
        print("STEP 1: Downloading FineWeb corpus")
        print(f"{'=' * 60}")

        # Step 2: Extract logprobs (the slow part)
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        _gpu_profile = gpu_profile_override or detect_gpu_profile()
        print(f"\n{'=' * 60}")
        print("STEP 2: Extracting teacher logprobs (Qwen-7B)")
        print(f"  GPU: {_gpu_profile['name']}  ({_gpu_profile['vram_gb']:.1f} GB)  "
              f"fp8={_gpu_profile['fp8']}  batch={_gpu_profile['batch_size']}  "
              f"seq={_gpu_profile['max_tokens']}")
        if n_gpus >= 2:
            print(f"  {n_gpus} GPUs detectadas → extraccion en paralelo (~3-5 hours)")
        elif _gpu_profile['fp8']:
            print(f"  L4 detected → FP8 W8A8, batch {BATCH_SIZE_TEACHER_L4}, ~2-3 hours")
        else:
            print("  (~6-10 hours on single T4, supports resume)")
        print(f"{'=' * 60}")

        if stream_to_hub:
            # === STREAMING MODE ===
            # 6-thread architecture with bounded queues for backpressure:
            #   Thread 1: corpus_download (fast, with HF_TOKEN auth)
            #   Thread 2: model_loader (NF4 quantization to VRAM)
            #   Thread 3: gpu_producer (extraction loop → extraction_q)
            #   Thread 4: cpu_consumer (byte+BPE processing, reads extraction_q)
            #   Thread 5: upload_worker (reads upload_q, uploads to HF)
            #   Main: orchestrator
            from huggingface_hub import HfApi
            hf_token = os.environ.get('HF_TOKEN')
            print(f"  [hub] HF_TOKEN: {'set (' + str(len(hf_token)) + ' chars)' if hf_token else 'NOT SET / EMPTY'}")
            api = HfApi(token=hf_token)
            try:
                api.create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True)
                print(f"  [hub] Repo ready: {DATASET_REPO}")
            except Exception as e:
                print(f"  [hub] create_repo FAILED: {e}")
                raise

            # Verify upload actually works before spending hours on GPU
            import io as _io
            _test_buf = _io.BytesIO(b"upload_test")
            try:
                api.upload_file(
                    path_or_fileobj=_test_buf,
                    path_in_repo=".upload_test",
                    repo_id=DATASET_REPO,
                    repo_type="dataset",
                )
                api.delete_file(".upload_test", repo_id=DATASET_REPO,
                                repo_type="dataset")
                print(f"  [hub] Upload verification OK (write access confirmed)")
            except Exception as e:
                print(f"\n  [hub] *** UPLOAD VERIFICATION FAILED ***")
                print(f"  [hub] Token can create repo but CANNOT upload files: {e}")
                print(f"  [hub] Fix HF_TOKEN permissions before running pipeline!")
                raise RuntimeError(f"HF upload verification failed: {e}")

            byte_dir = OUTPUT_DIR / "byte_marginalized"
            bpe_dir = OUTPUT_DIR / "bpe_tokenized"
            byte_dir.mkdir(parents=True, exist_ok=True)
            bpe_dir.mkdir(parents=True, exist_ok=True)

            byte_seq_offset = 0
            bpe_seq_offset = 0
            byte_carry = None
            bpe_carry = None
            total_byte_coverage = 0.0
            total_chunks = 0
            resume_from_chunk = 0

            # Resume desde checkpoint si existe
            ckpt, carry = _load_stream_checkpoint(api, DATASET_REPO)
            if ckpt is not None:
                resume_from_chunk = ckpt['last_completed_chunk'] + 1
                byte_seq_offset = ckpt['byte_seq_offset']
                bpe_seq_offset = ckpt['bpe_seq_offset']
                # Restore shard_offset from checkpoint (overrides CLI arg if checkpoint exists)
                if 'shard_offset' in ckpt:
                    shard_offset = ckpt['shard_offset']
                    print(f"  [resume] shard_offset={shard_offset} (from checkpoint)")
                byte_carry = carry['byte']
                bpe_carry = carry['bpe']
                print(f"  [resume] Saltando chunks 0-{resume_from_chunk - 1}, "
                      f"byte_seqs={byte_seq_offset}, bpe_seqs={bpe_seq_offset}")
            else:
                # No checkpoint — scan existing HF Hub shards to reconstruct state.
                # If a previous session wrote shards but ended before checkpoint (chunk 5),
                # we recover seq offsets and shard_offset from the actual data on HF.
                try:
                    from huggingface_hub import hf_hub_download as _hf_dl
                    import pyarrow.parquet as _pq

                    _existing = list(api.list_repo_files(DATASET_REPO, repo_type='dataset'))

                    def _shard_files(prefix):
                        out = []
                        for f in _existing:
                            if f.startswith(prefix + '/shard_') and f.endswith('.parquet'):
                                try:
                                    idx = int(f.split('shard_')[1].split('.')[0])
                                    out.append((idx, f))
                                except ValueError:
                                    pass
                        return sorted(out)

                    def _count_rows_on_hf(shard_list):
                        """Download each shard metadata-only and count rows."""
                        total = 0
                        max_idx = -1
                        for idx, fname in shard_list:
                            try:
                                _p = _hf_dl(repo_id=DATASET_REPO, filename=fname,
                                            repo_type='dataset')
                                _meta = _pq.read_metadata(_p)
                                total += _meta.num_rows
                                max_idx = max(max_idx, idx)
                            except Exception as _re:
                                print(f"  [shard_scan] Could not read {fname}: {_re}")
                        return total, max_idx

                    _byte_shards = _shard_files('byte_marginalized')
                    _bpe_shards  = _shard_files('bpe_tokenized')

                    print(f"  [shard_scan] Found {len(_byte_shards)} byte shards, "
                          f"{len(_bpe_shards)} BPE shards on HF Hub. Counting rows...")

                    _byte_rows, _byte_max_idx = _count_rows_on_hf(_byte_shards)
                    _bpe_rows,  _bpe_max_idx  = _count_rows_on_hf(_bpe_shards)

                    # Next free shard index
                    _safe_offset = max(
                        _byte_max_idx + 1 if _byte_max_idx >= 0 else 0,
                        _bpe_max_idx  + 1 if _bpe_max_idx  >= 0 else 0,
                        shard_offset)

                    # Restore seq offsets from actual shard row counts
                    if _byte_rows > 0:
                        byte_seq_offset = _byte_rows
                    if _bpe_rows > 0:
                        bpe_seq_offset = _bpe_rows

                    # Estimate which chunk to skip based on bpe sequences
                    # BPE seqs contain SEQ_LEN_TOKENS tokens each.
                    # Average FineWeb doc ~200 tokens. total_tokens / tokens_per_doc = docs.
                    if _bpe_rows > 0:
                        _total_tokens = _bpe_rows * SEQ_LEN_TOKENS
                        _avg_tokens_per_doc = 200  # FineWeb English average
                        _est_docs_done = int(_total_tokens / _avg_tokens_per_doc)
                        _est_chunks_done = max(0, _est_docs_done // chunk_size - 1)
                        resume_from_chunk = _est_chunks_done
                        print(f"  [shard_scan] ~{_est_docs_done:,} docs ({_total_tokens:,} tokens) "
                              f"in shards \u2192 skipping ~{_est_chunks_done} chunks")

                    shard_offset = _safe_offset

                    print(f"  [shard_scan] Recovered: byte_seqs={byte_seq_offset}, "
                          f"bpe_seqs={bpe_seq_offset}, shard_offset={shard_offset}, "
                          f"resume_from_chunk≈{resume_from_chunk}")
                    print(f"  [shard_scan] NOTE: carry state unknown (no checkpoint). "
                          f"Sequences will append correctly but boundary may shift slightly.")

                except Exception as _e:
                    print(f"  [shard_scan] Could not scan HF Hub: {_e}. "
                          f"Using defaults (shard_offset={shard_offset})")
            _pipeline_t0 = time.time()

            # Pre-import to avoid threading import lock and lazy module race conditions
            import transformers
            import datasets

            # ================================================================
            # PHASE 0: Fast Cold Start — corpus + models in parallel (<3 min)
            # ================================================================
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            _gpu_profile = gpu_profile_override or detect_gpu_profile()

            _preloaded_models = []
            _corpus_texts = []
            _load_errors = []

            def _load_corpus():
                try:
                    _texts = _fast_download_corpus(
                        corpus_name, n_docs, hf_token=hf_token)
                    _corpus_texts.extend(_texts)
                except Exception as _e:
                    _load_errors.append(("corpus", _e))

            def _load_models():
                try:
                    for _gid in range(max(n_gpus, 1)):
                        _dev = f"cuda:{_gid}" if n_gpus > 0 else "cpu"
                        _m, _tok = load_teacher(device=_dev, profile=_gpu_profile)
                        torch.cuda.empty_cache()
                        _preloaded_models.append((_m, _tok))
                    print(f"  [models] {len(_preloaded_models)} teacher(s) loaded", flush=True)
                except Exception as _e:
                    _load_errors.append(("models", _e))

            print(f"\n[Pipeline] Phase 0: Fast cold start (target <3 min)...")
            _t0_cold = time.time()
            _t_corpus = threading.Thread(target=_load_corpus, name="corpus-loader", daemon=True)
            _t_models = threading.Thread(target=_load_models, name="model-loader", daemon=True)
            _t_corpus.start(); _t_models.start()
            _t_corpus.join();  _t_models.join()

            if _load_errors:
                for _n, _e in _load_errors:
                    print(f"  [ERROR] {_n}: {_e}")
                raise _load_errors[0][1]

            texts = _corpus_texts
            # Sort by length: enables efficient bucket batching (short docs get
            # bigger batches since attention is O(S²)). ~3× throughput improvement.
            texts.sort(key=len)
            _cold_time = time.time() - _t0_cold
            print(f"[Pipeline] Cold start done in {_cold_time:.1f}s — "
                  f"{len(texts):,} docs + {len(_preloaded_models)} models ready.")

            # Save corpus sample (non-blocking, tiny)
            corpus_path = OUTPUT_DIR / "corpus_texts.json"
            with open(corpus_path, 'w') as f:
                json.dump(texts[:1000], f)

            # ================================================================
            # PHASE 1: VRAM Calibration — find optimal batch size per GPU
            # ================================================================
            print(f"\n[Pipeline] Phase 1: VRAM calibration (target {TARGET_VRAM_GB}GB)...")
            _calibrated_batch = {}
            for _gid in range(len(_preloaded_models)):
                _m, _tok = _preloaded_models[_gid]
                _cal_profile = dict(_gpu_profile)
                _calibrated_batch[_gid] = _calibrate_batch_size(_m, _tok, _cal_profile)
            print(f"[Pipeline] Calibrated: "
                  f"{', '.join(f'GPU{g}={_calibrated_batch[g]}' for g in _calibrated_batch)}")

            # ================================================================
            # PHASE 2: Concurrent Pipeline — GPU → CPU → Upload
            # ================================================================
            # Bounded queues enforce backpressure:
            # - extraction_q(maxsize=2): GPU blocks if CPU falls behind
            # - upload_q(maxsize=6): CPU blocks if upload falls behind
            # This keeps resource usage stable and predictable.
            print(f"\n[Pipeline] Phase 2: Starting pipeline "
                  f"(extraction_q={EXTRACTION_QUEUE_DEPTH}, "
                  f"upload_q={UPLOAD_QUEUE_DEPTH})...")

            # Oversubscribe CPU workers: each worker spends significant time
            # blocked on IPC (pickling large numpy arrays through pipes), so
            # having more workers than physical cores keeps all cores busy.
            # On 4 cores: 8 byte + 4 BPE = 12 workers total. The OS scheduler
            # ensures only 4 run at any instant, but the overlap of IPC waits
            # with compute keeps CPU utilization near 400%.
            _total_cpu = max(2, os.cpu_count() or 2)
            _byte_workers = max(2, _total_cpu * 2)   # 2x cores for byte (heavy + IPC stalls)
            _bpe_workers = max(1, _total_cpu)         # 1x cores for BPE (lighter)
            print(f"  [CPU] {_total_cpu} cores: {_byte_workers} byte + {_bpe_workers} BPE workers (oversubscribed)")
            _extraction_q = Queue(maxsize=EXTRACTION_QUEUE_DEPTH)
            _upload_q = Queue(maxsize=UPLOAD_QUEUE_DEPTH)
            _gpu_error = [None]
            _upload_error = [None]
            _upload_success_count = [0]   # total shards successfully uploaded
            _upload_fail_count = [0]      # total shard upload failures
            _chunks_since_reset = [0]
            _upload_done = threading.Event()
            _last_ckpt_shard = [0]  # track last checkpointed shard for every-N-shard checkpoint
            _last_upload_shard = [0]  # track last uploaded shard for batched uploads

            # ---- Persistent worker pools (no per-batch fork overhead) ----
            # max_tasks_per_child: restart workers after 50 tasks to free leaked memory
            # (Python processes accumulate fragmented heap over time; restart resets it)
            _bpe_pool = ProcessPoolExecutor(max_workers=_bpe_workers, max_tasks_per_child=50)
            _byte_pool = ProcessPoolExecutor(max_workers=_byte_workers, max_tasks_per_child=50)

            # ---- Upload Worker Thread ----
            def _upload_worker():
                """Dedicated upload thread: reads from upload_q, uploads to HF."""
                try:
                    while True:
                        item = _upload_q.get()
                        if item is None:  # sentinel
                            break
                        action, payload = item
                        try:
                            if action == 'shards':
                                _byte_dir, _bpe_dir = payload
                                _n1 = _upload_all_shards(_byte_dir, api, DATASET_REPO, "byte_marginalized")
                                _n2 = _upload_all_shards(_bpe_dir, api, DATASET_REPO, "bpe_tokenized")
                                _upload_success_count[0] += _n1 + _n2
                                gc.collect()  # free memory from uploaded shard data
                            elif action == 'checkpoint':
                                _ci, _bso, _pso, _bc, _pc, _so = payload
                                _save_stream_checkpoint(api, DATASET_REPO, _ci,
                                                        _bso, _pso, _bc, _pc,
                                                        shard_offset=_so)
                        except Exception as _e:
                            _upload_fail_count[0] += 1
                            print(f"  [Upload] *** ERROR #{_upload_fail_count[0]} ***: {_e}",
                                  flush=True)
                            traceback.print_exc()
                except Exception as _e:
                    _upload_error[0] = _e
                    print(f"  [Upload] *** FATAL: upload worker died: {_e} ***", flush=True)
                    traceback.print_exc()
                finally:
                    _upload_done.set()

            _upload_thread = threading.Thread(
                target=_upload_worker, daemon=True, name="upload-worker")
            _upload_thread.start()

            # ---- GPU Producer Thread ----
            _gpu_idle_total = [0.0]  # S11: total time GPU blocked on queue.put()

            def _gpu_producer():
                """Extract logprobs on GPU, push chunk info to extraction_q.

                The extraction_q has maxsize=EXTRACTION_QUEUE_DEPTH, so if CPU
                processing falls behind, this thread blocks on .put() — natural
                backpressure that prevents /tmp from filling up.
                """
                try:
                    n_chunks = (len(texts) + chunk_size - 1) // chunk_size
                    for _cs in range(0, len(texts), chunk_size):
                        _ce = min(_cs + chunk_size, len(texts))
                        _ci = _cs // chunk_size
                        if _ci < resume_from_chunk:
                            continue

                        # --- Model reset for VRAM defragmentation ---
                        _chunks_since_reset[0] += 1
                        if (_chunks_since_reset[0] > MODEL_RESET_EVERY_CHUNKS
                                and torch.cuda.is_available()):
                            _chunks_since_reset[0] = 0
                            print(f"\n  [GPU] Model reset triggered "
                                  f"(every {MODEL_RESET_EVERY_CHUNKS} chunks)...")
                            new_models = _reset_models(
                                _preloaded_models, _gpu_profile)
                            _preloaded_models.clear()
                            _preloaded_models.extend(new_models)
                            # Re-calibrate after reset
                            for _gid in range(len(_preloaded_models)):
                                _m, _tok = _preloaded_models[_gid]
                                _calibrated_batch[_gid] = _calibrate_batch_size(
                                    _m, _tok, dict(_gpu_profile))

                        _ctexts = texts[_cs:_ce]
                        _tmp = Path(f'/tmp/logprobs_chunk_{_ci:04d}.parquet')
                        _elapsed = time.time() - _pipeline_t0
                        print(f"\n  [GPU] Extracting chunk {_ci+1}/{n_chunks} "
                              f"(docs {_cs}-{_ce-1}) "
                              f"[{_elapsed/60:.1f}min elapsed]...", flush=True)
                        _extract_t0 = time.time()
                        run_logprob_extraction(
                            _ctexts, str(_tmp), resume=False,
                            preloaded_models=_preloaded_models)
                        _extract_elapsed = time.time() - _extract_t0
                        # S11: Track GPU idle time (blocked on queue.put)
                        _put_t0 = time.time()
                        _extraction_q.put((_ci, _cs, _ce))
                        _put_wait = time.time() - _put_t0
                        _gpu_idle_total[0] += _put_wait
                        if _put_wait > 0.5:
                            print(f"  [GPU IDLE] Blocked {_put_wait:.1f}s on extraction_q.put() "
                                  f"(CPU too slow) — extraction took {_extract_elapsed:.1f}s")
                        del _ctexts
                except Exception as _e:
                    traceback.print_exc()
                    _gpu_error[0] = _e
                finally:
                    _extraction_q.put(None)  # sentinel

            _gpu_thread = threading.Thread(
                target=_gpu_producer, name="gpu-producer", daemon=True)
            _gpu_thread.start()

            chunk_idx = max(0, resume_from_chunk - 1)
            _cpu_idle_total = 0.0  # S11: total time CPU blocked on extraction_q.get()

            # ---- CPU Consumer: byte + BPE in parallel, then queue upload ----
            while True:
                # S11: Track CPU idle time (waiting for GPU)
                _get_t0 = time.time()
                _item = _extraction_q.get()
                _get_wait = time.time() - _get_t0
                _cpu_idle_total += _get_wait
                if _item is None:
                    break
                if _get_wait > 0.5:
                    print(f"  [CPU IDLE] Blocked {_get_wait:.1f}s on extraction_q.get() "
                          f"(GPU too slow)")
                chunk_idx, chunk_start, chunk_end = _item
                total_chunks += 1

                _chunk_t0 = time.time()
                print(f"\n{'=' * 60}")
                print(f"CHUNK {chunk_idx+1}: docs {chunk_start}-{chunk_end-1} "
                      f"[CPU processing while GPU extracts next]")
                print(f"{'=' * 60}")

                # Parquet path used as glob stem
                tmp_parquet = Path(f'/tmp/logprobs_chunk_{chunk_idx:04d}.parquet')
                _tmp_stem = tmp_parquet.stem
                _part_files = sorted(tmp_parquet.parent.glob(f"{_tmp_stem}_part*.parquet"))
                if not _part_files:
                    raise RuntimeError(
                        f"Extraction failed for chunk {chunk_idx}: "
                        f"no part files found matching /tmp/{_tmp_stem}_part*.parquet")

                # Run byte + BPE in parallel threads (both CPU-only, no VRAM)
                _byte_res = [None]
                _bpe_res = [None]
                _cpu_errs = []

                def _byte_worker(_bc=byte_carry, _bso=byte_seq_offset):
                    try:
                        _byte_res[0] = build_byte_level_dataset(
                            str(tmp_parquet), str(byte_dir),
                            seq_offset=_bso, carry_state=_bc,
                            n_workers=_byte_workers,
                            shard_offset=shard_offset,
                            executor=_byte_pool)
                    except Exception as _e:
                        _cpu_errs.append(("byte", _e))

                def _bpe_worker(_pc=bpe_carry, _pso=bpe_seq_offset, _pool=_bpe_pool):
                    try:
                        _bpe_res[0] = build_bpe_dataset(
                            str(tmp_parquet), str(bpe_dir),
                            seq_offset=_pso, carry_state=_pc,
                            n_workers=_bpe_workers,
                            executor=_pool,
                            shard_offset=shard_offset)
                    except Exception as _e:
                        _cpu_errs.append(("bpe", _e))

                _bt = threading.Thread(target=_byte_worker, name="byte-proc")
                _pt = threading.Thread(target=_bpe_worker, name="bpe-proc")
                _byte_t0 = time.time()
                _bt.start(); _pt.start()

                # S3: Prefetch next chunk's parquet into pagecache while byte+BPE run
                _next_ci = chunk_idx + 1
                _next_tmp = Path(f'/tmp/logprobs_chunk_{_next_ci:04d}.parquet')
                _next_stem = _next_tmp.stem
                _next_parts = sorted(_next_tmp.parent.glob(f"{_next_stem}_part*.parquet"))
                if _next_parts:
                    for _nf in _next_parts:
                        try:
                            with open(_nf, 'rb') as _fh:
                                while _fh.read(1024 * 1024):  # 1MB reads → pagecache
                                    pass
                        except Exception:
                            pass  # file may not exist yet — that's fine

                _bt.join(); _pt.join()
                _cpu_proc_elapsed = time.time() - _byte_t0

                if _cpu_errs:
                    for _n, _e in _cpu_errs:
                        print(f"  [ERROR] {_n}: {_e}")
                    raise _cpu_errs[0][1]

                b_stats, byte_carry = _byte_res[0]
                p_stats, bpe_carry = _bpe_res[0]
                byte_seq_offset += b_stats['n_sequences']
                bpe_seq_offset += p_stats['n_sequences']
                total_byte_coverage += b_stats.get('coverage_pct', 0)

                # Clean up part files AFTER both builders finish (safe — no race)
                for _pf in _part_files:
                    _pf.unlink(missing_ok=True)

                _chunk_elapsed = time.time() - _chunk_t0
                _docs_in_chunk = chunk_end - chunk_start
                _docs_per_sec = _docs_in_chunk / max(_chunk_elapsed, 0.01)

                # Queue upload every UPLOAD_EVERY_SHARDS complete shards
                # (byte+BPE together so HF always has complete sets)
                _current_upload_shard = byte_seq_offset // SEQS_PER_SHARD
                if _current_upload_shard >= _last_upload_shard[0] + UPLOAD_EVERY_SHARDS:
                    _last_upload_shard[0] = _current_upload_shard
                    _upload_q.put(('shards', (byte_dir, bpe_dir)))

                # S12: Per-chunk resource summary
                _vram_msg = ""
                if torch.cuda.is_available():
                    for _gi in range(torch.cuda.device_count()):
                        free, total = torch.cuda.mem_get_info(_gi)
                        used = (total - free) / 1024**3
                        frag = _check_vram_fragmentation(_gi)
                        _vram_msg += f" GPU{_gi}={used:.1f}GB(frag={frag:.0%})"

                gc.collect()
                _ram_used, _ram_total, _ram_frac = _get_ram_usage()
                _check_ram_pressure("chunk-end")

                # RAM backpressure: if too high, wait for upload worker to drain shards
                if _ram_used > RAM_HARD_LIMIT_GB:
                    print(f"  [RAM] {_ram_used:.1f}GB > {RAM_HARD_LIMIT_GB}GB limit, "
                          f"waiting for upload queue to drain...", flush=True)
                    while not _upload_q.empty():
                        time.sleep(0.5)
                    gc.collect()
                    _ram_used, _ram_total, _ram_frac = _get_ram_usage()
                    print(f"  [RAM] After drain+GC: {_ram_used:.1f}/{_ram_total:.1f}GB")

                # S11+S12: Combined resource + idle summary
                try:
                    import psutil
                    _cpu_pct = psutil.cpu_percent(interval=0.1)
                except Exception:
                    _cpu_pct = -1
                print(f"  [Chunk {chunk_idx+1}] Done in {_chunk_elapsed:.1f}s "
                      f"(byte+BPE={_cpu_proc_elapsed:.1f}s, {_docs_per_sec:.1f} docs/s)")
                print(f"    CPU={_cpu_pct:.0f}%{_vram_msg}  "
                      f"RAM={_ram_used:.1f}/{_ram_total:.1f}GB({_ram_frac*100:.0f}%)")
                print(f"    Idle totals: GPU_blocked={_gpu_idle_total[0]:.1f}s "
                      f"CPU_blocked={_cpu_idle_total:.1f}s")
                print(f"    Uploads: {_upload_success_count[0]} ok, "
                      f"{_upload_fail_count[0]} failed, "
                      f"queue={_upload_q.qsize()}/{UPLOAD_QUEUE_DEPTH}")

                # Early warning if upload worker died
                if _upload_done.is_set() and _upload_error[0]:
                    print(f"  [Upload] *** UPLOAD WORKER DIED: {_upload_error[0]} ***",
                          flush=True)

                # Checkpoint every CHECKPOINT_EVERY_SHARDS complete shard pairs
                _current_shard = byte_seq_offset // SEQS_PER_SHARD
                if (_current_shard >= _last_ckpt_shard[0] + CHECKPOINT_EVERY_SHARDS
                        and _current_shard > 0):
                    _last_ckpt_shard[0] = _current_shard
                    _upload_q.put(('checkpoint', (
                        chunk_idx, byte_seq_offset, bpe_seq_offset,
                        byte_carry, bpe_carry, shard_offset)))
                    print(f"  [checkpoint] Triggered at shard {_current_shard} "
                          f"(every {CHECKPOINT_EVERY_SHARDS})")

            _gpu_thread.join()
            _bpe_pool.shutdown(wait=False)
            _byte_pool.shutdown(wait=False)

            # S11: Final pipeline idle summary
            _pipeline_elapsed = time.time() - _pipeline_t0
            print(f"\n{'=' * 60}")
            print(f"PIPELINE IDLE SUMMARY ({_pipeline_elapsed/60:.1f}min total)")
            print(f"  GPU blocked on queue.put(): {_gpu_idle_total[0]:.1f}s "
                  f"({_gpu_idle_total[0]/_pipeline_elapsed*100:.1f}% of pipeline)")
            print(f"  CPU blocked on queue.get(): {_cpu_idle_total:.1f}s "
                  f"({_cpu_idle_total/_pipeline_elapsed*100:.1f}% of pipeline)")
            if _gpu_idle_total[0] > _cpu_idle_total:
                print(f"  → BOTTLENECK: CPU is slower than GPU "
                      f"(GPU waited {_gpu_idle_total[0]-_cpu_idle_total:.1f}s more)")
            elif _cpu_idle_total > _gpu_idle_total[0]:
                print(f"  → BOTTLENECK: GPU is slower than CPU "
                      f"(CPU waited {_cpu_idle_total-_gpu_idle_total[0]:.1f}s more)")
            else:
                print(f"  → Pipeline perfectly balanced")
            print(f"{'=' * 60}")

            if _gpu_error[0]:
                raise _gpu_error[0]

            # Queue final uploads for any remaining shards
            _upload_q.put(('shards', (byte_dir, bpe_dir)))

            # Queue final checkpoint
            _upload_q.put(('checkpoint', (
                chunk_idx, byte_seq_offset, bpe_seq_offset,
                byte_carry, bpe_carry, shard_offset)))

            # Signal upload worker to finish and wait
            _upload_q.put(None)  # sentinel
            _upload_thread.join(timeout=300)  # 5 min max for final uploads

            if _upload_error[0]:
                print(f"  [WARN] Upload worker had errors: {_upload_error[0]}")

            # Upload any truly remaining shards (safety net)
            print("\n  Uploading remaining shards (safety net)...")
            _upload_all_shards(byte_dir, api, DATASET_REPO, "byte_marginalized")
            _upload_all_shards(bpe_dir, api, DATASET_REPO, "bpe_tokenized")

            # Save split info and upload
            _save_split_info(byte_seq_offset, byte_dir)
            _save_split_info(bpe_seq_offset, bpe_dir)
            for info_file in [byte_dir / 'split_info.json', bpe_dir / 'split_info.json']:
                if info_file.exists():
                    subdir = "byte_marginalized" if "byte" in info_file.parent.name else "bpe_tokenized"
                    api.upload_file(
                        path_or_fileobj=str(info_file),
                        path_in_repo=f"{subdir}/split_info.json",
                        repo_id=DATASET_REPO, repo_type="dataset",
                    )

            # Save projection matrix and upload
            proj = _get_projection_matrix(TEACHER_HIDDEN_DIM)
            proj_path = OUTPUT_DIR / "projection_matrix.npy"
            np.save(str(proj_path), proj.numpy())
            api.upload_file(
                path_or_fileobj=str(proj_path),
                path_in_repo="projection_matrix.npy",
                repo_id=DATASET_REPO, repo_type="dataset",
            )
            proj_path.unlink(missing_ok=True)

            byte_stats = {'n_sequences': byte_seq_offset,
                          'coverage_pct': total_byte_coverage / max(total_chunks, 1)}
            bpe_stats = {'n_sequences': bpe_seq_offset}

            _total_elapsed = time.time() - _pipeline_t0
            print(f"\n[Pipeline] Total time: {_total_elapsed/60:.1f} min "
                  f"({_total_elapsed/3600:.2f} hours)")
            print(f"[Pipeline] Cold start: {_cold_time:.1f}s, "
                  f"Chunks: {total_chunks}, "
                  f"Throughput: {len(texts)/_total_elapsed:.1f} docs/s")

        else:
            run_logprob_extraction(texts, str(logprobs_path), resume=True)

    if not stream_to_hub:
        if skip_extraction:
            assert logprobs_path.exists(), (
                f"Logprobs not found at {logprobs_path}. "
                f"Run without --skip_extraction first."
            )

        # Step 3a: Byte-level dataset
        print(f"\n{'=' * 60}")
        print("STEP 3a: Building byte-level dataset (runs 08-15)")
        print(f"{'=' * 60}")

        byte_dir = OUTPUT_DIR / "byte_marginalized"
        byte_stats, _ = build_byte_level_dataset(
            str(logprobs_path),
            str(byte_dir),
        )
        _save_split_info(byte_stats['n_sequences'], byte_dir)

        # Step 3b: BPE dataset
        print(f"\n{'=' * 60}")
        print("STEP 3b: Building BPE dataset (runs 01-07)")
        print(f"{'=' * 60}")

        bpe_dir = OUTPUT_DIR / "bpe_tokenized"
        bpe_stats, _ = build_bpe_dataset(
            str(logprobs_path),
            str(bpe_dir),
        )
        _save_split_info(bpe_stats['n_sequences'], bpe_dir)

        # Save projection matrix for reproducibility
        proj = _get_projection_matrix(TEACHER_HIDDEN_DIM)
        np.save(str(OUTPUT_DIR / "projection_matrix.npy"), proj.numpy())

        # Step 4: Upload o unmount
        if use_hf_mount and mount_path is not None:
            print(f"\n{'=' * 60}")
            print("STEP 4: Datos ya en HF Bucket (hf-mount, sin upload separado)")
            print(f"{'=' * 60}")
            from hf_mount import unmount
            unmount(mount_path)
        else:
            print(f"\n{'=' * 60}")
            print("STEP 4: Uploading to HuggingFace Hub")
            print(f"{'=' * 60}")
            upload_dataset_to_hub()

    # Summary
    print(f"\n{'=' * 60}")
    print("DATASET READY — Summary")
    print(f"{'=' * 60}")
    print(f"  Byte-level: {byte_stats['n_sequences']:,} sequences")
    print(f"  BPE:        {bpe_stats['n_sequences']:,} sequences")
    print(f"  Coverage teacher (byte): {byte_stats['coverage_pct']:.1f}%")
    if stream_to_hub:
        print(f"  Hub: https://huggingface.co/datasets/{DATASET_REPO} (streaming mode)")
    elif use_hf_mount:
        print(f"  Bucket: {DATASET_BUCKET} (via hf-mount)")
    else:
        print(f"  Hub: https://huggingface.co/datasets/{DATASET_REPO}")
    print(f"\n  All 15 ablation runs can start now.")
    print(f"  Teacher hidden (projected {ALIGN_DIM}d) included for EmbeddingAligner.")
    print(f"  Teacher does NOT need to run again.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        # No arguments -> run byte_marginalization tests
        print("Sin argumentos -> corriendo tests\n")
        print("=" * 60)
        test_alignment_cases()
        print()
        test_first_byte_marginalization()
        print()
        test_document_processing()
        print()
        test_document_packing()
        print("\n" + "=" * 60)
        print("Todos los tests pasaron.")
    elif '--logprobs-dir' in sys.argv or '--validate' in sys.argv:
        # Byte marginalization mode
        args = parse_args()
        if args.validate:
            builder = DistillationDatasetBuilder(
                logprobs_dir=args.logprobs_dir or ".",
                output_dir=args.output_dir,
                seq_len=args.seq_len,
            )
            builder.validate_sample()
        elif args.logprobs_dir:
            builder = DistillationDatasetBuilder(
                logprobs_dir=args.logprobs_dir,
                output_dir=args.output_dir,
                seq_len=args.seq_len,
            )
            builder.build()
        else:
            print("Error: --logprobs-dir requerido")
    else:
        # Full pipeline mode
        parser = argparse.ArgumentParser(
            description="Build shared dataset for CPU-1 ablation study"
        )
        parser.add_argument('--n_docs', type=int, default=50_000,
                            help="Number of FineWeb documents")
        parser.add_argument('--skip_extraction', action='store_true',
                            help="Skip logprob extraction (if already exists)")
        parser.add_argument('--only_upload', action='store_true',
                            help="Only upload existing dataset to Hub")
        parser.add_argument('--hf-mount', action='store_true',
                            help="Write dataset directly to HF Bucket via hf-mount "
                                 "(no local copy, no upload step)")
        parser.add_argument('--stream-to-hub', action='store_true',
                            help="Stream dataset directly to HF Hub chunk-by-chunk "
                                 "(enables checkpointing, required for long runs on Kaggle)")
        parser.add_argument('--shard-offset', type=int, default=0,
                            help="Shard index offset — use 1 if series 0000 is already "
                                 "on HF Hub (shard_0001.parquet, shard_0002.parquet, ...)")
        parser.add_argument('--lightning-l4', action='store_true',
                            help="Force L4 profile (Lightning AI 24GB L4): "
                                 f"FP8 W8A8 quantization, batch {BATCH_SIZE_TEACHER_L4}, seq {MAX_EXTRACTION_TOKENS_L4}")
        args = parser.parse_args()

        _forced_profile = None
        if args.lightning_l4:
            _forced_profile = {
                'name': 'NVIDIA L4 (forced)', 'vram_gb': 24.0,
                'batch_size': BATCH_SIZE_TEACHER_L4,
                'max_tokens': MAX_EXTRACTION_TOKENS_L4,
                'fp8': True, 'dtype': torch.bfloat16,
                'logit_chunk_size': 512,
                'cpu_workers': 4,
            }
            print("[Config] --lightning-l4: L4 profile forced "
                  f"(batch={BATCH_SIZE_TEACHER_L4}, seq={MAX_EXTRACTION_TOKENS_L4}, fp8=True)")

        if args.only_upload:
            upload_dataset_to_hub()
        else:
            build_full_dataset(n_docs=args.n_docs,
                               skip_extraction=args.skip_extraction,
                               use_hf_mount=args.hf_mount,
                               stream_to_hub=args.stream_to_hub,
                               gpu_profile_override=_forced_profile,
                               shard_offset=args.shard_offset)
