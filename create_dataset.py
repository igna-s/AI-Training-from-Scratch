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

import argparse
import enum
import gc
import json
import math
import os
import shutil
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
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

TEACHER_MODEL = "Qwen/Qwen2.5-7B"

# Module-level cache: tokenizer id → {token_id: bytes}
_TOKEN_BYTES_CACHE: dict = {}
DATASET_REPO = "Cukinator/cpu1-ablation-dataset"
DATASET_BUCKET = "Cukinator/cpu1-ablation-dataset"  # HF Bucket (read-write via hf-mount)
OUTPUT_DIR = Path("./shared_dataset")
TOP_K_LOGPROBS = 128
PATCH_SIZE = 4
SEQ_LEN_PATCHES = 5000
SEQ_LEN_TOKENS = 5000
MAX_EXTRACTION_TOKENS = 2048  # T4 SDPA usa O(seq²) → 4096 OOM con batch≥2; 2048 usa 0.9 GB vs 3.5 GB
TEACHER_HIDDEN_DIM = 3584   # Qwen2.5-7B hidden size
ALIGN_DIM = 128             # Project teacher hidden 3584→128 (Johnson-Lindenstrauss)
SEQS_PER_SHARD = 200        # Sequences per parquet shard
SEED = 42
BATCH_SIZE_TEACHER = 2  # T4 15GB: model ~4GB + activations ~3GB/doc → 2 docs safe

# Lazy-initialized random orthogonal projection matrix
_PROJECTION_MATRIX = None

def _get_projection_matrix(teacher_dim: int, align_dim: int = ALIGN_DIM):
    """Random orthogonal projection, seeded for reproducibility."""
    global _PROJECTION_MATRIX
    if _PROJECTION_MATRIX is None:
        rng = np.random.RandomState(SEED)
        rand_mat = rng.randn(teacher_dim, align_dim).astype(np.float32)
        q, _ = np.linalg.qr(rand_mat)
        _PROJECTION_MATRIX = torch.from_numpy(q[:, :align_dim])
    return _PROJECTION_MATRIX


# ============================================================
# STEP 1: Extract teacher logprobs
# ============================================================

def load_teacher(model_name: str = TEACHER_MODEL, device: str = "cuda:0"):
    """Load Qwen-7B in 4-bit NF4 (~4GB VRAM)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"[Teacher] Loading {model_name} in 4-bit NF4 on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
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
    )
    input_ids = enc['input_ids'].to(model.device)

    with torch.no_grad():
        logits = model(input_ids).logits[0]  # [seq_len, vocab_size]

    log_probs = torch.log_softmax(logits, dim=-1)
    top_logprobs, top_indices = torch.topk(log_probs, k=top_k, dim=-1)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    byte_offset = 0
    positions = []

    for pos_idx, (token_id, token_str) in enumerate(
        zip(input_ids[0].tolist(), tokens)
    ):
        try:
            token_byte_seq = tokenizer.convert_tokens_to_string(
                [token_str]
            ).encode('utf-8')
        except Exception:
            token_byte_seq = b'\x00'

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
        byte_offset += len(token_byte_seq)

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
    byte_offset = 0
    positions = []

    for pos_idx, (token_id, token_str) in enumerate(
        zip(input_ids[0].tolist(), tokens)
    ):
        try:
            token_byte_seq = tokenizer.convert_tokens_to_string(
                [token_str]
            ).encode('utf-8')
        except Exception:
            token_byte_seq = b'\x00'

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
        byte_offset += len(token_byte_seq)

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
    captured: dict = {}

    def _mid_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured['mid'] = h.detach()      # [B, S, D]

    def _last_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured['last'] = h.detach()     # [B, S, D]

    hook_mid  = model.model.layers[mid_idx].register_forward_hook(_mid_hook)
    hook_last = model.model.layers[-1].register_forward_hook(_last_hook)

    try:
        enc = tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=seq_len,
        )
        input_ids = enc['input_ids'].to(model.device)        # [B, S_pad]
        attn_mask = enc['attention_mask'].to(model.device)   # [B, S_pad]

        with torch.no_grad():
            # Forward populates captured via hooks; we discard the output logits
            model(input_ids, attention_mask=attn_mask)

    finally:
        hook_mid.remove()
        hook_last.remove()

    B, S_pad = input_ids.shape

    # ── Project mid-layer hidden → ALIGN_DIM (on CPU to save VRAM) ────────
    proj = _get_projection_matrix(model.config.hidden_size)
    mid_proj = (captured['mid'].float().cpu() @ proj).half()  # [B, S_pad, ALIGN_DIM]

    # ── Chunked logit computation from last-layer hidden + final norm ─────
    with torch.no_grad():
        last_normed = model.model.norm(captured['last'])   # [B, S_pad, D]

    all_top_idx: list = []
    all_top_lp:  list = []
    with torch.no_grad():
        for cs in range(0, S_pad, logit_chunk_size):
            ce = min(cs + logit_chunk_size, S_pad)
            chunk  = last_normed[:, cs:ce, :].contiguous()   # [B, C, D]
            logits = model.lm_head(chunk).float()             # [B, C, V]
            log_p  = torch.log_softmax(logits, dim=-1)
            top_lp, top_idx = torch.topk(log_p, k=top_k, dim=-1)
            all_top_idx.append(top_idx.cpu())
            all_top_lp.append(top_lp.cpu())
            del logits, log_p

    top_idx_all = torch.cat(all_top_idx, dim=1)   # [B, S_pad, K]
    top_lp_all  = torch.cat(all_top_lp,  dim=1)   # [B, S_pad, K]

    # Liberar tensores GPU que ya no se necesitan
    del last_normed, all_top_idx, all_top_lp
    del captured['mid'], captured['last']

    input_ids_cpu = input_ids.cpu()
    attn_mask_cpu = attn_mask.cpu()
    del input_ids, attn_mask

    # ── Build one result dict per sample ──────────────────────────────────
    results = []
    for b in range(B):
        actual_len  = int(attn_mask_cpu[b].sum().item())
        ids         = input_ids_cpu[b, :actual_len].tolist()

        byte_offset = 0
        positions   = []
        for pos_idx, token_id in enumerate(ids):
            token_bytes_seq = bytes_cache.get(token_id, b'\x00')

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
            byte_offset += len(token_bytes_seq)

        hidden_proj = mid_proj[b, :actual_len, :]   # [L, ALIGN_DIM]
        results.append({
            'text_bytes':    list(texts[b].encode('utf-8')),
            'positions':     positions,
            'n_tokens':      actual_len,
            'teacher_hidden': hidden_proj.tolist(),
            'teacher_dim':   int(hidden_proj.shape[-1]),
        })

    return results


def _append_to_parquet(records: list[dict], path: Path) -> None:
    """Append records to a Parquet file."""
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


def run_logprob_extraction(
    corpus_texts: list[str],
    output_path: str,
    resume: bool = True,
) -> None:
    """Extract logprobs + projected hidden states for entire corpus.

    Uses all available GPUs in parallel (one Qwen-7B teacher per GPU).
    Each T4 holds one 4-bit NF4 model (~4GB VRAM), so 2×T4 = 2× throughput.

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

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if n_gpus >= 2:
        print(f"[Extraction] Using {n_gpus} GPUs in parallel "
              f"({', '.join(torch.cuda.get_device_name(i) for i in range(n_gpus))})")
        _run_extraction_multi_gpu(corpus_texts, output_path, start_idx, n_gpus)
    else:
        _run_extraction_single_gpu(corpus_texts, output_path, start_idx)

    # Liberar teacher models de GPU después de la extracción
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
        vram_free = [torch.cuda.mem_get_info(i)[0] / (1024**3) for i in range(torch.cuda.device_count())]
        print(f"[Extraction] VRAM freed. Available: {['%.1fG' % v for v in vram_free]}")

    print(f"[Extraction] Complete: {output_path}")


def _extract_batch_with_oom_retry(model, tokenizer, batch_texts, gpu_tag="GPU 0"):
    """Try extraction, halve batch on OOM until batch_size=1."""
    import traceback
    sub_batches = [batch_texts]
    while sub_batches:
        texts = sub_batches.pop(0)
        try:
            return extract_logprobs_and_hidden_batch(
                model, tokenizer, texts,
                top_k=TOP_K_LOGPROBS, seq_len=MAX_EXTRACTION_TOKENS,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            if len(texts) > 1:
                mid = len(texts) // 2
                print(f"\n  [{gpu_tag}] OOM with {len(texts)} docs, splitting to {mid}+{len(texts)-mid}")
                r1 = _extract_batch_with_oom_retry(model, tokenizer, texts[:mid], gpu_tag)
                r2 = _extract_batch_with_oom_retry(model, tokenizer, texts[mid:], gpu_tag)
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


def _run_extraction_single_gpu(corpus_texts, output_path, start_idx, batch_size: int = BATCH_SIZE_TEACHER):
    """Single-GPU batched extraction (batch_size docs per forward pass)."""
    model, tokenizer = load_teacher()
    batch_results = []
    _FLUSH_EVERY = 50

    valid_indices = [i for i in range(start_idx, len(corpus_texts))
                     if corpus_texts[i].strip()]

    for batch_start in tqdm(range(0, len(valid_indices), batch_size),
                            desc="Extracting logprobs"):
        batch_indices = valid_indices[batch_start:batch_start + batch_size]
        batch_texts   = [corpus_texts[i] for i in batch_indices]

        batch_out = _extract_batch_with_oom_retry(model, tokenizer, batch_texts)

        for i, result in zip(batch_indices, batch_out):
            batch_results.append({
                'doc_id':         i,
                'text_bytes':     result['text_bytes'],
                'positions':      json.dumps(result['positions']),
                'n_tokens':       result['n_tokens'],
                'teacher_hidden': [v for row in result['teacher_hidden'] for v in row],
                'teacher_dim':    result['teacher_dim'],
            })
        del batch_out

        if len(batch_results) >= _FLUSH_EVERY:
            _append_to_parquet(batch_results, output_path)
            batch_results = []
            gc.collect()

    if batch_results:
        _append_to_parquet(batch_results, output_path)


def _run_extraction_multi_gpu(corpus_texts, output_path, start_idx, n_gpus):
    """Multi-GPU extraction: one Qwen-7B teacher per GPU, docs split evenly.

    Usa threading porque las operaciones CUDA liberan el GIL,
    permitiendo que ambas GPUs computen en paralelo.
    Lock para escritura thread-safe al parquet compartido.
    """
    output_path = Path(output_path)
    lock = threading.Lock()

    remaining_indices = list(range(start_idx, len(corpus_texts)))
    # Interleave para balancear carga (docs largos/cortos distribuidos)
    gpu_indices = [remaining_indices[i::n_gpus] for i in range(n_gpus)]

    # Cargar modelos SECUENCIALMENTE para evitar OOM por carga paralela en GPU 0
    print("[Extraction] Loading models sequentially...")
    models = []
    for gpu_id in range(n_gpus):
        m, tok = load_teacher(device=f"cuda:{gpu_id}")
        torch.cuda.empty_cache()
        models.append((m, tok))
    print("[Extraction] All models loaded. Starting parallel extraction...")

    _FLUSH_EVERY = 50  # flush to parquet every N docs to limit RAM (~500MB vs ~7GB)

    def worker(gpu_id, doc_indices, model, tokenizer, batch_size: int = BATCH_SIZE_TEACHER):
        batch_results = []

        valid_indices = [i for i in doc_indices if corpus_texts[i].strip()]

        for batch_start in tqdm(range(0, len(valid_indices), batch_size),
                                desc=f"GPU {gpu_id}", position=gpu_id):
            batch_indices = valid_indices[batch_start:batch_start + batch_size]
            batch_texts   = [corpus_texts[i] for i in batch_indices]

            batch_out = _extract_batch_with_oom_retry(
                model, tokenizer, batch_texts, gpu_tag=f"GPU {gpu_id}")

            for i, result in zip(batch_indices, batch_out):
                batch_results.append({
                    'doc_id':         i,
                    'text_bytes':     result['text_bytes'],
                    'positions':      json.dumps(result['positions']),
                    'n_tokens':       result['n_tokens'],
                    'teacher_hidden': [v for row in result['teacher_hidden'] for v in row],
                    'teacher_dim':    result['teacher_dim'],
                })
            del batch_out

            # Flush every N docs to keep RAM bounded
            if len(batch_results) >= _FLUSH_EVERY:
                with lock:
                    _append_to_parquet(batch_results, output_path)
                batch_results = []
                gc.collect()

        # Save remaining
        if batch_results:
            with lock:
                _append_to_parquet(batch_results, output_path)
            del batch_results
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  [GPU {gpu_id}] Extraction done", flush=True)

    threads = []
    for gpu_id in range(n_gpus):
        model, tokenizer = models[gpu_id]
        t = threading.Thread(target=worker, args=(gpu_id, gpu_indices[gpu_id], model, tokenizer))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()


# ============================================================
# STEP 2a: Marginalize BPE→Byte
# ============================================================

def _marginalize_one_row(row: dict) -> dict:
    """Marginalizar un doc BPE→byte. Diseñado para multiprocessing (sin estado compartido)."""
    text_bytes = bytes(row['text_bytes'])
    positions = json.loads(row['positions'])

    byte_to_teacher = {}
    for pos in positions:
        byte_off = pos['byte_offset']
        top_toks = pos['top_tokens']
        byte_probs = {}
        for item in top_toks:
            if not item['bytes']:
                continue
            first_byte = item['bytes'][0]
            prob = math.exp(item['logprob'])
            byte_probs[first_byte] = byte_probs.get(first_byte, 0.0) + prob
        total = sum(byte_probs.values())
        if total > 0:
            byte_probs = {k: v / total for k, v in byte_probs.items()}
        byte_to_teacher[byte_off] = byte_probs

    # Teacher hidden alignment (for EmbeddingAligner, runs 13/15)
    has_hidden = 'teacher_hidden' in row and row['teacher_hidden'] is not None
    teacher_dim = row.get('teacher_dim', 0) if has_hidden else 0
    byte_offset_to_pos = {}
    if has_hidden:
        for pos_idx, pos in enumerate(positions):
            byte_offset_to_pos[pos['byte_offset']] = pos_idx
        hidden_flat = row['teacher_hidden']

    n_patches = len(text_bytes) // PATCH_SIZE
    patches, targets, teacher_probs, teacher_mask = [], [], [], []
    teacher_hidden_patches = [] if has_hidden else None
    n_with_teacher = 0

    for p_idx in range(n_patches - 1):
        patch_start = p_idx * PATCH_SIZE
        target_start = (p_idx + 1) * PATCH_SIZE
        patches.append(list(text_bytes[patch_start:patch_start + PATCH_SIZE]))
        targets.append(list(text_bytes[target_start:target_start + PATCH_SIZE]))
        has_teacher = target_start in byte_to_teacher
        if has_teacher:
            bp = byte_to_teacher[target_start]
            teacher_probs.append([bp.get(b, 0.0) for b in range(256)])
            n_with_teacher += 1
        else:
            teacher_probs.append([0.0] * 256)
        teacher_mask.append(has_teacher)

        if has_hidden:
            if has_teacher and target_start in byte_offset_to_pos:
                pos_idx = byte_offset_to_pos[target_start]
                start = pos_idx * teacher_dim
                end = start + teacher_dim
                teacher_hidden_patches.append(hidden_flat[start:end])
            else:
                teacher_hidden_patches.append([0.0] * teacher_dim)

    result = {
        'patches': patches, 'targets': targets,
        'teacher_probs': teacher_probs, 'teacher_mask': teacher_mask,
        'n_with_teacher': n_with_teacher,
    }
    if has_hidden:
        result['teacher_hidden'] = teacher_hidden_patches
        result['teacher_dim'] = teacher_dim
    return result


def build_byte_level_dataset(
    logprobs_parquet: str,
    output_dir: str,
    seq_len: int = SEQ_LEN_PATCHES,
    n_workers: int = 0,
    seq_offset: int = 0,
    carry_state: dict = None,
) -> tuple:
    """
    Convert BPE logprobs to byte-level distributions.
    Parallelizado con ProcessPoolExecutor para saturar CPU multi-core.

    Output per sequence:
        patches:        [seq_len, 4] uint8
        targets:        [seq_len, 4] uint8
        teacher_probs:  [seq_len, 256] float32  (sparse)
        teacher_mask:   [seq_len] bool
        teacher_hidden: [seq_len, teacher_dim] float32  (optional, if extracted)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logprobs_table = pq.read_table(logprobs_parquet)
    n_total_rows = len(logprobs_table)

    # Autodetectar workers si no se especifica
    if n_workers <= 0:
        n_workers = max(1, (os.cpu_count() or 4) - 1)

    if carry_state is not None:
        all_patches = carry_state['patches']
        all_targets = carry_state['targets']
        all_teacher_probs = carry_state['teacher_probs']
        all_teacher_mask = carry_state['teacher_mask']
        all_teacher_hidden = carry_state['teacher_hidden']
        has_hidden = carry_state['has_hidden']
        teacher_dim = carry_state['teacher_dim']
    else:
        all_patches = []
        all_targets = []
        all_teacher_probs = []
        all_teacher_mask = []
        all_teacher_hidden = []
        has_hidden = False
        teacher_dim = 0
    stats = {'n_sequences': 0, 'n_patches_total': 0, 'n_with_teacher': 0}

    # Process in batches of 50 rows to limit RAM (each row ~10MB in pylist form)
    _MARGIN_BATCH = 50
    print(f"[Byte dataset] Marginalizing {n_total_rows} docs in batches of {_MARGIN_BATCH}...")
    results = []
    for batch_start in range(0, n_total_rows, _MARGIN_BATCH):
        batch_end = min(batch_start + _MARGIN_BATCH, n_total_rows)
        rows = logprobs_table.slice(batch_start, batch_end - batch_start).to_pylist()
        if n_workers > 1 and len(rows) > 10:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                batch_results = list(executor.map(
                    _marginalize_one_row, rows,
                    chunksize=max(1, len(rows) // (n_workers * 4)),
                ))
        else:
            batch_results = [_marginalize_one_row(r) for r in rows]
        results.extend(batch_results)
        del rows, batch_results
    del logprobs_table

    # Aglomerar resultados y empaquetar en secuencias
    # Collect records per shard, write in batch (avoids O(n²) read-concat-write)
    shard_records: dict[int, list[dict]] = {}

    for result in results:
        all_patches.extend(result['patches'])
        all_targets.extend(result['targets'])
        all_teacher_probs.extend(result['teacher_probs'])
        all_teacher_mask.extend(result['teacher_mask'])
        stats['n_with_teacher'] += result['n_with_teacher']
        stats['n_patches_total'] += len(result['patches'])
        if 'teacher_hidden' in result:
            has_hidden = True
            teacher_dim = result['teacher_dim']
            all_teacher_hidden.extend(result['teacher_hidden'])

        while len(all_patches) >= seq_len:
            seq_idx = seq_offset + stats['n_sequences']
            hidden_slice = all_teacher_hidden[:seq_len] if has_hidden else None
            record = _make_byte_record(
                all_patches[:seq_len], all_targets[:seq_len],
                all_teacher_probs[:seq_len], all_teacher_mask[:seq_len],
                teacher_hidden=hidden_slice, teacher_dim=teacher_dim,
            )
            shard = seq_idx // SEQS_PER_SHARD
            shard_records.setdefault(shard, []).append(record)
            all_patches = all_patches[seq_len:]
            all_targets = all_targets[seq_len:]
            all_teacher_probs = all_teacher_probs[seq_len:]
            all_teacher_mask = all_teacher_mask[seq_len:]
            if has_hidden:
                all_teacher_hidden = all_teacher_hidden[seq_len:]
            stats['n_sequences'] += 1

    # Batch write all records per shard
    for shard, records in shard_records.items():
        shard_path = output_dir / f"shard_{shard:04d}.parquet"
        _write_byte_shard(records, shard_path)

    coverage = stats['n_with_teacher'] / max(stats['n_patches_total'], 1) * 100
    stats['coverage_pct'] = coverage

    print(f"[Byte dataset] {stats['n_sequences']} sequences (chunk), total offset {seq_offset + stats['n_sequences']}")
    print(f"[Byte dataset] Teacher coverage: {coverage:.1f}%")

    next_carry = {
        'patches': all_patches, 'targets': all_targets,
        'teacher_probs': all_teacher_probs, 'teacher_mask': all_teacher_mask,
        'teacher_hidden': all_teacher_hidden,
        'has_hidden': has_hidden, 'teacher_dim': teacher_dim,
    }
    return stats, next_carry


def _make_byte_record(patches, targets, teacher_probs, teacher_mask,
                      teacher_hidden=None, teacher_dim: int = 0) -> dict:
    """Build a flat record dict for one byte-level sequence."""
    record = {
        'patches':       [b for p in patches for b in p],
        'targets':       [b for t in targets for b in t],
        'teacher_probs': [v for row in teacher_probs for v in row],
        'teacher_mask':  list(teacher_mask),
    }
    if teacher_hidden is not None:
        record['teacher_hidden'] = [v for row in teacher_hidden for v in row]
        record['teacher_dim'] = teacher_dim
    return record


def _write_byte_shard(records: list[dict], shard_path: Path) -> None:
    """Write multiple byte-level records to a shard parquet in one shot."""
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
    if shard_path.exists():
        existing = pq.read_table(str(shard_path))
        table = pa.concat_tables([existing, table])
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
) -> tuple:
    """
    Build BPE dataset from the same teacher logprobs.
    Produces dataset for runs 01-07 (BPE).
    Paralelizado con ProcessPoolExecutor: cada worker tokeniza un doc independiente.

    Teacher logprobs are kept in BPE space (no marginalization).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logprobs_table = pq.read_table(logprobs_parquet)
    n_total_rows = len(logprobs_table)

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

    # In-process tokenization in batches of 50 to limit RAM.
    global _BPE_TOKENIZER_CACHE
    if '_BPE_TOKENIZER_CACHE' not in globals() or _BPE_TOKENIZER_CACHE is None:
        from transformers import AutoTokenizer
        _BPE_TOKENIZER_CACHE = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    _bpe_tok = _BPE_TOKENIZER_CACHE

    _BPE_BATCH = 50
    print(f"[BPE dataset] Tokenizing {n_total_rows} docs in-process...")

    # Aglomerar resultados y empaquetar en secuencias
    shard_records: dict[int, list[dict]] = {}

    for batch_start in range(0, n_total_rows, _BPE_BATCH):
        batch_end = min(batch_start + _BPE_BATCH, n_total_rows)
        rows = logprobs_table.slice(batch_start, batch_end - batch_start).to_pylist()

        for row in rows:
            text_bytes = bytes(row['text_bytes'])
            positions = json.loads(row['positions'])
            text = text_bytes.decode('utf-8', errors='replace')
            enc = _bpe_tok(text, truncation=True, max_length=seq_len * 2)
            ids = enc['input_ids']

            for token_id, pos_data in zip(ids, positions[:len(ids)]):
                top_toks = pos_data.get('top_tokens', [])
                probs = [0.0] * TOP_K_LOGPROBS
                top_ids_k = [0] * TOP_K_LOGPROBS
                has_teacher = len(top_toks) > 0
                if has_teacher:
                    for k_idx, item in enumerate(top_toks[:TOP_K_LOGPROBS]):
                        probs[k_idx] = math.exp(item['logprob'])
                        top_ids_k[k_idx] = item.get('token_id', 0)
                buffer_ids.append(token_id)
                buffer_probs.append(probs)
                buffer_top_ids.append(top_ids_k)
                buffer_mask.append(has_teacher)
            stats['n_tokens_total'] += len(ids)

        del rows

        while len(buffer_ids) >= seq_len + 1:
            seq_idx = seq_offset + stats['n_sequences']
            record = _make_bpe_record(
                buffer_ids[:seq_len],
                buffer_ids[1:seq_len + 1],
                buffer_probs[:seq_len],
                buffer_top_ids[:seq_len],
                buffer_mask[:seq_len],
            )
            shard = seq_idx // SEQS_PER_SHARD
            shard_records.setdefault(shard, []).append(record)
            buffer_ids = buffer_ids[seq_len:]
            buffer_probs = buffer_probs[seq_len:]
            buffer_top_ids = buffer_top_ids[seq_len:]
            buffer_mask = buffer_mask[seq_len:]
            stats['n_sequences'] += 1

    del logprobs_table

    # Batch write all records per shard
    for shard, records in shard_records.items():
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
                            byte_carry: dict, bpe_carry: dict) -> None:
    """Guarda checkpoint en HF Hub para poder retomar si se corta la sesion."""
    import pickle, io
    checkpoint = {
        'last_completed_chunk': chunk_idx,
        'byte_seq_offset': byte_seq_offset,
        'bpe_seq_offset': bpe_seq_offset,
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
    """Upload ALL shards (sealed + incomplete). Returns count uploaded."""
    count = 0
    for shard_path in sorted(output_dir.glob("shard_*.parquet")):
        _hf_upload_with_retry(api, str(shard_path),
                              f"{repo_subdir}/{shard_path.name}", repo_id)
        shard_path.unlink()
        count += 1
    if count:
        print(f"  [hub] Uploaded {count} shards from {repo_subdir}/", flush=True)
    return count


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
                       n_docs: int = 50_000,
                       skip_extraction: bool = False,
                       use_hf_mount: bool = False,
                       stream_to_hub: bool = False,
                       chunk_size: int = 1000, **_kwargs) -> None:
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

    # chunk_size=500: ~250 docs/GPU, ~63 batches at batch_size=4.
    # RAM for 500-doc logprobs: ~1-2GB. Safe on Kaggle's 13GB.
    chunk_size = min(chunk_size, 500)
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

        DATASET_SEED = 42
        fw = load_dataset(corpus_name, name="sample-10BT", split="train",
                          streaming=True)
        # Shuffle con seed fijo para reproducibilidad
        fw = fw.shuffle(seed=DATASET_SEED, buffer_size=10_000)
        texts = []
        for i, doc in enumerate(fw):
            if i >= n_docs:
                break
            texts.append(doc['text'])

        print(f"  Corpus: {len(texts):,} documents (seed={DATASET_SEED})")

        # Save corpus sample for reference
        corpus_path = OUTPUT_DIR / "corpus_texts.json"
        with open(corpus_path, 'w') as f:
            json.dump(texts[:1000], f)

        # Step 2: Extract logprobs (the slow part)
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"\n{'=' * 60}")
        print("STEP 2: Extracting teacher logprobs (Qwen-7B)")
        if n_gpus >= 2:
            print(f"  {n_gpus} GPUs detectadas → extraccion en paralelo (~3-5 hours)")
        else:
            print("  (~6-10 hours on single T4, supports resume)")
        print(f"{'=' * 60}")

        if stream_to_hub:
            # === STREAMING MODE ===
            # Procesar en chunks: extract → byte/BPE → upload shards → delete local.
            # Evita acumular el parquet de logprobs completo en disco.
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
                byte_carry = carry['byte']
                bpe_carry = carry['bpe']
                print(f"  [resume] Saltando chunks 0-{resume_from_chunk - 1}, "
                      f"byte_seqs={byte_seq_offset}, bpe_seqs={bpe_seq_offset}")

            _pipeline_t0 = time.time()
            _next_forced_upload_min = 30  # first forced upload at 30 min
            _checkpoint_every = 5        # save checkpoint every N chunks

            for chunk_start in range(0, len(texts), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(texts))
                chunk_texts = texts[chunk_start:chunk_end]
                chunk_idx = chunk_start // chunk_size
                total_chunks += 1

                if chunk_idx < resume_from_chunk:
                    continue

                print(f"\n{'=' * 60}")
                print(f"CHUNK {chunk_idx+1}: docs {chunk_start}-{chunk_end-1} of {len(texts)}")
                print(f"{'=' * 60}")

                # Extract logprobs for this chunk → temp file
                tmp_parquet = Path(f'/tmp/logprobs_chunk_{chunk_idx:04d}.parquet')
                run_logprob_extraction(chunk_texts, str(tmp_parquet), resume=False)

                if not tmp_parquet.exists():
                    raise RuntimeError(
                        f"Extraction failed for chunk {chunk_idx}: "
                        f"{tmp_parquet} was not created."
                    )

                # Byte-level dataset
                print(f"\n  [Chunk {chunk_idx+1}] Building byte-level dataset...")
                b_stats, byte_carry = build_byte_level_dataset(
                    str(tmp_parquet), str(byte_dir),
                    seq_offset=byte_seq_offset, carry_state=byte_carry,
                )
                byte_seq_offset += b_stats['n_sequences']
                total_byte_coverage += b_stats.get('coverage_pct', 0)

                # Upload sealed byte shards
                _upload_sealed_shards(byte_dir, api, DATASET_REPO,
                                      "byte_marginalized", byte_seq_offset)

                # BPE dataset
                print(f"  [Chunk {chunk_idx+1}] Building BPE dataset...")
                p_stats, bpe_carry = build_bpe_dataset(
                    str(tmp_parquet), str(bpe_dir),
                    seq_offset=bpe_seq_offset, carry_state=bpe_carry,
                )
                bpe_seq_offset += p_stats['n_sequences']

                # Upload sealed BPE shards
                _upload_sealed_shards(bpe_dir, api, DATASET_REPO,
                                      "bpe_tokenized", bpe_seq_offset)

                # Periodic forced upload of ALL shards (including incomplete)
                elapsed_min = (time.time() - _pipeline_t0) / 60
                if elapsed_min >= _next_forced_upload_min:
                    print(f"\n  [FORCED UPLOAD @ {elapsed_min:.0f}min] Uploading all shards...")
                    _upload_all_shards(byte_dir, api, DATASET_REPO, "byte_marginalized")
                    _upload_all_shards(bpe_dir, api, DATASET_REPO, "bpe_tokenized")
                    _next_forced_upload_min = elapsed_min + 30

                # Delete temp logprobs chunk + free memory
                tmp_parquet.unlink(missing_ok=True)
                del chunk_texts
                gc.collect()
                if torch.cuda.is_available():
                    for _gi in range(torch.cuda.device_count()):
                        with torch.cuda.device(_gi):
                            torch.cuda.empty_cache()
                print(f"  [Chunk {chunk_idx+1}] Done. Byte: {byte_seq_offset} seqs, BPE: {bpe_seq_offset} seqs")

                # Guardar checkpoint cada N chunks (no en cada uno)
                if (chunk_idx + 1) % _checkpoint_every == 0:
                    _save_stream_checkpoint(api, DATASET_REPO, chunk_idx,
                                            byte_seq_offset, bpe_seq_offset,
                                            byte_carry, bpe_carry)

            # Final checkpoint before uploading remaining shards
            _save_stream_checkpoint(api, DATASET_REPO, chunk_idx,
                                    byte_seq_offset, bpe_seq_offset,
                                    byte_carry, bpe_carry)

            # Upload any remaining (incomplete) shards
            print("\n  Uploading remaining shards...")
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
        args = parser.parse_args()

        if args.only_upload:
            upload_dataset_to_hub()
        else:
            build_full_dataset(n_docs=args.n_docs,
                               skip_extraction=args.skip_extraction,
                               use_hf_mount=args.hf_mount)
