"""
Dataset creation for CPU-1: BPE-to-byte alignment and dataset pipeline.

Section 1 - Core algorithms: BPE-to-byte alignment, first-byte marginalization,
            document packing, and dynamic patch marginalization.
Section 2 - Pipeline: teacher logprob extraction (Qwen-14B), byte-level and
            BPE dataset construction, and HuggingFace Hub upload.

Usage:
    # Full pipeline
    python create_dataset.py --n_docs 50000

    # Build byte marginalization dataset from logprobs
    python create_dataset.py --logprobs-dir ./logprobs --output-dir ./dataset
"""

import argparse
import enum
import json
import math
import os
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

TEACHER_MODEL = "Qwen/Qwen2.5-14B"
DATASET_REPO = "Cukinator/cpu1-ablation-dataset"
OUTPUT_DIR = Path("./shared_dataset")
TOP_K_LOGPROBS = 128
PATCH_SIZE = 4
SEQ_LEN_PATCHES = 5000
SEQ_LEN_TOKENS = 5000
ALIGN_DIM = 128             # Project teacher hidden 5120→128 (Johnson-Lindenstrauss)
SEED = 42
BATCH_SIZE_TEACHER = 4

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

def load_teacher(model_name: str = TEACHER_MODEL):
    """Load Qwen-14B in 4-bit NF4 for logprob extraction (~8GB VRAM)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[Teacher] Loading {model_name} in 4-bit NF4...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        device_map="cuda:0",
    )
    model.eval()
    print(f"[Teacher] Ready. Vocab size: {tokenizer.vocab_size}")
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

    Always extracts teacher hidden states (projected 5120→ALIGN_DIM)
    for EmbeddingAligner (runs 13, 15). Uses ~50% more VRAM but
    storage is small thanks to random orthogonal projection.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_idx = 0
    if resume and output_path.exists():
        existing = pq.read_table(str(output_path))
        start_idx = len(existing)
        print(f"[Extraction] Resuming from doc {start_idx}/{len(corpus_texts)}")

    model, tokenizer = load_teacher()
    batch_results = []

    for i in tqdm(range(start_idx, len(corpus_texts)), desc="Extracting logprobs"):
        text = corpus_texts[i]
        if not text.strip():
            continue

        result = extract_logprobs_and_hidden(
            model, tokenizer, text, TOP_K_LOGPROBS,
        )
        record = {
            'doc_id':     i,
            'text_bytes': result['text_bytes'],
            'positions':  json.dumps(result['positions']),
            'n_tokens':   result['n_tokens'],
            # Flatten [n_tokens, ALIGN_DIM] → [n_tokens * ALIGN_DIM]
            'teacher_hidden': [
                v for row in result['teacher_hidden'] for v in row
            ],
            'teacher_dim': result['teacher_dim'],
        }
        batch_results.append(record)

        # Save every 100 docs for crash safety
        if len(batch_results) >= 100:
            _append_to_parquet(batch_results, output_path)
            batch_results = []
            print(f"  Saved through doc {i}")

    if batch_results:
        _append_to_parquet(batch_results, output_path)

    print(f"[Extraction] Complete: {output_path}")


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
) -> dict:
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
    rows = logprobs_table.to_pylist()

    # Autodetectar workers si no se especifica
    if n_workers <= 0:
        n_workers = max(1, (os.cpu_count() or 4) - 1)

    all_patches = []
    all_targets = []
    all_teacher_probs = []
    all_teacher_mask = []
    all_teacher_hidden = []
    has_hidden = False
    teacher_dim = 0
    stats = {'n_sequences': 0, 'n_patches_total': 0, 'n_with_teacher': 0}

    # Descomposición de datos: cada worker procesa un documento independiente
    print(f"[Byte dataset] Marginalizing with {n_workers} workers...")
    if n_workers > 1 and len(rows) > 10:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(tqdm(
                executor.map(_marginalize_one_row, rows, chunksize=max(1, len(rows) // (n_workers * 4))),
                total=len(rows), desc="Marginalizing BPE→byte (parallel)",
            ))
    else:
        results = [_marginalize_one_row(r) for r in tqdm(rows, desc="Marginalizing BPE→byte")]

    # Aglomerar resultados y empaquetar en secuencias
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
            hidden_slice = all_teacher_hidden[:seq_len] if has_hidden else None
            _save_byte_sequence(
                all_patches[:seq_len], all_targets[:seq_len],
                all_teacher_probs[:seq_len], all_teacher_mask[:seq_len],
                output_dir, stats['n_sequences'],
                teacher_hidden=hidden_slice, teacher_dim=teacher_dim,
            )
            all_patches = all_patches[seq_len:]
            all_targets = all_targets[seq_len:]
            all_teacher_probs = all_teacher_probs[seq_len:]
            all_teacher_mask = all_teacher_mask[seq_len:]
            if has_hidden:
                all_teacher_hidden = all_teacher_hidden[seq_len:]
            stats['n_sequences'] += 1

    coverage = stats['n_with_teacher'] / max(stats['n_patches_total'], 1) * 100
    stats['coverage_pct'] = coverage

    print(f"[Byte dataset] {stats['n_sequences']} sequences")
    print(f"[Byte dataset] Teacher coverage: {coverage:.1f}%")

    _save_split_info(stats['n_sequences'], output_dir)
    return stats


def _save_byte_sequence(patches, targets, teacher_probs, teacher_mask,
                        output_dir: Path, seq_idx: int,
                        teacher_hidden=None, teacher_dim: int = 0) -> None:
    """Save one byte-level sequence as a single Parquet row (flat arrays)."""
    # Store the entire sequence (seq_len patches) as ONE row with flat arrays.
    # Shapes: patches/targets → [seq_len*4], teacher_probs → [seq_len*256],
    # teacher_mask → [seq_len]. The Dataset wrapper reshapes on load.
    # Optional: teacher_hidden → [seq_len*teacher_dim] float32
    flat_patches = [b for p in patches for b in p]
    flat_targets = [b for t in targets for b in t]
    flat_probs = [v for row in teacher_probs for v in row]

    columns = {
        'patches':       pa.array([flat_patches], type=pa.list_(pa.uint8())),
        'targets':       pa.array([flat_targets], type=pa.list_(pa.uint8())),
        'teacher_probs': pa.array([flat_probs],   type=pa.list_(pa.float32())),
        'teacher_mask':  pa.array([teacher_mask], type=pa.list_(pa.bool_())),
    }
    if teacher_hidden is not None:
        flat_hidden = [v for row in teacher_hidden for v in row]
        columns['teacher_hidden'] = pa.array(
            [flat_hidden], type=pa.list_(pa.float32()),
        )
        columns['teacher_dim'] = pa.array([teacher_dim], type=pa.int32())
    table = pa.table(columns)
    shard = seq_idx // 1000
    shard_path = output_dir / f"shard_{shard:04d}.parquet"
    if shard_path.exists():
        existing = pq.read_table(str(shard_path))
        pq.write_table(pa.concat_tables([existing, table]), str(shard_path))
    else:
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
    from transformers import AutoTokenizer
    if not hasattr(_tokenize_bpe_row, '_tokenizer'):
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
) -> dict:
    """
    Build BPE dataset from the same teacher logprobs.
    Produces dataset for runs 01-07 (BPE).
    Paralelizado con ProcessPoolExecutor: cada worker tokeniza un doc independiente.

    Teacher logprobs are kept in BPE space (no marginalization).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logprobs_table = pq.read_table(logprobs_parquet)
    rows = logprobs_table.to_pylist()

    if n_workers <= 0:
        n_workers = max(1, (os.cpu_count() or 4) - 1)

    stats = {'n_sequences': 0, 'n_tokens_total': 0}
    buffer_ids = []
    buffer_probs = []
    buffer_top_ids = []
    buffer_mask = []

    # Descomposición de datos: cada worker procesa un documento independiente
    print(f"[BPE dataset] Tokenizing with {n_workers} workers...")
    work_args = [(row, TEACHER_MODEL, seq_len, TOP_K_LOGPROBS) for row in rows]

    if n_workers > 1 and len(rows) > 10:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(tqdm(
                executor.map(_tokenize_bpe_row, work_args,
                             chunksize=max(1, len(rows) // (n_workers * 4))),
                total=len(rows), desc="Building BPE dataset (parallel)",
            ))
    else:
        results = [_tokenize_bpe_row(a) for a in tqdm(work_args, desc="Building BPE dataset")]

    # Aglomerar resultados y empaquetar en secuencias
    for result in results:
        buffer_ids.extend(result['ids'])
        buffer_probs.extend(result['probs'])
        buffer_top_ids.extend(result['top_ids'])
        buffer_mask.extend(result['mask'])
        stats['n_tokens_total'] += len(result['ids'])

        while len(buffer_ids) >= seq_len + 1:
            _save_bpe_sequence(
                buffer_ids[:seq_len],
                buffer_ids[1:seq_len + 1],
                buffer_probs[:seq_len],
                buffer_top_ids[:seq_len],
                buffer_mask[:seq_len],
                output_dir,
                stats['n_sequences'],
            )
            buffer_ids = buffer_ids[seq_len:]
            buffer_probs = buffer_probs[seq_len:]
            buffer_top_ids = buffer_top_ids[seq_len:]
            buffer_mask = buffer_mask[seq_len:]
            stats['n_sequences'] += 1

    print(f"[BPE dataset] {stats['n_sequences']} sequences")
    _save_split_info(stats['n_sequences'], output_dir)
    return stats


def _save_bpe_sequence(input_ids, labels, teacher_probs, teacher_ids_top,
                       teacher_mask, output_dir: Path, seq_idx: int) -> None:
    """Save one BPE sequence as a single Parquet row (flat arrays).

    teacher_probs and teacher_ids_top are stored flat ([seq_len * TOP_K])
    so the Dataset wrapper can reshape them to [seq_len, TOP_K] on load.
    """
    flat_probs = [v for row in teacher_probs for v in row]
    flat_ids = [v for row in teacher_ids_top for v in row]

    table = pa.table({
        'input_ids':        pa.array([input_ids],  type=pa.list_(pa.int32())),
        'labels':           pa.array([labels],     type=pa.list_(pa.int32())),
        'teacher_probs_bpe': pa.array([flat_probs], type=pa.list_(pa.float32())),
        'teacher_ids_bpe':  pa.array([flat_ids],   type=pa.list_(pa.int32())),
        'teacher_mask_bpe': pa.array([teacher_mask], type=pa.list_(pa.bool_())),
    })
    shard = seq_idx // 1000
    shard_path = output_dir / f"shard_{shard:04d}.parquet"
    if shard_path.exists():
        existing = pq.read_table(str(shard_path))
        pq.write_table(pa.concat_tables([existing, table]), str(shard_path))
    else:
        pq.write_table(table, str(shard_path))


# ============================================================
# STEP 3: Split info & upload
# ============================================================

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


def upload_dataset_to_hub(local_dir: str = str(OUTPUT_DIR)) -> None:
    """Upload complete dataset to HuggingFace Hub."""
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
                       skip_extraction: bool = False) -> None:
    """
    Complete dataset construction pipeline.

    Step 1: Download FineWeb corpus
    Step 2: Extract logprobs + projected hidden states with Qwen-14B
    Step 3a: Marginalize BPE→byte (with teacher_hidden for runs 13, 15)
    Step 3b: Build BPE dataset
    Step 4: Upload to HuggingFace Hub

    Only needs to run ONCE. Covers all 15 ablation runs.
    """
    from datasets import load_dataset

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logprobs_path = OUTPUT_DIR / "teacher_logprobs" / "logprobs_raw.parquet"

    if not skip_extraction:
        # Step 1: Download corpus
        print(f"\n{'=' * 60}")
        print("STEP 1: Downloading FineWeb corpus")
        print(f"{'=' * 60}")

        fw = load_dataset(corpus_name, name="sample-10BT", split="train",
                          streaming=True)
        texts = []
        for i, doc in enumerate(fw):
            if i >= n_docs:
                break
            texts.append(doc['text'])

        print(f"  Corpus: {len(texts):,} documents")

        # Save corpus sample for reference
        corpus_path = OUTPUT_DIR / "corpus_texts.json"
        with open(corpus_path, 'w') as f:
            json.dump(texts[:1000], f)

        # Step 2: Extract logprobs (the slow part)
        print(f"\n{'=' * 60}")
        print("STEP 2: Extracting teacher logprobs (Qwen-14B)")
        print("  (~6-10 hours on T4, supports resume)")
        print(f"{'=' * 60}")

        run_logprob_extraction(texts, str(logprobs_path), resume=True)
    else:
        assert logprobs_path.exists(), (
            f"Logprobs not found at {logprobs_path}. "
            f"Run without --skip_extraction first."
        )

    # Step 3a: Byte-level dataset
    print(f"\n{'=' * 60}")
    print("STEP 3a: Building byte-level dataset (runs 08-15)")
    print(f"{'=' * 60}")

    byte_stats = build_byte_level_dataset(
        str(logprobs_path),
        str(OUTPUT_DIR / "byte_marginalized"),
    )

    # Step 3b: BPE dataset
    print(f"\n{'=' * 60}")
    print("STEP 3b: Building BPE dataset (runs 01-07)")
    print(f"{'=' * 60}")

    bpe_stats = build_bpe_dataset(
        str(logprobs_path),
        str(OUTPUT_DIR / "bpe_tokenized"),
    )

    # Save projection matrix for reproducibility
    proj = _get_projection_matrix(5120)
    np.save(str(OUTPUT_DIR / "projection_matrix.npy"), proj.numpy())

    # Step 4: Upload
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
        args = parser.parse_args()

        if args.only_upload:
            upload_dataset_to_hub()
        else:
            build_full_dataset(n_docs=args.n_docs,
                               skip_extraction=args.skip_extraction)
