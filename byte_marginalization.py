"""
First-byte marginalization y construccion del dataset de distillation.

Toma los logprobs crudos del teacher (de extract_logprobs.py) y produce
el dataset final alineado con patches de 4 bytes para entrenar CPU-1.

Resuelve el problema de alineamiento BPE <-> byte patches:
- Caso 1: patch boundary = BPE boundary -> senal teacher (first-byte marginal)
- Caso 2: patch boundary dentro de un token BPE -> solo CE (byte deterministico)
- Caso 3: multiples tokens BPE dentro de un patch -> senal solo en el primero

Uso:
    # Construir dataset de distillation
    python byte_marginalization.py --logprobs-dir ./logprobs --output-dir ./dataset

    # Validar dataset
    python byte_marginalization.py --output-dir ./dataset --validate

    # Sin argumentos: correr tests
    python byte_marginalization.py
"""

import argparse
import enum
import json
import math
import os
from pathlib import Path
from typing import NamedTuple

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None


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

    def build(self):
        """
        Pipeline completo: lee logprobs -> procesa docs -> empaqueta -> guarda Parquet.

        Flujo:
            1. Lee todos los logprobs_*.parquet de logprobs_dir
            2. Procesa cada documento: bytes -> patches + teacher signals
            3. Empaqueta documentos en secuencias de seq_len patches
            4. Guarda como distillation_dataset.parquet con zstd
        """
        if pq is None:
            raise ImportError("pyarrow requerido: pip install pyarrow")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cargar todos los Parquet de logprobs
        parquet_files = sorted(self.logprobs_dir.glob("logprobs_*.parquet"))
        if not parquet_files:
            print(f"No hay archivos logprobs_*.parquet en {self.logprobs_dir}")
            return

        print(f"Procesando {len(parquet_files)} archivos de logprobs...")

        all_docs = []
        for pf in parquet_files:
            table = pq.read_table(pf)
            for i in range(len(table)):
                record = {
                    'doc_id': table['doc_id'][i].as_py(),
                    'text_bytes': table['text_bytes'][i].as_py(),
                    'positions': json.loads(table['positions'][i].as_py()),
                }
                processed = self.process_document(record)
                if processed:
                    all_docs.append(processed)

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
# 5. Tests
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


if __name__ == "__main__":
    main()
