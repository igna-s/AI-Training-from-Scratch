"""
Extraccion offline de logprobs desde un modelo teacher para distillation.

Soporta dos backends:
- Ollama (local, top-20 logprobs, lento pero simple)
- Transformers (GPU, distribucion completa, rapido)

Uso:
    # Ollama (default)
    python extract_logprobs.py --input corpus/ --output logprobs/ --model qwen2.5:14b

    # Transformers (GPU, mas rapido)
    python extract_logprobs.py --input corpus/ --output logprobs/ \\
        --backend transformers --model Qwen/Qwen2.5-14B

    # Validar output existente
    python extract_logprobs.py --output logprobs/ --validate

    # Sin argumentos: correr tests
    python extract_logprobs.py
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq


# =============================================================================
# 1. OllamaTeacher -- Extraccion via Ollama API
# =============================================================================

class OllamaTeacher:
    """
    Extrae logprobs de un modelo teacher corriendo en Ollama.

    Estrategia: enviar prefijos incrementales, pedir 1 token de prediccion
    cada vez. Lento pero correcto (teacher forcing manual).

    Ollama soporta maximo top_logprobs=20.
    """

    def __init__(
        self,
        model: str = "qwen2.5:14b",
        endpoint: str = "http://localhost:11434",
        top_logprobs: int = 20,
        timeout: int = 120,
    ):
        """
        Args:
            model:        tag del modelo en Ollama (ej: "qwen2.5:14b")
            endpoint:     URL del servidor Ollama local
            top_logprobs: cantidad de logprobs a pedir (max 20 en Ollama)
            timeout:      timeout en segundos por request HTTP
        """
        self.model = model
        self.endpoint = endpoint.rstrip('/')
        self.top_logprobs = top_logprobs
        self.timeout = timeout

    def get_next_token_logprobs(self, prefix: str) -> dict | None:
        """
        Envia prefix al teacher y obtiene logprobs del siguiente token.

        Returns:
            dict con 'token', 'top_logprobs' list, o None si falla.
            Cada entry en top_logprobs: {'token': str, 'bytes': list[int], 'logprob': float}
        """
        import requests

        # temperature=0 para obtener la distribucion real del teacher (no muestreada).
        # num_predict=1 porque solo necesitamos los logprobs del siguiente token,
        # no generar multiples tokens.
        payload = {
            "model": self.model,
            "prompt": prefix,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 1,
            },
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
        }

        try:
            resp = requests.post(
                f"{self.endpoint}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[WARN] Ollama request failed: {e}")
            return None

        # Extraer logprobs de la respuesta.
        # Ollama retorna top_logprobs como dict {token_str: logprob} por posicion generada.
        if 'logprobs' not in data:
            return None

        logprobs_data = data['logprobs']
        if not logprobs_data or 'top_logprobs' not in logprobs_data:
            return None

        top = logprobs_data['top_logprobs']
        if not top or len(top) == 0:
            return None

        # top[0] es un dict de token -> logprob para la primera posicion generada.
        # Convertimos cada token a sus bytes UTF-8 para la posterior marginalizacion.
        result = []
        for token_str, logprob in top[0].items():
            token_bytes = list(token_str.encode('utf-8'))
            result.append({
                'token': token_str,
                'bytes': token_bytes,       # bytes UTF-8 del token (para first-byte marginal)
                'logprob': float(logprob),  # log-probabilidad del teacher
            })

        predicted_token = data.get('response', '')
        return {
            'predicted_token': predicted_token,
            'top_logprobs': result,
        }

    def extract_full_sequence(
        self,
        text: str,
        max_tokens: int | None = None,
        progress_callback=None,
    ) -> list[dict]:
        """
        Extrae logprobs para toda la secuencia via prefijos incrementales.

        Para cada posicion i, envia text[:i] y obtiene P(next_token | text[:i]).

        Returns:
            Lista de dicts, uno por posicion con logprobs.
        """
        results = []
        text_bytes = text.encode('utf-8')
        byte_pos = 0

        while byte_pos < len(text_bytes):
            # Teacher forcing: siempre usar el texto real como prefijo,
            # no la prediccion del teacher (que puede divergir del texto)
            prefix = text_bytes[:byte_pos].decode('utf-8', errors='replace')

            data = self.get_next_token_logprobs(prefix)
            if data is None:
                break

            # Registrar logprobs para esta posicion
            results.append({
                'byte_position': byte_pos,
                'top_tokens': data['top_logprobs'],
                'ground_truth_byte': text_bytes[byte_pos],
            })

            # Avanzar: usar el largo del token predicho como estimacion
            # del paso de tokenizacion del teacher
            predicted = data.get('predicted_token', '')
            step = len(predicted.encode('utf-8')) if predicted else 1
            byte_pos += max(step, 1)

            if progress_callback:
                progress_callback(byte_pos, len(text_bytes))

            if max_tokens and len(results) >= max_tokens:
                break

        return results


# =============================================================================
# 2. TransformersTeacher -- Extraccion via HuggingFace Transformers
# =============================================================================

class TransformersTeacher:
    """
    Extrae logprobs usando transformers con forward pass completo.
    Mucho mas rapido que Ollama (un forward = todos los logits).
    Da la distribucion COMPLETA sobre el vocabulario, no solo top-20.

    Requiere GPU con >=12GB VRAM para Qwen-14B en NF4.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-14B",
        load_in_4bit: bool = True,
        device_map: str = "auto",
        top_k: int = 100,
    ):
        """
        Args:
            model_name:   nombre del modelo en HuggingFace Hub
            load_in_4bit: cuantizar a NF4 para caber en ~12GB VRAM (T4/L4)
            device_map:   "auto" distribuye capas entre GPUs disponibles
            top_k:        top-k logprobs a guardar por posicion (100 >> Ollama's 20)
        """
        self.model_name = model_name
        self.top_k = top_k
        self._load_model(model_name, load_in_4bit, device_map)

    def _load_model(self, model_name: str, load_in_4bit: bool, device_map: str):
        """Carga modelo y tokenizer. NF4 reduce VRAM de ~28GB a ~8GB para 14B params."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = {"device_map": device_map, "torch_dtype": torch.float16}
        if load_in_4bit:
            # BitsAndBytes NF4: cuantizacion 4-bit con compute en FP16.
            # Reduce VRAM drasticamente con perdida minima de calidad.
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()

    def extract_full_sequence(
        self,
        text: str,
        max_length: int = 2048,
    ) -> list[dict]:
        """
        Forward pass completo sobre el texto.
        Retorna logprobs para cada posicion de token.

        Returns:
            Lista de dicts con token info y logprobs.
        """
        import torch

        # Tokenizar el texto completo de una vez (no incremental como Ollama).
        # truncation=True corta a max_length si el texto es muy largo.
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=max_length)
        input_ids = inputs["input_ids"].to(self.model.device)

        # Un solo forward pass produce logits para TODAS las posiciones.
        # Ventaja sobre Ollama: O(1) forwards vs O(N) forwards.
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits  # [1, seq_len, vocab_size]

        # log_softmax convierte logits crudos a log-probabilidades.
        # Usar log-space evita underflow numerico con probabilidades muy chicas.
        log_probs = torch.log_softmax(logits[0], dim=-1)  # [seq_len, vocab]

        results = []
        tokens = input_ids[0].tolist()
        byte_pos = 0

        for i in range(len(tokens) - 1):
            # Modelo autoregresivo: logits[i] predice token[i+1].
            # Iteramos hasta len-1 porque el ultimo token no predice nada.
            pos_logprobs = log_probs[i]  # [vocab_size]

            # Top-k: guardar solo los k tokens mas probables (ahorra disco)
            topk_values, topk_indices = pos_logprobs.topk(self.top_k)

            top_tokens = []
            for val, idx in zip(topk_values.tolist(), topk_indices.tolist()):
                token_str = self.tokenizer.decode([idx])
                token_bytes = list(token_str.encode('utf-8'))
                top_tokens.append({
                    'token': token_str,
                    'token_id': idx,
                    'bytes': token_bytes,
                    'logprob': val,
                })

            # Reconstruir posicion en bytes: necesario para alinear con patches.
            # El tokenizer BPE trabaja en tokens, pero el student trabaja en bytes.
            current_token_str = self.tokenizer.decode([tokens[i]])
            current_bytes = list(current_token_str.encode('utf-8'))

            # byte_position = donde empieza el token PREDICHO (i+1),
            # no el actual (i). La prediccion en posicion i aplica al
            # siguiente token, cuyo primer byte esta en byte_pos + len(current).
            # Este byte_position se usa despues para alinear con patch boundaries.
            predicted_byte_pos = byte_pos + len(current_bytes)

            results.append({
                'token_position': i,
                'byte_position': predicted_byte_pos,
                'token_str': current_token_str,
                'token_bytes': current_bytes,
                'top_tokens': top_tokens,
                'ground_truth_token_id': tokens[i + 1],
            })

            byte_pos = predicted_byte_pos

        return results


# =============================================================================
# 3. BytePatchAligner -- Alineamiento BPE -> byte patches
# =============================================================================

class BytePatchAligner:
    """
    Calcula boundaries de patches de bytes y determina alineamiento
    con tokens BPE del teacher.
    """

    def __init__(self, patch_size: int = 4):
        self.patch_size = patch_size

    def find_bpe_byte_boundaries(self, tokens_with_bytes: list[dict]) -> list[int]:
        """
        Calcula posiciones en bytes donde empiezan los tokens BPE.

        Args:
            tokens_with_bytes: lista de dicts con 'token_bytes' (list[int])
        Returns:
            lista de posiciones de byte donde empieza cada token
        """
        boundaries = []
        pos = 0
        for token_info in tokens_with_bytes:
            boundaries.append(pos)
            pos += len(token_info.get('token_bytes', token_info.get('bytes', [])))
        return boundaries

    def identify_aligned_positions(
        self,
        byte_boundaries: list[int],
        total_bytes: int,
    ) -> list[bool]:
        """
        Para cada boundary de patch, determina si coincide con un boundary BPE.

        Returns:
            lista de bools, uno por patch boundary (excluyendo posicion 0)
        """
        bpe_set = set(byte_boundaries)
        n_patches = (total_bytes + self.patch_size - 1) // self.patch_size
        aligned = []
        for p in range(1, n_patches):
            patch_start = p * self.patch_size
            aligned.append(patch_start in bpe_set)
        return aligned

    def compute_first_byte_marginals(
        self,
        top_tokens: list[dict],
    ) -> dict[int, float]:
        """
        First-byte marginalization: convierte P(next_token) en P(next_byte).

        Para cada token candidato, toma SOLO su primer byte y acumula probabilidad.
        Si multiples tokens empiezan con el mismo byte, suma sus probabilidades.
        Normaliza sobre los bytes observados.

        Args:
            top_tokens: lista de {token, bytes, logprob}
        Returns:
            dict byte_value -> probabilidad normalizada
        """
        byte_probs: dict[int, float] = {}
        for item in top_tokens:
            token_bytes = item.get('bytes', [])
            if not token_bytes:
                continue
            first_byte = token_bytes[0]
            prob = math.exp(item['logprob'])
            byte_probs[first_byte] = byte_probs.get(first_byte, 0.0) + prob

        # Normalizar (top-20/top-k no cubre 100% de la masa)
        total = sum(byte_probs.values())
        if total > 0:
            byte_probs = {k: v / total for k, v in byte_probs.items()}

        return byte_probs


# =============================================================================
# 4. Guardado a Parquet
# =============================================================================

def save_to_parquet(records: list[dict], output_path: str):
    """
    Guarda registros de logprobs en formato Parquet con compresion zstd.

    Args:
        records:     lista de dicts con 'doc_id', 'text_bytes', 'positions'
        output_path: ruta del archivo .parquet de salida

    Formato columnar:
        doc_id:     str — identificador del documento
        text_bytes: binary — texto original como bytes crudos
        positions:  large_string — JSON serializado de los logprobs por posicion
        n_tokens:   int32 — cantidad de posiciones con logprobs (para stats rapidos)
    """
    if not records:
        return

    doc_ids = []
    text_bytes_col = []
    positions_col = []
    n_tokens_col = []

    for rec in records:
        doc_ids.append(rec.get('doc_id', ''))
        text_bytes_col.append(rec.get('text_bytes', b''))
        positions_col.append(json.dumps(rec.get('positions', [])))
        n_tokens_col.append(len(rec.get('positions', [])))

    table = pa.table({
        'doc_id': pa.array(doc_ids, type=pa.string()),
        'text_bytes': pa.array(text_bytes_col, type=pa.binary()),
        'positions': pa.array(positions_col, type=pa.large_string()),
        'n_tokens': pa.array(n_tokens_col, type=pa.int32()),
    })

    pq.write_table(table, output_path, compression='zstd')


def load_from_parquet(path: str) -> list[dict]:
    """Carga registros de logprobs desde Parquet. Inversa de save_to_parquet."""
    table = pq.read_table(path)
    records = []
    for i in range(len(table)):
        records.append({
            'doc_id': table['doc_id'][i].as_py(),
            'text_bytes': table['text_bytes'][i].as_py(),
            'positions': json.loads(table['positions'][i].as_py()),
            'n_tokens': table['n_tokens'][i].as_py(),
        })
    return records


# =============================================================================
# 5. Procesamiento del corpus
# =============================================================================

def iter_corpus(input_path: str) -> Iterator[tuple[str, str]]:
    """
    Itera sobre documentos del corpus.
    Soporta: directorio con .txt, archivo unico, o un .txt con un doc por linea.

    Yields:
        (doc_id, text)
    """
    p = Path(input_path)

    if p.is_dir():
        for f in sorted(p.glob("**/*.txt")):
            text = f.read_text(encoding='utf-8', errors='replace').strip()
            if text:
                yield f.stem, text
    elif p.is_file():
        # Un documento por linea (lineas vacias se ignoran)
        doc_idx = 0
        with open(p, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield f"doc_{doc_idx:06d}", line
                    doc_idx += 1
    else:
        raise FileNotFoundError(f"No encontrado: {input_path}")


def get_progress_file(output_dir: str) -> Path:
    """Ruta al archivo de progreso para resume."""
    return Path(output_dir) / "_progress.json"


def load_progress(output_dir: str) -> set[str]:
    """Carga doc_ids ya procesados."""
    pf = get_progress_file(output_dir)
    if pf.exists():
        with open(pf) as f:
            return set(json.load(f))
    return set()


def save_progress(output_dir: str, processed: set[str]):
    """Guarda doc_ids procesados."""
    pf = get_progress_file(output_dir)
    with open(pf, 'w') as f:
        json.dump(sorted(processed), f)


def process_corpus(args: argparse.Namespace):
    """
    Pipeline principal de extraccion de logprobs.

    Flujo:
        1. Instanciar teacher (Ollama o Transformers)
        2. Iterar documentos del corpus
        3. Extraer logprobs + calcular first-byte marginals
        4. Guardar en batches de Parquet con compresion zstd

    Resume automático: si se interrumpe, recarga doc_ids ya procesados
    desde _progress.json y salta los completados.
    """
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume support: carga set de doc_ids ya procesados
    processed = load_progress(str(output_dir))
    if processed:
        print(f"Resumiendo: {len(processed)} documentos ya procesados")

    # Crear teacher segun backend elegido
    if args.backend == 'ollama':
        teacher = OllamaTeacher(
            model=args.model,
            endpoint=args.endpoint,
            top_logprobs=args.top_logprobs,
        )
    else:
        teacher = TransformersTeacher(
            model_name=args.model,
            load_in_4bit=args.load_in_4bit,
            top_k=args.top_k_transformers,
        )

    aligner = BytePatchAligner(patch_size=4)

    # Procesar documentos
    batch_records = []
    batch_size = args.save_every
    total_docs = 0
    total_tokens = 0

    for doc_id, text in iter_corpus(args.input):
        if doc_id in processed:
            continue

        t0 = time.time()

        # Extraer logprobs
        positions = teacher.extract_full_sequence(text)
        if not positions:
            print(f"  [SKIP] {doc_id}: sin logprobs")
            processed.add(doc_id)
            continue

        # Calcular first-byte marginals para cada posicion.
        # Convierte P(next_token) -> P(next_byte) tomando solo el primer byte
        # de cada token candidato. Esto permite entrenar el student a nivel byte.
        for pos in positions:
            pos['first_byte_probs'] = aligner.compute_first_byte_marginals(
                pos['top_tokens']
            )

        record = {
            'doc_id': doc_id,
            'text_bytes': text.encode('utf-8'),
            'positions': positions,
        }
        batch_records.append(record)
        processed.add(doc_id)

        total_docs += 1
        total_tokens += len(positions)
        elapsed = time.time() - t0

        print(f"  [{total_docs}] {doc_id}: {len(positions)} tokens, "
              f"{elapsed:.1f}s ({len(positions)/max(elapsed,0.01):.0f} tok/s)")

        # Guardar batch
        if len(batch_records) >= batch_size:
            batch_file = output_dir / f"logprobs_{total_docs:06d}.parquet"
            save_to_parquet(batch_records, str(batch_file))
            save_progress(str(output_dir), processed)
            print(f"  -> Guardado {batch_file.name} ({len(batch_records)} docs)")
            batch_records = []

    # Guardar sobrantes
    if batch_records:
        batch_file = output_dir / f"logprobs_{total_docs:06d}.parquet"
        save_to_parquet(batch_records, str(batch_file))
        save_progress(str(output_dir), processed)

    print(f"\nCompleto: {total_docs} docs, {total_tokens} tokens")
    print(f"Output: {output_dir}")


# =============================================================================
# 6. Validacion
# =============================================================================

def validate_output(output_dir: str, n_samples: int = 100):
    """
    Valida la calidad del output de extraccion.

    Args:
        output_dir: directorio con archivos logprobs_*.parquet
        n_samples:  posiciones a muestrear para la validacion

    Verifica que la masa de probabilidad de los top-k tokens es razonable.
    Si masa < 0.5 en muchas posiciones, el teacher tiene alta entropia y
    conviene usar mas top-k (o backend transformers con distribucion completa).
    """
    output_path = Path(output_dir)
    parquet_files = sorted(output_path.glob("logprobs_*.parquet"))

    if not parquet_files:
        print(f"No hay archivos parquet en {output_dir}")
        return

    total_positions = 0
    total_mass = 0.0
    high_entropy_count = 0

    for pf in parquet_files:
        records = load_from_parquet(str(pf))
        for rec in records:
            for pos in rec['positions'][:n_samples]:
                top_tokens = pos.get('top_tokens', [])
                if not top_tokens:
                    continue

                mass = sum(math.exp(t['logprob']) for t in top_tokens)
                total_mass += mass
                total_positions += 1

                if mass < 0.5:
                    high_entropy_count += 1

            if total_positions >= n_samples:
                break
        if total_positions >= n_samples:
            break

    if total_positions == 0:
        print("Sin posiciones para validar")
        return

    avg_mass = total_mass / total_positions
    print(f"Validacion ({total_positions} posiciones):")
    print(f"  Masa promedio top-k: {avg_mass:.4f}")
    print(f"  Posiciones con masa < 0.5: {high_entropy_count} ({high_entropy_count/total_positions:.1%})")
    print(f"  {'[OK]' if avg_mass > 0.8 else '[WARN]'} "
          f"Masa promedio {'OK' if avg_mass > 0.8 else 'baja (considerar mas tokens o backend transformers)'}")


# =============================================================================
# 7. Tests
# =============================================================================

def test_byte_patch_aligner():
    """Verifica alineamiento BPE -> byte patches."""
    aligner = BytePatchAligner(patch_size=4)

    # Simular tokens BPE para "Hello world"
    tokens = [
        {'token_bytes': [72, 101, 108, 108, 111]},    # "Hello" (5 bytes)
        {'token_bytes': [32, 119, 111, 114, 108, 100]},  # " world" (6 bytes)
    ]

    boundaries = aligner.find_bpe_byte_boundaries(tokens)
    assert boundaries == [0, 5], f"Boundaries: {boundaries}"

    aligned = aligner.identify_aligned_positions(boundaries, total_bytes=11)
    # Patch boundaries: 4, 8 (excluyendo 0)
    # BPE boundaries: 0, 5
    # Patch 1 empieza en byte 4 -> NO coincide con BPE (0 o 5)
    # Patch 2 empieza en byte 8 -> NO coincide
    assert aligned == [False, False], f"Aligned: {aligned}"

    print("[OK] BytePatchAligner: boundaries correctas")


def test_first_byte_marginals():
    """Verifica first-byte marginalization."""
    aligner = BytePatchAligner()

    top_tokens = [
        {'token': 'the', 'bytes': [116, 104, 101], 'logprob': -0.5},
        {'token': 'The', 'bytes': [84, 104, 101], 'logprob': -1.2},
        {'token': 'to', 'bytes': [116, 111], 'logprob': -1.5},
    ]

    probs = aligner.compute_first_byte_marginals(top_tokens)

    # P(116) = exp(-0.5) + exp(-1.5) = 0.6065 + 0.2231 = 0.8296
    # P(84)  = exp(-1.2) = 0.3012
    # Total = 1.1308
    # Normalized: P(116) = 0.7336, P(84) = 0.2664
    assert abs(probs[116] - 0.7336) < 0.001, f"P(116) = {probs[116]}"
    assert abs(probs[84] - 0.2664) < 0.001, f"P(84) = {probs[84]}"
    assert abs(sum(probs.values()) - 1.0) < 0.001

    print("[OK] First-byte marginalization: valores correctos")


def test_parquet_roundtrip():
    """Verifica guardado/carga Parquet."""
    import tempfile

    records = [
        {
            'doc_id': 'test_001',
            'text_bytes': b'Hello world',
            'positions': [
                {
                    'byte_position': 0,
                    'top_tokens': [
                        {'token': 'H', 'bytes': [72], 'logprob': -0.1},
                    ],
                    'ground_truth_byte': 72,
                }
            ],
        }
    ]

    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        tmp_path = f.name

    try:
        save_to_parquet(records, tmp_path)
        loaded = load_from_parquet(tmp_path)

        assert len(loaded) == 1
        assert loaded[0]['doc_id'] == 'test_001'
        assert loaded[0]['text_bytes'] == b'Hello world'
        assert len(loaded[0]['positions']) == 1
        assert loaded[0]['positions'][0]['byte_position'] == 0
    finally:
        os.unlink(tmp_path)

    print("[OK] Parquet roundtrip")


def test_corpus_iterator():
    """Verifica iterador de corpus."""
    import tempfile

    # Crear archivo temporal con lineas
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                     encoding='utf-8') as f:
        f.write("Hello world\n")
        f.write("Second document\n")
        f.write("\n")  # linea vacia (debe ignorarse)
        f.write("Third doc\n")
        tmp_path = f.name

    try:
        docs = list(iter_corpus(tmp_path))
        assert len(docs) == 3
        assert docs[0] == ('doc_000000', 'Hello world')
        assert docs[1] == ('doc_000001', 'Second document')
        assert docs[2] == ('doc_000002', 'Third doc')
    finally:
        os.unlink(tmp_path)

    print("[OK] Corpus iterator")


# =============================================================================
# 8. CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extraccion de logprobs del teacher")

    p.add_argument('--input', type=str, default=None,
                   help="Ruta al corpus (directorio con .txt o archivo)")
    p.add_argument('--output', type=str, default='./logprobs',
                   help="Directorio de salida para Parquet")

    # Backend
    p.add_argument('--backend', choices=['ollama', 'transformers'], default='ollama')

    # Ollama
    p.add_argument('--model', type=str, default='qwen2.5:14b',
                   help="Modelo teacher (Ollama tag o HF model name)")
    p.add_argument('--endpoint', type=str, default='http://localhost:11434')
    p.add_argument('--top-logprobs', type=int, default=20)

    # Transformers
    p.add_argument('--load-in-4bit', action='store_true', default=True)
    p.add_argument('--top-k-transformers', type=int, default=100,
                   help="Top-k logprobs a guardar (transformers)")

    # Procesamiento
    p.add_argument('--save-every', type=int, default=100,
                   help="Guardar a Parquet cada N documentos")
    p.add_argument('--validate', action='store_true',
                   help="Validar output existente en --output")

    return p.parse_args()


def main():
    import sys

    if len(sys.argv) == 1:
        print("Sin argumentos -> corriendo tests\n")
        print("=" * 60)
        test_byte_patch_aligner()
        print()
        test_first_byte_marginals()
        print()
        test_parquet_roundtrip()
        print()
        test_corpus_iterator()
        print("\n" + "=" * 60)
        print("Todos los tests pasaron.")
        return

    args = parse_args()

    if args.validate:
        validate_output(args.output)
        return

    if args.input is None:
        print("Error: --input es requerido para extraccion")
        return

    process_corpus(args)


if __name__ == "__main__":
    main()
