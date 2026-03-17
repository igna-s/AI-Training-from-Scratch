"""
Runner de inferencia para CPU-1.

Soporta:
- Carga de checkpoints
- Generacion de texto autoregresiva (4 bytes por step)
- Sampling con temperature / top-k / top-p
- Sliding window con position_ids absolutas
- Benchmark de throughput
- Modo interactivo (REPL)

Uso:
    # Generar texto desde un prompt
    python inference.py --checkpoint ckpt.pt --prompt "Hello world"

    # Modo interactivo
    python inference.py --checkpoint ckpt.pt --interactive

    # Benchmark de throughput
    python inference.py --benchmark --n-layers 2

    # Sin checkpoint (pesos random, para testing)
    python inference.py --prompt "Test" --max-patches 10 --n-layers 2
"""

import argparse
import time

import torch
import torch.nn.functional as F
from torch import Tensor

from model import (
    CPU1Model, ChunkEncoder, InContextRetrieval,
    PATCH_SIZE, VOCAB_SIZE, WINDOW_SIZE, N_SINKS, D_MODEL,
)


# =============================================================================
# 1. Encoding / Decoding
# =============================================================================

def text_to_patches(text: str) -> Tensor:
    """
    Convierte texto UTF-8 a patches de 4 bytes.

    Args:
        text: string de entrada
    Returns:
        [1, n_patches, 4] long
    """
    raw = text.encode('utf-8')
    n = len(raw)

    # Pad a multiplo de 4
    pad = (PATCH_SIZE - n % PATCH_SIZE) % PATCH_SIZE
    raw = raw + b'\x00' * pad

    n_patches = len(raw) // PATCH_SIZE
    patches = []
    for i in range(n_patches):
        chunk = raw[i * PATCH_SIZE:(i + 1) * PATCH_SIZE]
        patches.append(list(chunk))

    return torch.tensor(patches, dtype=torch.long).unsqueeze(0)  # [1, S, 4]


def patches_to_text(patches: Tensor) -> str:
    """
    Convierte patches a texto UTF-8.

    Args:
        patches: [S, 4] o [1, S, 4] long/uint8
    Returns:
        string decodificado
    """
    if patches.dim() == 3:
        patches = patches[0]
    flat = patches.reshape(-1).tolist()
    # Remover trailing zeros (padding)
    raw = bytes(flat).rstrip(b'\x00')
    return raw.decode('utf-8', errors='replace')


# =============================================================================
# 2. Sampling
# =============================================================================

def sample_byte(
    logits: Tensor,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
) -> int:
    """
    Samplea un byte (0-255) de los logits con temperature + top-k + top-p.

    Args:
        logits: [256] -- logits crudos para un byte
    Returns:
        byte value (0-255)
    """
    if temperature == 0.0:
        return logits.argmax().item()

    logits = logits / temperature

    # Top-k filtering: solo conservar los k bytes mas probables.
    # El resto se pone a -inf para que softmax les de prob ~0.
    if top_k > 0 and top_k < logits.shape[0]:
        topk_vals, _ = logits.topk(top_k)
        threshold = topk_vals[-1]
        logits = logits.where(logits >= threshold, torch.tensor(float('-inf'), device=logits.device))

    # Top-p (nucleus) filtering: mantener el conjunto minimo de tokens
    # cuya probabilidad acumulada supera top_p. Adaptativo al nivel de
    # certeza del modelo (mas tokens si distribucion plana, menos si peaked).
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(descending=True)
        cumprobs = torch.softmax(sorted_logits, dim=0).cumsum(dim=0)

        # Shift para mantener al menos el token mas probable (remove_mask[0] = False)
        remove_mask = cumprobs > top_p
        remove_mask[1:] = remove_mask[:-1].clone()
        remove_mask[0] = False
        sorted_logits[remove_mask] = float('-inf')

        # Restaurar orden original con scatter
        logits = torch.zeros_like(logits)
        logits.scatter_(0, sorted_idx, sorted_logits)

    probs = torch.softmax(logits, dim=0)
    return torch.multinomial(probs, 1).item()


# =============================================================================
# 3. Generador
# =============================================================================

class CPU1Generator:
    """
    Generador autoregresivo para CPU-1.

    Cada step genera 4 bytes (un patch) usando las 4 cabezas de prediccion.
    Usa sliding window con position_ids absolutas para manejar secuencias
    mas largas que la ventana de atencion.

    Sin KV cache (re-computa atencion cada step sobre la ventana activa).
    Para produccion, implementar KV cache circular de 516 slots por capa.
    """

    def __init__(
        self,
        model: CPU1Model,
        device: torch.device,
        max_context: int = WINDOW_SIZE + N_SINKS,
    ):
        """
        Args:
            model:       modelo CPU-1 ya cargado
            device:      dispositivo de computo (cuda/cpu)
            max_context: ventana maxima en patches (512 + 4 sinks = 516 por defecto)
        """
        self.model = model
        self.device = device
        self.max_context = max_context
        self.model.eval()

    def _build_context(self, patches: Tensor) -> tuple[Tensor, Tensor]:
        """
        Construye la ventana de contexto con position_ids absolutas.

        Si la secuencia es mas larga que max_context, mantiene:
        - Los N_SINKS primeros patches (posiciones 0..3)
        - Los ultimos (max_context - N_SINKS) patches

        Position_ids son absolutas (no relativas a la ventana).
        """
        S = patches.shape[1]

        # Secuencia cabe completa en la ventana -> sin truncamiento
        if S <= self.max_context:
            position_ids = torch.arange(S, device=self.device).unsqueeze(0)
            return patches, position_ids

        # Sliding window con attention sinks:
        # Los primeros N_SINKS patches (4) siempre se mantienen con sus posiciones originales.
        # Estos "sinks" acumulan atencion global y estabilizan la generacion.
        sinks = patches[:, :N_SINKS, :]
        sink_pos = torch.arange(N_SINKS, device=self.device)

        # Los patches mas recientes llenan el resto de la ventana.
        # Position IDs son absolutas (no 0-indexed), para que RoPE refleje la posicion real.
        n_recent = self.max_context - N_SINKS
        recent = patches[:, -n_recent:, :]
        recent_start = S - n_recent
        recent_pos = torch.arange(recent_start, S, device=self.device)

        # Contexto final: [sinks | recientes] con gap en position_ids
        context = torch.cat([sinks, recent], dim=1)
        position_ids = torch.cat([sink_pos, recent_pos]).unsqueeze(0)

        return context, position_ids

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_patches: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        stream: bool = False,
    ) -> str:
        """
        Genera texto a partir de un prompt.

        Args:
            prompt: texto de entrada (UTF-8)
            max_patches: patches a generar (x4 = bytes)
            temperature: 0 = greedy, >0 = sampling
            top_k: filtrar a top-k candidatos (0 = sin filtro)
            top_p: nucleus sampling threshold (1.0 = sin filtro)
            stream: si True, imprime bytes a medida que se generan
        Returns:
            texto completo (prompt + generado)
        """
        # Encode prompt
        patches = text_to_patches(prompt).to(self.device)

        if stream:
            print(prompt, end='', flush=True)

        for _ in range(max_patches):
            # Ventana de contexto + position_ids absolutas
            context, position_ids = self._build_context(patches)

            # Forward pass
            logits = self.model(context, position_ids)

            # Samplear 4 bytes (uno por head) del ultimo patch.
            # Cada head predice un byte distinto del siguiente patch:
            # head 0 -> byte 0, head 1 -> byte 1, etc.
            next_bytes = []
            for head_logits in logits:
                byte_val = sample_byte(
                    head_logits[0, -1, :],  # [256] logits del ultimo patch
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                next_bytes.append(byte_val)

            # Concatenar nuevo patch a la secuencia.
            # Sin KV cache: cada step re-procesa toda la ventana.
            new_patch = torch.tensor(
                [next_bytes], dtype=torch.long, device=self.device,
            ).unsqueeze(0)  # [1, 1, 4]
            patches = torch.cat([patches, new_patch], dim=1)

            # Stream output
            if stream:
                chunk = bytes(next_bytes)
                try:
                    print(chunk.decode('utf-8', errors='replace'), end='', flush=True)
                except Exception:
                    pass

            # Early stop: patch de puros ceros
            if all(b == 0 for b in next_bytes):
                break

        if stream:
            print()

        return patches_to_text(patches)

    def benchmark(
        self,
        prompt_patches: int = 64,
        gen_patches: int = 64,
    ) -> dict[str, float]:
        """
        Benchmark de throughput: mide latencia de prefill y generacion.

        Args:
            prompt_patches: patches del prompt (prefill)
            gen_patches:    patches a generar (decode)

        Returns:
            dict con prefill_ms, ms_per_patch, patches_per_sec, bytes_per_sec
        """
        # Datos aleatorios para benchmark (no buscamos coherencia, solo throughput)
        patches = torch.randint(0, 256, (1, prompt_patches, PATCH_SIZE),
                                dtype=torch.long, device=self.device)

        # Warmup: primer forward fuerza compilaciones JIT y asignaciones de CUDA
        context, pos = self._build_context(patches)
        _ = self.model(context, pos)

        # Medir prefill
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        context, pos = self._build_context(patches)
        _ = self.model(context, pos)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1000

        # Medir generacion
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        for _ in range(gen_patches):
            context, pos = self._build_context(patches)
            logits = self.model(context, pos)

            next_bytes = [logits[h][0, -1, :].argmax().item() for h in range(4)]
            new_patch = torch.tensor(
                [next_bytes], dtype=torch.long, device=self.device,
            ).unsqueeze(0)
            patches = torch.cat([patches, new_patch], dim=1)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        gen_ms = (time.perf_counter() - t0) * 1000

        ms_per_patch = gen_ms / gen_patches
        patches_per_sec = gen_patches / (gen_ms / 1000)
        bytes_per_sec = patches_per_sec * PATCH_SIZE

        return {
            'prefill_patches': prompt_patches,
            'prefill_ms': prefill_ms,
            'gen_patches': gen_patches,
            'gen_total_ms': gen_ms,
            'ms_per_patch': ms_per_patch,
            'patches_per_sec': patches_per_sec,
            'bytes_per_sec': bytes_per_sec,
        }


# =============================================================================
# 4. Carga de modelo
# =============================================================================

def load_model(
    checkpoint_path: str | None,
    device: torch.device,
    n_layers: int | None = None,
) -> CPU1Model:
    """
    Carga modelo desde checkpoint o crea uno con pesos random.

    Args:
        checkpoint_path: ruta al .pt (None = pesos random, util para testing)
        device:          dispositivo destino (cuda/cpu)
        n_layers:        override de capas (default: 20, usar 1-2 para testing rapido)

    Returns:
        CPU1Model en eval mode, movido al device especificado
    """
    kwargs = {}
    if n_layers is not None:
        kwargs['n_layers'] = n_layers

    model = CPU1Model(**kwargs)

    if checkpoint_path is not None:
        # map_location='cpu' evita OOM al cargar checkpoints grandes.
        # Despues de load_state_dict, se mueve al device con .to(device).
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        # Soporta tanto checkpoints del training (con 'model_state_dict')
        # como state_dicts crudos (solo pesos).
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state)

    model = model.to(device)
    model.eval()
    return model


# =============================================================================
# 5. Modo interactivo
# =============================================================================

def interactive_mode(generator: CPU1Generator, args: argparse.Namespace):
    """
    REPL interactivo para generar texto.

    Args:
        generator: instancia de CPU1Generator ya inicializada
        args:      namespace con temperature, top_k, top_p, max_patches
    """
    print("CPU-1 Inference (interactive)")
    print(f"  Device: {generator.device}")
    print(f"  Max context: {generator.max_context} patches")
    print(f"  Temperature: {args.temperature}, Top-k: {args.top_k}, Top-p: {args.top_p}")
    print("  Escribi un prompt y presiona Enter. 'quit' para salir.\n")

    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print("\nChau!")
            break

        if prompt.strip().lower() in ('quit', 'exit', 'q'):
            break

        if not prompt.strip():
            continue

        t0 = time.perf_counter()
        output = generator.generate(
            prompt,
            max_patches=args.max_patches,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stream=True,
        )
        elapsed = time.perf_counter() - t0

        n_gen_bytes = len(output.encode('utf-8')) - len(prompt.encode('utf-8'))
        print(f"  [{n_gen_bytes} bytes, {elapsed:.2f}s]")
        print()


# =============================================================================
# 6. Tests
# =============================================================================

def test_encoding():
    """Verifica encoding/decoding roundtrip."""
    texts = [
        "Hello world",
        "CPU-1 test!",
        "",
        "a",
        "ab",
        "abc",
        "abcd",
    ]
    for text in texts:
        patches = text_to_patches(text)
        decoded = patches_to_text(patches)
        assert decoded == text, f"Roundtrip failed: '{text}' -> '{decoded}'"

    print("[OK] Encoding roundtrip para ASCII")

    # UTF-8 multibyte
    utf8_text = "cafe"  # safe ASCII for Windows console
    patches = text_to_patches(utf8_text)
    decoded = patches_to_text(patches)
    assert decoded == utf8_text, f"UTF-8 roundtrip failed: '{utf8_text}' -> '{decoded}'"
    print("[OK] Encoding roundtrip para UTF-8")


def test_sampling():
    """Verifica sampling basico."""
    # Greedy: deberia devolver el argmax
    logits = torch.zeros(256)
    logits[42] = 10.0
    assert sample_byte(logits, temperature=0.0) == 42
    print("[OK] Greedy sampling")

    # Con temperature: deberia ser valido (0-255)
    logits = torch.randn(256)
    for _ in range(100):
        b = sample_byte(logits, temperature=0.8, top_k=10, top_p=0.9)
        assert 0 <= b <= 255

    print("[OK] Stochastic sampling (100 muestras)")


def test_context_building():
    """Verifica que sliding window mantiene position_ids absolutas."""
    model = CPU1Model(n_layers=1)
    gen = CPU1Generator(model, torch.device('cpu'), max_context=16)

    # Secuencia corta: sin truncamiento
    patches = torch.randint(0, 256, (1, 10, 4), dtype=torch.long)
    ctx, pos = gen._build_context(patches)
    assert ctx.shape == (1, 10, 4)
    assert pos.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    print("[OK] Context corto (sin truncar)")

    # Secuencia larga: con truncamiento
    patches = torch.randint(0, 256, (1, 100, 4), dtype=torch.long)
    ctx, pos = gen._build_context(patches)
    assert ctx.shape[1] == 16  # max_context
    # Sinks: pos 0-3, Recent: pos 88-99
    expected_pos = list(range(N_SINKS)) + list(range(100 - (16 - N_SINKS), 100))
    assert pos.tolist() == [expected_pos], f"Position IDs incorrectas: {pos.tolist()}"
    print("[OK] Context largo (truncado con sinks + absoluto)")


def test_generate():
    """Verifica generacion end-to-end (pesos random, output no tiene sentido)."""
    model = CPU1Model(n_layers=1)
    gen = CPU1Generator(model, torch.device('cpu'), max_context=32)

    output = gen.generate("Hi", max_patches=3, temperature=0.8)
    assert len(output) > 0
    print(f"[OK] Generate: 'Hi' -> {len(output)} chars")


def test_chunk_encoder():
    """Verifica ChunkEncoder."""
    model = CPU1Model(n_layers=5)
    model.eval()

    encoder = ChunkEncoder(model, n_encoder_layers=4, chunk_size=8)
    patches = torch.randint(0, 256, (1, 24, 4), dtype=torch.long)

    with torch.no_grad():
        embeddings = encoder.encode(patches)

    # 24 patches, chunk_size=8 -> 3 chunks
    assert embeddings.shape == (1, 3, D_MODEL), f"Shape: {embeddings.shape}"
    print(f"[OK] ChunkEncoder: {patches.shape} -> {embeddings.shape}")


def test_retrieval():
    """Verifica InContextRetrieval."""
    model = CPU1Model(n_layers=5)
    model.eval()

    encoder = ChunkEncoder(model, n_encoder_layers=4, chunk_size=8)
    retrieval = InContextRetrieval(top_k=2)

    # Agregar chunks
    patches = torch.randint(0, 256, (1, 16, 4), dtype=torch.long)
    with torch.no_grad():
        embs = encoder.encode(patches)
    retrieval.add(embs, patches, chunk_size=8)

    assert retrieval.cache_embeddings.shape[0] == 2  # 16/8 = 2 chunks

    # Retrieve
    query = torch.randn(D_MODEL)
    results = retrieval.retrieve(query)
    assert len(results) == 2
    assert all(r.shape[1] == 4 for r in results)  # cada chunk tiene patches de 4 bytes

    retrieval.clear()
    assert retrieval.cache_embeddings is None
    print("[OK] InContextRetrieval: add + retrieve + clear")


# =============================================================================
# 7. CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CPU-1 Inference Runner")

    p.add_argument('--checkpoint', type=str, default=None,
                   help="Ruta al checkpoint .pt")
    p.add_argument('--prompt', type=str, default=None,
                   help="Texto de entrada para generacion")
    p.add_argument('--interactive', action='store_true',
                   help="Modo REPL interactivo")
    p.add_argument('--benchmark', action='store_true',
                   help="Correr benchmark de throughput")

    # Generacion
    p.add_argument('--max-patches', type=int, default=128,
                   help="Patches a generar (x4 = bytes)")
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top-k', type=int, default=50)
    p.add_argument('--top-p', type=float, default=0.9)
    p.add_argument('--stream', action='store_true',
                   help="Imprimir bytes a medida que se generan")

    # Modelo
    p.add_argument('--n-layers', type=int, default=None,
                   help="Override de capas (default: 20)")
    p.add_argument('--device', type=str, default=None,
                   help="Device (auto-detect si no se especifica)")

    # Benchmark
    p.add_argument('--bench-prompt-patches', type=int, default=64)
    p.add_argument('--bench-gen-patches', type=int, default=64)

    return p.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Modelo
    model = load_model(args.checkpoint, device, n_layers=args.n_layers)
    n_params = sum(p.numel() for p in model.parameters())

    generator = CPU1Generator(model, device)

    if args.benchmark:
        print(f"Benchmark CPU-1 ({n_params:,} params) en {device}")
        results = generator.benchmark(
            prompt_patches=args.bench_prompt_patches,
            gen_patches=args.bench_gen_patches,
        )
        print(f"  Prefill: {results['prefill_patches']} patches -> {results['prefill_ms']:.1f} ms")
        print(f"  Gen: {results['gen_patches']} patches -> {results['gen_total_ms']:.1f} ms")
        print(f"  Latencia: {results['ms_per_patch']:.1f} ms/patch")
        print(f"  Throughput: {results['patches_per_sec']:.1f} patches/s = {results['bytes_per_sec']:.0f} bytes/s")
        return

    if args.interactive:
        interactive_mode(generator, args)
        return

    if args.prompt:
        output = generator.generate(
            args.prompt,
            max_patches=args.max_patches,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stream=args.stream,
        )
        if not args.stream:
            print(output)
        return

    # Sin argumentos: correr tests
    print("Sin argumentos -> corriendo tests\n")
    print("=" * 60)
    test_encoding()
    print()
    test_sampling()
    print()
    test_context_building()
    print()
    test_generate()
    print()
    test_chunk_encoder()
    print()
    test_retrieval()
    print("\n" + "=" * 60)
    print("Todos los tests pasaron.")


if __name__ == "__main__":
    main()
