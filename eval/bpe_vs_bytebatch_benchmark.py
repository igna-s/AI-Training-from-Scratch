import time, random, re, struct
from collections import defaultdict

# ── Texto de prueba ──────────────────────────────────────────────────────────
random.seed(42)
words = ["medical", "legal", "patient", "diagnosis", "contract", "court",
         "treatment", "liability", "prescription", "evidence", "doctor", "law"]
TEXT = " ".join(random.choices(words, k=50_000))
TEXT_BYTES = TEXT.encode("utf-8")
N = len(TEXT_BYTES)
print(f"Texto: {N:,} bytes\n")

# ════════════════════════════════════════════════════════════════════════════
# 1. BYTE BATCH DE 4
# ════════════════════════════════════════════════════════════════════════════

def byte_batch_4(data: bytes):
    tokens = []
    for i in range(0, len(data) - 3, 4):
        tokens.append(struct.unpack_from('>I', data, i)[0])
    rem = len(data) % 4
    if rem:
        tokens.append(int.from_bytes(data[-rem:], 'big'))
    return tokens

_ = byte_batch_4(TEXT_BYTES)  # warmup

RUNS = 50
t0 = time.perf_counter()
for _ in range(RUNS):
    toks_bb = byte_batch_4(TEXT_BYTES)
t1 = time.perf_counter()
bb_ms = (t1 - t0) / RUNS * 1000
bb_throughput = N / ((t1 - t0) / RUNS) / 1e6

print(f"[BYTE BATCH 4]")
print(f"  Tokens producidos : {len(toks_bb):,}")
print(f"  Tiempo promedio   : {bb_ms:.3f} ms")
print(f"  Throughput        : {bb_throughput:.1f} MB/s\n")

# ════════════════════════════════════════════════════════════════════════════
# 2. BPE PYTHON PURO (~500 merges)
# ════════════════════════════════════════════════════════════════════════════

def build_bpe_vocab(text: str, num_merges: int):
    vocab = defaultdict(int)
    for word in text.split():
        vocab[" ".join(list(word)) + " </w>"] += 1

    def get_pairs(vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            syms = word.split()
            for i in range(len(syms) - 1):
                pairs[(syms[i], syms[i+1])] += freq
        return pairs

    merges = []
    for _ in range(num_merges):
        pairs = get_pairs(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        bigram = re.escape(" ".join(best))
        new_vocab = {}
        for word, freq in vocab.items():
            new_vocab[re.sub(bigram, "".join(best), word)] = freq
        vocab = new_vocab
    return merges

def apply_bpe(text: str, merges: list):
    tokens = []
    for word in text.split():
        word_tok = list(word) + ["</w>"]
        for a, b in merges:
            i, new_tok = 0, []
            while i < len(word_tok):
                if i < len(word_tok) - 1 and word_tok[i] == a and word_tok[i+1] == b:
                    new_tok.append(a + b); i += 2
                else:
                    new_tok.append(word_tok[i]); i += 1
            word_tok = new_tok
        tokens.extend(word_tok)
    return tokens

NUM_MERGES = 500
print(f"Construyendo vocab BPE ({NUM_MERGES} merges)...")
t_build = time.perf_counter()
merges = build_bpe_vocab(TEXT[:5000], NUM_MERGES)
print(f"  Vocab build time  : {(time.perf_counter() - t_build)*1000:.1f} ms")

_ = apply_bpe(TEXT[:1000], merges)  # warmup

RUNS_BPE = 5
t0 = time.perf_counter()
for _ in range(RUNS_BPE):
    toks_bpe = apply_bpe(TEXT, merges)
t1 = time.perf_counter()
bpe_ms = (t1 - t0) / RUNS_BPE * 1000
bpe_throughput = N / ((t1 - t0) / RUNS_BPE) / 1e6

print(f"\n[BPE Python puro ({NUM_MERGES} merges)]")
print(f"  Tokens producidos : {len(toks_bpe):,}")
print(f"  Tiempo promedio   : {bpe_ms:.1f} ms")
print(f"  Throughput        : {bpe_throughput:.3f} MB/s\n")

# ════════════════════════════════════════════════════════════════════════════
# 3. TIKTOKEN (BPE en Rust — requiere: pip install tiktoken)
# ════════════════════════════════════════════════════════════════════════════

try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")  # mismo encoding que GPT-4

    _ = enc.encode(TEXT[:1000])  # warmup

    RUNS_TT = 50
    t0 = time.perf_counter()
    for _ in range(RUNS_TT):
        toks_tt = enc.encode(TEXT)
    t1 = time.perf_counter()
    tt_ms = (t1 - t0) / RUNS_TT * 1000
    tt_throughput = N / ((t1 - t0) / RUNS_TT) / 1e6

    print(f"[TIKTOKEN — cl100k_base (Rust)]")
    print(f"  Tokens producidos : {len(toks_tt):,}")
    print(f"  Tiempo promedio   : {tt_ms:.3f} ms")
    print(f"  Throughput        : {tt_throughput:.1f} MB/s\n")

    tiktoken_available = True

except ImportError:
    print("[TIKTOKEN] No instalado — corré:  pip install tiktoken\n")
    tiktoken_available = False
    tt_ms = None

# ════════════════════════════════════════════════════════════════════════════
# RESUMEN
# ════════════════════════════════════════════════════════════════════════════
print("=" * 55)
print(f"{'Método':<28} {'Tiempo':>10} {'Throughput':>12}")
print("-" * 55)
print(f"{'Byte Batch 4':<28} {bb_ms:>9.2f}ms {bb_throughput:>10.1f} MB/s")
if tiktoken_available:
    print(f"{'Tiktoken (Rust BPE)':<28} {tt_ms:>9.2f}ms {tt_throughput:>10.1f} MB/s")
print(f"{'BPE Python puro':<28} {bpe_ms:>9.1f}ms {bpe_throughput:>10.3f} MB/s")
print("=" * 55)
print(f"\nRATIO  BPE Python / ByteBatch4  : {bpe_ms/bb_ms:.0f}x más lento")
if tiktoken_available:
    print(f"RATIO  BPE Python / Tiktoken    : {bpe_ms/tt_ms:.0f}x más lento")
    print(f"RATIO  Tiktoken  / ByteBatch4   : {tt_ms/bb_ms:.1f}x más lento")
