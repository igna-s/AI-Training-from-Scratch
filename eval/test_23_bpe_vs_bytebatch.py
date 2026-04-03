"""
Test 23 — BPE vs ByteBatch Throughput

Compara velocidad de tokenizacion:
  - ByteBatch4: el esquema del proyecto (4 bytes → 1 token)
  - BPE Python puro: implementacion naive
  - Tiktoken (opcional): BPE en Rust, referencia de estado del arte

Metrica: MB/s de throughput por metodo
Criterio: ByteBatch4 debe ser >= 100x más rápido que BPE Python puro
"""
import re
import struct
import time
from collections import defaultdict


def _byte_batch_4(data: bytes) -> list:
    tokens = []
    for i in range(0, len(data) - 3, 4):
        tokens.append(struct.unpack_from('>I', data, i)[0])
    rem = len(data) % 4
    if rem:
        tokens.append(int.from_bytes(data[-rem:], 'big'))
    return tokens


def _build_bpe_vocab(text: str, num_merges: int) -> list:
    vocab = defaultdict(int)
    for word in text.split():
        vocab[" ".join(list(word)) + " </w>"] += 1

    def get_pairs(v):
        pairs = defaultdict(int)
        for word, freq in v.items():
            syms = word.split()
            for i in range(len(syms) - 1):
                pairs[(syms[i], syms[i + 1])] += freq
        return pairs

    merges = []
    for _ in range(num_merges):
        pairs = get_pairs(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        bigram = re.escape(" ".join(best))
        vocab = {re.sub(bigram, "".join(best), w): f for w, f in vocab.items()}
    return merges


def _apply_bpe(text: str, merges: list) -> list:
    tokens = []
    for word in text.split():
        word_tok = list(word) + ["</w>"]
        for a, b in merges:
            i, new_tok = 0, []
            while i < len(word_tok):
                if i < len(word_tok) - 1 and word_tok[i] == a and word_tok[i + 1] == b:
                    new_tok.append(a + b); i += 2
                else:
                    new_tok.append(word_tok[i]); i += 1
            word_tok = new_tok
        tokens.extend(word_tok)
    return tokens


def test_bpe_vs_bytebatch(n_words: int = 50_000, num_merges: int = 500,
                           runs_bb: int = 50, runs_bpe: int = 5):
    import random
    random.seed(42)
    words = ["medical", "legal", "patient", "diagnosis", "contract", "court",
             "treatment", "liability", "prescription", "evidence", "doctor", "law"]
    text = " ".join(random.choices(words, k=n_words))
    text_bytes = text.encode("utf-8")
    n_bytes = len(text_bytes)

    # --- ByteBatch4 ---
    _byte_batch_4(text_bytes)  # warmup
    t0 = time.perf_counter()
    for _ in range(runs_bb):
        toks_bb = _byte_batch_4(text_bytes)
    bb_throughput = n_bytes / ((time.perf_counter() - t0) / runs_bb) / 1e6

    # --- BPE Python ---
    merges = _build_bpe_vocab(text[:5000], num_merges)
    _apply_bpe(text[:1000], merges)  # warmup
    t0 = time.perf_counter()
    for _ in range(runs_bpe):
        toks_bpe = _apply_bpe(text, merges)
    bpe_throughput = n_bytes / ((time.perf_counter() - t0) / runs_bpe) / 1e6

    ratio = bb_throughput / max(bpe_throughput, 1e-9)

    # --- Tiktoken (opcional) ---
    tt_throughput = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        enc.encode(text[:1000])  # warmup
        t0 = time.perf_counter()
        for _ in range(runs_bb):
            enc.encode(text)
        tt_throughput = n_bytes / ((time.perf_counter() - t0) / runs_bb) / 1e6
    except ImportError:
        pass

    tt_str = f" | tiktoken={tt_throughput:.1f} MB/s" if tt_throughput else ""
    print(f"[Test 23] ByteBatch4={bb_throughput:.1f} MB/s | BPE={bpe_throughput:.3f} MB/s | ratio={ratio:.0f}x{tt_str}")
    assert ratio >= 100, f"ByteBatch4 deberia ser >=100x mas rapido que BPE, got {ratio:.1f}x"

    result = {
        'bytebatch_mb_s': round(bb_throughput, 1),
        'bpe_mb_s': round(bpe_throughput, 3),
        'ratio_bpe_bb': round(ratio, 1),
    }
    if tt_throughput:
        result['tiktoken_mb_s'] = round(tt_throughput, 1)
    return result


if __name__ == "__main__":
    test_bpe_vs_bytebatch()
