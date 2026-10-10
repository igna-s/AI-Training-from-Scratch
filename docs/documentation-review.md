# Documentation evidence review — 2026-10-10

The current documentation describes the checked implementations and recorded results with explicit limits. This review corrected stale references and unsupported benchmark recommendations. It does **not** certify every training job, intermediate checkpoint, dataset row, or hardware path.

The scope is all **14 pre-existing documentation and dependency text files** in the GitHub `main` tree: ten Markdown documents, the corpus attribution, and three requirements files. The authenticated GitHub tree matched the local 131-file baseline at `22c386ce541e12718ab656a3f879775777b7b22e`; `main` was the only listed branch and there were no GitHub releases. Historical commits and the retained experiment tag were checked where current documents cite them. They remain dated source/evidence snapshots, rather than current usage instructions. The wiki endpoint returned 404; no wiki content was available to review.

The [machine-readable record](../experiments/evaluation/documentation-review-2026-10-10.json) contains the document hashes, verification counts, metadata response, CPU example results, and limitations. Its document hashes normalize line endings to LF. This review is an additional document beyond the original fourteen.

## Documents reviewed

| Document | Finding and evidence |
|---|---|
| [Main README](../README.md) | All 34 quality rows, 34 timing rows, and three generated excerpts match the original records. Architecture totals and failure counts agree with the inventories. The AMD-credit statement is author-reported project context, not proof of a specific checkpoint's job. |
| [Architecture review](architecture-review.md) | The corrected component descriptions agree with source and the fresh 133 constructor inventories. Synthetic probes reproduce the scan defect, gate lengths, unused larger-model boundary module, nonmonotonic decay initialization, and real-data loader error. |
| [FP32 model card](model_cards/cpu1-ablations-final.md) | Quality, timing, correctness, architecture, vocabulary-mapping, and error tables match all 34 records. The loading example runs with the identical cached pinned `run_04` export. Source access and GPU-validation limits are now explicit. |
| [Checkpoint archive card](model_cards/cpu1-ablation-checkpoints.md) | All 34 folder rows match the pinned inventory; shared export results match the FP32 audit. The archive's contents have not all been loaded. Packed-example syntax and its saved `run_04` loading evidence were checked; no archive binary was downloaded for this review. |
| [Dataset card](model_cards/cpu1-ablation-dataset.md) | File counts, sizes, missing shard IDs, schemas, and sample defects agree with the records. Service-reported row totals were checked on October 10 and are identified as service totals. The three cached shard inspections and pinned tokenizer probe reproduce the stored findings. |
| [Provenance review](provenance/README.md) | All 148 restored-file records match the retained source blobs, sizes, hashes, and commits. All 88 mapped commits exist under the retained tag. The 40 indexed source versions agree with their blobs; six historical syntax failures are reproducible. Recovery and later review dates are distinguished. |
| [Usage guide](usage.md) | Both CPU Python examples run unchanged with an identical cached pinned `run_02` export replacing only the download. Training recipe preview and arguments were checked without launching a GPU job. The private source, D: evaluator requirement, cache behavior, and dataset limitations are explicit. |
| [Published experiment guide](../experiments/README.md) | All 34 manifest/configuration records match the saved steps, model hashes, configuration hashes, and parameter counts. Both explicit-config training commands parse. They describe real training, and a successful training/resume job remains unverified. |
| [Evaluation evidence guide](../experiments/evaluation/README.md) | All 42 frozen evidence checksums match. The retained original evaluator's Git blob matches its recorded SHA-256. Historical measurements and later checks are clearly separated. |
| [Native kernel guide](../inference/links.md) | Corrected the MLGRU paper reference and removed unmeasured optimal-mode/model-size claims. Source shows random tensors and fixed-formula byte feedback. Build recipes are identified as unvalidated in this environment. |
| [Corpus attribution](../experiments/evaluation/model-audit-2026-10-06/CORPUS_LICENSE.txt) | Describes the saved WikiText selection/truncation and upstream CC-BY-SA-3.0/GFDL metadata. Model licensing does not relicense the excerpts. |
| [Audit requirements](../requirements/audit.txt) | Referenced requirements file exists; declared package constraints are satisfied by the installed review environment. PyTorch is installed separately as documented. |
| [Data requirements](../requirements/data.txt) | Referenced requirements file exists; installed dependencies satisfy the declared constraints. Optional teacher quantization dependencies are not represented as mandatory. |
| [Training requirements](../requirements/training.txt) | Installed dependencies satisfy the declared constraints. These lower bounds are not a complete lockfile or a guarantee for every permitted future version. |

## Corrections made

1. **Native benchmark scope.** `inference/benchmark.c` creates random weights and inputs. Its local loop computes logits but feeds back `(byte_position * 17) % 256`. That is synthetic kernel work, not sampled text from a trained checkpoint. The guide no longer labels execution modes universally optimal or assigns unmeasured 170M/400M whole-model sizes.
2. **Paper attribution.** [Scalable MatMul-free Language Modeling](https://arxiv.org/abs/2406.02528) is the MLGRU reference. [Were RNNs All We Needed?](https://arxiv.org/abs/2410.01201) concerns minLSTM/minGRU and is listed separately. Neither reference establishes reproduction of that paper's results here.
3. **Canonical source links.** Current cards, their generator, publication helpers, and restored-file URLs use `igna-s/AI-Training-from-Scratch`. Historical tag and commit pins were preserved. The obsolete README anchor in the native guide was replaced.
4. **Private source and GPU claims.** Cards state that source access requires GitHub authorization. The AMD/ROCm entry point exists, but CPU loading and architecture checks do not validate a fresh GPU training job.
5. **Download accounting and dates.** The three inspection Parquets total approximately 158.5 MB; an uncached `--tokenizer` also downloads tokenizer/configuration files. The fourteen tests belong to the October 8 recovery; later reviews ran eighteen. Dataset row totals are attributed to the service and check date.

The [architecture document](architecture-review.md) also retains the preceding corrections: SwiGLU belongs to the enclosing Transformer block, FPResidual includes SiLU, learned decay does not enforce a monotonic hierarchy, and its original source hashes include Windows CRLF bytes.

## What the evidence establishes

The [original CPU results](../experiments/evaluation/model-audit-2026-10-06/) and [later checkpoint recheck](../experiments/evaluation/readme-validation-2026-10-10.json) support these outcomes:

| Check | Recorded outcome |
|---|---|
| FP32 strict loading | 34/34 |
| Finite loss | 30/34; four nonfinite |
| Prefix causality | 16 PASS, 14 FAIL, 4 ERROR |
| Recurrent chunk equivalence | 26 FAIL; eight Transformer N/A |
| Export inventory | 34 files, approximately 4.178 GB |
| Archive metadata | 1,346 checkpoint files, 34 folders, 21 final files, 215,178,202,122 bytes |
| Dataset metadata | 917 Parquet files, 68,018,868,047 compressed bytes |
| Dataset-service rows | 24,101 byte rows + 21,928 BPE rows = 46,029 |

The architecture inventory covers all 34 published configurations in both constructors, all 30 current presets in both entry points, and five larger-model variants: **68 + 60 + 5 = 133 constructions**. Parameter and module inventories establish structure, not trained accuracy or reproduction of a paper.

This documentation review additionally passed **18 existing offline regression tests**, reproduced the two usage examples and FP32 card example, checked three platform previews and both training-command parsers, verified all five Python examples' syntax, and checked internal file links and relative heading anchors. The two model cards exactly match their updated generator; rendering writes were intercepted so the frozen evidence was not rewritten.

The `run_02` usage example reproduces mean NLL **1.4600811004638672** and the saved 128-byte continuation. That NLL concerns the usage example's single short string; it is distinct from `run_02`'s **1.9702** NLL on the 32-passage evaluation sample. The FP32 `run_04` shape example returns four finite `[1, 5, 256]` heads; shape success does not establish useful text generation.

Cached files were used for these execution checks. No new model weights or training shards were downloaded. The fresh external dataset check retrieved metadata. This revision updates GitHub documentation and source-link generation; the accompanying Hugging Face cards retain their previously published versions.

## What remains unverified

- Full fresh CUDA/ROCm training, production readiness, and end-to-end checkpoint resume.
- Every intermediate archive binary and all 917 dataset shards. Inventory metadata does not validate their complete contents.
- Dataset-wide teacher correctness, exact exposure, document separation, and deterministic reconstruction. The three-shard review already shows reader, mask, and UTF-8 defects.
- A rerun of every original timing measurement and continuation. The full tables preserve the October 6 protocol; the later recheck covered checkpoint identity, loading, losses, causality, and chunks.
- Native compilation, trained-checkpoint loading, native/PyTorch numerical parity, and useful native text-generation throughput.
- Exact source/environment/tokenizer/teacher/extraction attribution for every historical training job. Configured budgets and run names do not establish completed exposure or convergence.
- Every permitted dependency version. Requirements are broad lower bounds; the checked environment is one concrete installation.

The original records and historical tag remain unchanged. A low loss from a model that fails causality is not evidence of valid autoregressive prediction quality. The project remains an experimental code and artifact collection with documented failures.
