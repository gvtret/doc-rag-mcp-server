# Benchmark results

Reproducible benchmark numbers for the `doc-rag` pipeline, produced by
`scripts/bench.py`. The benchmark is intentionally self-contained: a
synthetic Lorem-equivalent corpus is generated in-process, then the
script measures embedding encode throughput, FAISS index build time,
and FAISS reconstruct-on-delete time.

## Method

```bash
python scripts/bench.py --size <N> --device <cpu|cuda> \
    --model BAAI/bge-large-en-v1.5 --batch-size <B>
```

- **Encode** measures `SentenceTransformer.encode()` throughput on N
  synthetic chunks (~80 words each), with L2 normalisation enabled
  (the production setting in `config.yaml`).
- **Index build** measures `faiss.IndexFlatIP(dim).add(vectors)`.
- **Reconstruct** measures the prune-and-rebuild path used by
  `delete_documents` — 10 % of `doc_id`s are marked for deletion, the
  remaining vectors are retrieved via `index.reconstruct(i)` and added
  to a fresh `IndexFlatIP`.

Each row below is from a single run on a quiet machine; numbers are
indicative, not benchmarks-grade reproducible.

## Reference numbers

### Production model — `BAAI/bge-large-en-v1.5` (1024 dim)

| Size | Device | Hardware | Batch | Encode | Throughput | Reconstruct |
|---:|---|---|---:|---:|---:|---:|
| 1 000  | CUDA | GTX 1650, 4 GB VRAM, WSL2 | 16 | 38.3 s  | 26 ch/s   | 22 ms |
| 10 000 | CUDA | GTX 1650, 4 GB VRAM, WSL2 | 16 | 412.5 s | 24 ch/s   | 79 ms |
| 1 000  | CPU  | i7-class, 8 cores         |  8 | 287.5 s | 3.5 ch/s  | 2 ms  |
| 1 000  | CPU  | QEMU x86_64, 8 vCPU       |  8 | 481.2 s | 2.1 ch/s  | 7 ms  |

GPU/CPU speedup on identical hardware (large model): roughly **7.5×**.
The advantage grows over `bge-small` because the encoder becomes more
expensive per chunk and CPU saturates earlier. GPU throughput is also
**flat across corpus size** (26 ch/s at 1 K vs. 24 ch/s at 10 K),
which means the bottleneck is per-chunk model evaluation rather than
batch overhead — useful to know when sizing future hardware.

A virtualised CPU runs ~1.7× slower than the same number of bare-metal
cores: the QEMU server hits 2.1 ch/s vs. 3.5 ch/s on bare i7. That
matches the typical hypervisor overhead for compute-bound workloads.
For server-side capacity planning, **a full rebuild of ~4 500 chunks
on the QEMU server projects to roughly 36 minutes** — well within an
off-hours `doc-rag ingest` window, see `docs/deploy.md` § "Scheduled
ingest".

### Smoke checks — `BAAI/bge-small-en-v1.5` (384 dim)

| Size | Device | Hardware | Batch | Encode | Throughput |
|---:|---|---|---:|---:|---:|
| 200 | CPU  | i7-class, 8 cores         | 32 | 5.0 s | 40 ch/s |
| 200 | CUDA | GTX 1650, 4 GB VRAM, WSL2 | 32 | 1.8 s | 113 ch/s |

The `bge-small` row is included only to show the GPU/CPU ratio on
identical hardware: with the small model the speedup is ~3×; with the
large model it grows because CPU is the bottleneck.

## Index size scaling

`IndexFlatIP` is an exact (not approximate) index — search cost is
linear in the index size. For the project's typical corpora
(thousands to a few tens of thousands of chunks) this is fine; we
stop being competitive vs. approximate-NN indexes (`IndexHNSWFlat`,
`IndexIVFFlat`) at roughly 10⁵+ vectors.

| Index size | Memory (1024 dim, float32) | Approx. search latency |
|---:|---:|---:|
| 1 000 chunks  | ~4 MB   | < 1 ms |
| 10 000 chunks | ~40 MB  | ~5 ms |
| 50 000 chunks | ~200 MB | ~25 ms |
| 100 000 chunks | ~400 MB | ~50 ms |

Anything above 50 000 chunks is a good moment to switch to an
approximate index — out of scope for v1.x.

## Reproducing

The default benchmark target is GPU with the large model. To extend
the table or refresh numbers, run:

```bash
mkdir -p build/bench
python scripts/bench.py --size 1000 --device cuda \
    --model BAAI/bge-large-en-v1.5 --batch-size 16 --save
```

The JSON saved under `build/bench/<size>-<device>-<ts>.json` follows
`schema_version: 1` — the same shape across releases.
