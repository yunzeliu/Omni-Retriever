# Evaluation

All evaluation flows are two-stage:

1. **Extract embeddings** with `scripts/extract_embeddings.sh`.
2. **Score** them with `scripts/evaluate.sh`.

Output: Recall@1/5/10, NDCG@10, MRR, median rank per direction
(diagonal-relevance, one positive per query).

## OmniRetriever-Bench

```bash
huggingface-cli download YunzeLiu/OmniRetriever-Bench --repo-type dataset \
    --local-dir benchmark

bash scripts/extract_embeddings.sh \
    benchmark/manifests/omniretriever_bench.jsonl \
    output/embeddings/omniretriever_bench.npz

bash scripts/evaluate.sh output/embeddings/omniretriever_bench.npz
```

Expected aggregate row (R@1):

| AVG-single | AVG-dual | **AVG-all** |
| ---: | ---: | ---: |
| 28.63 | 41.05 | **34.84** |

For the 12-direction breakdown, score one direction at a time by slicing
the canonical `.npz` (see [`benchmark.md`](benchmark.md)).

## External benchmarks (paper Tables 2–3)

Same pipeline; substitute the manifest. Expected R@1 (T→\*):

| Benchmark | Direction | R@1 |
| --- | --- | ---: |
| Clotho | T→A | 19.14 |
| SoundDescs | T→A | 25.00 |
| MSR-VTT | T→V | 47.9 |
| MSVD | T→V | 65.6 |
| DiDeMo | T→V | 45.1 |
| VATEX | T→V | 58.7 |

## Programmatic API

```python
from omniretriever.evaluation import score_benchmark

results = score_benchmark("output/embeddings/omniretriever_bench.npz")
print(results["t2m"]["R@1"], results["m2t"]["R@1"])
```
