# Qwen3-VL-Embedding-2B Feasibility Validation

Date: 2026-09-18

## Decision

Qwen3-VL-Embedding-2B is feasible as the first production embedding backend for Lecture Memory on an Apple Silicon laptop with 16 GB of unified memory.

The spike successfully embedded slide images and retrieved the correct image from natural-language queries in both English and Chinese. Task 3.2 can proceed, with explicit MPS support added by the Lecture Memory adapter.

## Environment

- Apple M2 Pro with 16 GB unified memory
- Python 3.12.14 in an isolated temporary environment
- PyTorch 2.8.0
- Transformers 4.57.6
- Qwen3-VL-Embedding-2B
- MPS inference using FP16 and SDPA attention
- Maximum input image budget: 524,288 pixels

The downloaded model occupied approximately 4.0 GiB. The temporary Python environment occupied approximately 854 MiB. Neither artifact is tracked by Git.

## Method

The synthetic corpus contained six 1280 x 720 technical lecture slides:

1. Divide and Conquer
2. Karatsuba Multiplication
3. Convex Sets
4. Principal Component Analysis
5. Breadth-First Search
6. Support Vector Machines

Eight text queries were embedded using the retrieval instruction:

```text
Retrieve the lecture slide image most relevant to the user's query.
```

The test used four English queries and four Chinese queries. Query and slide embeddings were normalized, and slides were ranked by dot product (equivalent to cosine similarity for normalized vectors).

## Results

| Metric | Result |
| --- | ---: |
| Model load time | 13.49 s |
| Embedding dimension | 2,048 |
| MPS model allocation after load | 3.96 GiB |
| MPS driver allocation after load | 4.63 GiB |
| First slide, including MPS warm-up | 11.59 s |
| Subsequent slide latency | 1.54-1.64 s per slide |
| Eight-query batch latency | 3.58 s |
| English Top-1 accuracy | 4/4 |
| Chinese Top-1 accuracy | 4/4 |
| Overall Top-1 accuracy | 8/8 |
| Overall Top-3 accuracy | 8/8 |

Representative rankings:

| Query | Expected and retrieved slide | Top-1 score | Runner-up score |
| --- | --- | ---: | ---: |
| `three recursive multiplication calls` | Karatsuba Multiplication | 0.60636 | 0.44839 |
| `dimensionality reduction using eigenvectors` | Principal Component Analysis | 0.63943 | 0.33423 |
| `shortest path in an unweighted graph using a queue` | Breadth-First Search | 0.65342 | 0.23980 |
| `maximum-margin separating hyperplane` | Support Vector Machines | 0.69912 | 0.40823 |
| `哪一页讲了三个递归乘法调用？` | Karatsuba Multiplication | 0.56332 | 0.46945 |
| `哪一页介绍了凸集合中线段仍位于集合内部？` | Convex Sets | 0.68455 | 0.35517 |
| `用特征向量进行降维` | Principal Component Analysis | 0.56964 | 0.29737 |
| `无权图中使用队列寻找最短路径` | Breadth-First Search | 0.56226 | 0.21577 |

## Implementation Findings

1. The upstream helper automatically selects CUDA or CPU, but does not select Apple MPS. The Lecture Memory adapter must implement `CUDA -> MPS -> CPU` device selection rather than relying on the upstream default.
2. The model should load lazily and remain resident for the lifetime of an indexing or search process. Reloading it per request would add roughly 13 seconds of avoidable latency.
3. A warm-up inference should run after loading because the first MPS image inference was substantially slower than steady-state inference.
4. Slide embeddings should be generated in bounded batches. Image resolution and batch size must remain configurable to control unified-memory pressure.
5. Model dependencies should be an optional installation extra so the base application and unit tests do not require a multi-gigabyte ML environment.

## Limitations

This was a small technical spike, not a retrieval-quality benchmark. The slides were synthetic, visually simple, and contained clear English text. The perfect Top-1 result must not be interpreted as expected accuracy on real lecture decks.

Before declaring the embedding layer production-ready, it must be tested on real exported lecture PDFs containing diagrams, equations, dense layouts, low-resolution scans, and visually similar pages. The later benchmark phase will measure Recall@K, MRR, and latency on a larger labeled dataset.

## Next Step

Proceed with Task 3.2: implement the production `Qwen3VLEmbeddingProvider` behind the existing model-independent embedding interface, including lazy loading, device selection, normalized outputs, batch inference, configuration, and actionable errors.

## References

- [Qwen3-VL-Embedding repository](https://github.com/QwenLM/Qwen3-VL-Embedding)
- [Qwen3-VL-Embedding-2B model card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)

