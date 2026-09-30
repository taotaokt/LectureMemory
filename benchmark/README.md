# Retrieval benchmark data

`queries.jsonl` is the first real-material retrieval benchmark for Lecture Memory. It contains
50 student-style queries labeled with a stable course code, lecture name, source filename, and
one or more relevant PDF page numbers.

The dataset intentionally mixes:

- Chinese and English queries;
- semantic paraphrases that do not simply copy slide titles;
- visual descriptions of graphs, tables, trees, and handwritten layouts;
- queries derived from personal notes;
- visually or semantically similar pages that are easy to confuse.

The referenced course PDFs are private classroom materials and are not included in the Git
repository. Place authorized local copies under `data/raw/CS344/` and `data/raw/CS440/` before
validating or evaluating the dataset. The `course`, `lecture`, `source_file`, and `page_number`
fields are stable across machines; an optional `lecture_id` can pin a page label to a particular
ingested database.

Validate the schema and all local PDF page references with:

```bash
python -m scripts.validate_benchmark
```

To additionally verify that every label resolves to an ingested `SlidePage`, provide the local
SQLite database:

```bash
python -m scripts.validate_benchmark --database data/database/lecture_memory.db
```

Once those lectures exist in the configured database, run the dependency-free bilingual BM25
baseline with:

```bash
python -m scripts.evaluate --methods bm25
```

For the complete comparison, first build the FAISS index and then evaluate all three methods:

```bash
python -m scripts.build_index
python -m scripts.evaluate
```

The full run compares BM25, Qwen multimodal embeddings, and Qwen embeddings plus reranking. It
prints Recall@1, Recall@5, Recall@10, MRR, and average latency, then writes detailed JSON and a
summary CSV under `benchmark/results/`. Use `--help` to override paths, methods, retrieval depth,
or the output run ID. BM25 indexes extracted slide text plus notes attached to each page; the
multimodal methods map both slide and attached-note results back to the same page labels.

Each JSONL record uses this shape:

```json
{
  "id": "cs344-l1-q01",
  "query": "What input makes sequential search perform the maximum comparisons?",
  "course": "CS344",
  "lecture": "L1 - Algorithmic Analysis",
  "source_file": "L1 - Algorithmic Analysis.pdf",
  "relevant_pages": [{"page_number": 6}],
  "language": "en",
  "query_type": "semantic",
  "difficulty": "paraphrase"
}
```

Allowed metadata values are deliberately small and validated:

- `language`: `en`, `zh`
- `query_type`: `semantic`, `visual`, `note`
- `difficulty`: `standard`, `paraphrase`, `confusable`, `hard`

Do not add a label based only on a slide title. Review the source page, phrase the query as a
student would remember it, and validate the page reference before committing it.
