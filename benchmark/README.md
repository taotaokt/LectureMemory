# Retrieval benchmark data

`queries.jsonl` is the first real-material retrieval benchmark for Lecture Memory. It contains
30 student-style queries labeled with a stable course name, lecture name, source filename, and
one or more relevant PDF page numbers.

The seed set intentionally mixes:

- Chinese and English queries;
- semantic paraphrases that do not simply copy slide titles;
- visual descriptions of graphs, tables, trees, and handwritten layouts;
- queries derived from personal notes;
- visually or semantically similar pages that are easy to confuse.

The referenced course PDFs are private classroom materials and are not included in this
repository. Place authorized local copies under `data/raw/CS344/` and `data/raw/CS440/` before
using this dataset. The `source_file` and `page_number` fields provide stable labels before local
database lecture IDs have been assigned.

This 30-query seed is suitable for early integration testing. Expand it to at least 50 queries
before reporting retrieval quality as a project benchmark.
