# Lecture Memory

## Codex Project Specification & Implementation Plan

Version: 0.1
Project type: AI / Multimodal Retrieval / Study Memory
Primary goal: Build a small but complete AI study-memory system that allows students to search lecture slides and personal notes semantically.

---

# 1. Project Goal

Build a local application called **Lecture Memory**.

The application should help a student answer questions such as:

```text
Where did the professor explain convex sets?

Find the slide showing different convex and non-convex shapes.

When did we discuss three-subproblem integer multiplication?

Find my note about why Karatsuba is faster.

Where have I seen PCA before?
```

The system should search across:

```text
lecture PDFs
individual slide pages
personal notes
lecture metadata
extracted concepts
```

The primary use case is:

```text
I remember learning or seeing something,
but I do not remember exactly where.
```

This is NOT initially a chatbot.

The core product is:

```text
course memory
+
multimodal retrieval
+
personal notes
+
evaluation
```

---

# 2. Product Positioning

Do NOT build this as:

```text
generic PDF chatbot
generic RAG system
AI note summarizer
ChatGPT wrapper
general file search
```

Build it as:

> A persistent AI memory system for university lecture materials.

The central abstraction is not a document.

The central abstraction is:

```text
Course
    ↓
Lecture
    ↓
Lecture Memory
```

Each lecture should eventually contain:

```text
slides
notes
metadata
concepts
page embeddings
searchable content
```

---

# 3. Target Version

The first complete public GitHub version should be:

## V1.5 — Lecture Memory + Retrieval + Evaluation

Required capabilities:

| Capability            | Required |
| --------------------- | -------- |
| Create course         | Yes      |
| Create lecture        | Yes      |
| Upload PDF            | Yes      |
| Render PDF pages      | Yes      |
| Store slide images    | Yes      |
| Multimodal embeddings | Yes      |
| Text-note embeddings  | Yes      |
| Vector retrieval      | Yes      |
| Reranking             | Yes      |
| Personal notes        | Yes      |
| Concept extraction    | Yes      |
| Search slides + notes | Yes      |
| Slide preview         | Yes      |
| Metadata storage      | Yes      |
| Retrieval benchmark   | Yes      |
| Recall@K evaluation   | Yes      |
| MRR evaluation        | Yes      |
| Chatbot               | No       |
| Audio recording       | No       |
| Transcription         | No       |
| Knowledge graph       | No       |
| Agent                 | No       |
| MCP                   | No       |
| Fine-tuning           | No       |

Do not implement V2 features before V1.5 is stable.

---

# 4. Expected User Experience

User launches the application.

Home screen:

```text
Lecture Memory

Courses

CS344
12 lectures

Linear Optimization
8 lectures

Machine Learning
10 lectures
```

User enters a course.

Example:

```text
CS344

Search course memory
[________________________________]

Lectures

Lecture 01
Algorithm Analysis

Lecture 02
Divide and Conquer

Lecture 03
Integer Multiplication
```

User searches:

```text
where did we discuss three-subproblem multiplication?
```

Expected result:

```text
CS344 — Lecture 03
Page 17

[slide preview]

Relevant note:
"Why does reducing four recursive calls to three improve complexity?"

Concepts:
Karatsuba
Divide and Conquer

Visual similarity:
0.84

Reranker score:
0.93
```

Clicking the result should open the corresponding slide preview.

---

# 5. Core Architecture

Use the following conceptual architecture.

```text
                    ┌──────────────────┐
                    │      Course      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     Lecture      │
                    └────────┬─────────┘
                             │
               ┌─────────────┼─────────────┐
               │             │             │
               ▼             ▼             ▼
           PDF Slides      Notes       Metadata
               │             │
               ▼             ▼
         Page Rendering   Text Chunking
               │             │
               ▼             ▼
        Visual Embedding  Text Embedding
               │             │
               └──────┬──────┘
                      ▼
                Vector Index
                      │
Query ────────────────┘
                      │
                      ▼
                   Top-K
                      │
                      ▼
                  Reranker
                      │
                      ▼
                 Final Results
```

---

# 6. Recommended Technology Stack

Backend:

```text
Python 3.11+
FastAPI
Pydantic
SQLAlchemy
SQLite
```

Frontend:

```text
Streamlit
```

Do not use React for V1 unless explicitly requested later.

PDF processing:

```text
PyMuPDF
```

Image handling:

```text
Pillow
```

Embedding:

```text
Qwen3-VL-Embedding
```

Reranking:

```text
Qwen3-VL-Reranker
```

Vector search:

```text
FAISS
```

Evaluation:

```text
Python
Pandas
NumPy
```

Testing:

```text
pytest
```

Code quality:

```text
ruff
```

Optional:

```text
mypy
```

---

# 7. Repository Structure

Create the project using approximately this structure:

```text
lecture-memory/
│
├── README.md
├── PROJECT_SPEC.md
├── pyproject.toml
├── .gitignore
├── .env.example
│
├── app/
│   ├── __init__.py
│   │
│   ├── main.py
│   │
│   ├── config.py
│   │
│   ├── database.py
│   │
│   ├── models.py
│   │
│   ├── schemas.py
│   │
│   ├── repositories/
│   │   ├── course_repository.py
│   │   ├── lecture_repository.py
│   │   └── note_repository.py
│   │
│   ├── ingestion/
│   │   ├── pdf_renderer.py
│   │   ├── document_processor.py
│   │   └── metadata.py
│   │
│   ├── embeddings/
│   │   ├── base.py
│   │   ├── qwen_embedding.py
│   │   └── cache.py
│   │
│   ├── retrieval/
│   │   ├── index.py
│   │   ├── search.py
│   │   ├── reranker.py
│   │   └── result.py
│   │
│   ├── concepts/
│   │   └── extractor.py
│   │
│   └── services/
│       ├── course_service.py
│       ├── lecture_service.py
│       ├── ingestion_service.py
│       └── search_service.py
│
├── frontend/
│   ├── streamlit_app.py
│   ├── pages/
│   │   ├── course.py
│   │   ├── lecture.py
│   │   └── search.py
│   └── components/
│       ├── slide_card.py
│       └── search_result.py
│
├── data/
│   ├── raw/
│   ├── rendered/
│   ├── embeddings/
│   ├── indexes/
│   └── database/
│
├── benchmark/
│   ├── queries.jsonl
│   ├── evaluator.py
│   ├── metrics.py
│   └── results/
│
├── scripts/
│   ├── ingest_pdf.py
│   ├── rebuild_index.py
│   └── evaluate.py
│
└── tests/
    ├── test_pdf_renderer.py
    ├── test_database.py
    ├── test_index.py
    ├── test_search.py
    └── test_metrics.py
```

This structure may be adjusted when necessary, but maintain strong separation between:

```text
data storage
document ingestion
model inference
retrieval
business logic
frontend
evaluation
```

Do not create one giant Python file.

---

# 8. Core Data Model

Use SQLite initially.

## Course

Fields:

```text
id
name
code
description
created_at
updated_at
```

Example:

```text
id: 1
name: Design and Analysis of Computer Algorithms
code: CS344
```

---

## Lecture

Fields:

```text
id
course_id
title
lecture_number
lecture_date
source_pdf
created_at
updated_at
```

Example:

```text
title:
Divide and Conquer

lecture_number:
3
```

---

## SlidePage

Fields:

```text
id
lecture_id
page_number
image_path
text_content
embedding_id
created_at
```

Do not store large embedding vectors directly inside SQLite unless necessary.

Prefer storing embedding arrays separately and linking them through IDs.

---

## Note

Fields:

```text
id
lecture_id
page_id optional
content
created_at
updated_at
```

A note may relate to:

```text
whole lecture
or
specific slide
```

---

## Concept

Fields:

```text
id
name
normalized_name
description optional
```

---

## LectureConcept

Fields:

```text
lecture_id
concept_id
confidence
source
```

Possible sources:

```text
slides
notes
auto_extracted
manual
```

---

# 9. Search Result Schema

Every search result should use one normalized object.

Example:

```python
SearchResult(
    result_type="slide",
    course_id=1,
    course_name="CS344",
    lecture_id=3,
    lecture_title="Divide and Conquer",
    page_number=17,
    preview_path="...",
    raw_similarity=0.84,
    reranker_score=0.93,
    text_preview="...",
    related_notes=[...],
    concepts=[...]
)
```

Do not let frontend code manipulate raw FAISS results directly.

The retrieval layer must return clean domain objects.

---

# 10. Development Rules for Codex

Codex must complete this project incrementally.

Never build the entire application in one pass.

For each task:

```text
1. Inspect existing repository.
2. Explain briefly what will be modified.
3. Modify only files required for the task.
4. Add tests.
5. Run relevant tests.
6. Fix failures.
7. Report changed files.
8. Report remaining limitations.
9. Stop.
```

Do NOT automatically move to the next task.

The human will review each stage before continuing.

Every task must preserve working behavior from previous tasks.

Do not perform large unrelated refactors while implementing a feature.

---

# 11. Phase 0 — Repository Bootstrap

## Task 0.1 — Project initialization

Goal:

Create the basic repository skeleton.

Implement:

```text
pyproject.toml
README.md
.gitignore
.env.example
package layout
pytest configuration
ruff configuration
```

Acceptance criteria:

```text
python environment installs successfully
pytest executes
ruff executes
basic app import succeeds
```

No AI model should be added yet.

---

## Task 0.2 — Configuration system

Create:

```text
app/config.py
```

Configuration should include:

```text
DATA_DIR
DATABASE_PATH
RENDERED_DIR
EMBEDDING_DIR
INDEX_DIR

DEVICE
MODEL_NAME
RERANKER_MODEL_NAME
```

Configuration should support environment-variable override.

Acceptance criteria:

```text
configuration loads with defaults
environment overrides work
directories can be created automatically
```

---

# 12. Phase 1 — Core Data Layer

## Task 1.1 — Database

Create SQLite database setup.

Implement:

```text
database initialization
SQLAlchemy session
database migration strategy
```

For this small project, simple explicit initialization is acceptable.

Avoid overengineering.

---

## Task 1.2 — Course model

Implement:

```text
Course model
create_course
list_courses
get_course
delete_course
```

Tests must cover CRUD behavior.

---

## Task 1.3 — Lecture model

Implement relationship:

```text
Course
  └── Lecture
```

Implement:

```text
create_lecture
list_course_lectures
get_lecture
delete_lecture
```

Tests required.

---

## Task 1.4 — Notes

Implement:

```text
create_note
edit_note
delete_note
list_lecture_notes
```

Support optional slide-page association.

At the end of Phase 1 the application should understand:

```text
courses
lectures
notes
```

No PDF processing yet.

---

# 13. Phase 2 — PDF Ingestion

## Task 2.1 — PDF renderer

Create:

```text
app/ingestion/pdf_renderer.py
```

Input:

```text
PDF path
```

Output:

```text
list of rendered page images
```

Each file should follow a stable naming scheme:

```text
course_<id>/
lecture_<id>/
page_0001.png
page_0002.png
...
```

Rendering should preserve sufficient quality for multimodal models.

Suggested initial resolution:

```text
approximately 150–200 DPI
```

Do not optimize aggressively yet.

Tests should use a small fixture PDF.

---

## Task 2.2 — SlidePage model

Add:

```text
SlidePage
```

Store:

```text
lecture ID
page number
image path
optional extracted text
```

Ingestion should populate one record per PDF page.

---

## Task 2.3 — Ingestion pipeline

Create a service:

```text
ingest_lecture_pdf()
```

Pipeline:

```text
PDF
↓
validate
↓
copy/store source file
↓
render pages
↓
create SlidePage records
↓
return ingestion summary
```

Example output:

```text
Lecture:
CS344 Lecture 3

Pages:
31

Rendered:
31

Failed:
0
```

Important:

Ingestion must be idempotent or detect duplicate processing.

Do not silently duplicate pages.

---

# 14. Phase 3 — Embedding Layer

## Task 3.1 — Embedding interface

Create an abstract interface.

Example methods:

```python
embed_text(text)
embed_image(image_path)
embed_query(query)
```

The rest of the application must NOT depend directly on Qwen-specific code.

---

## Task 3.2 — Qwen3-VL embedding adapter

Implement:

```text
Qwen3-VL-Embedding
```

Requirements:

```text
lazy model loading
device configuration
batch inference where possible
normalized vectors
clear error messages
```

Support:

```text
text
image
query
```

Do not fine-tune the model.

---

## Task 3.3 — Embedding cache

Do not recompute unchanged embeddings every application restart.

Store:

```text
entity ID
entity type
model name
embedding file path
content hash
created_at
```

If source content has not changed:

```text
reuse embedding
```

If source changed:

```text
recompute
```

---

## Task 3.4 — Slide embedding

Add pipeline:

```text
rendered page
↓
Qwen image embedding
↓
embedding cache
```

At completion, every slide page should have an embedding.

---

## Task 3.5 — Note embedding

Embed personal notes using text embeddings.

At this stage both:

```text
slide pages
notes
```

must be searchable.

---

# 15. Phase 4 — Vector Retrieval

## Task 4.1 — FAISS index abstraction

Create:

```text
app/retrieval/index.py
```

Responsibilities:

```text
build index
save index
load index
add vectors
search vectors
map vector IDs to entities
```

Do not expose FAISS directly to higher-level services.

---

## Task 4.2 — Unified retrieval

Search should support:

```text
slide pages
notes
```

Input:

```text
query
course filter optional
lecture filter optional
top_k
```

Output:

```text
SearchResult[]
```

Initial search logic:

```text
query
↓
query embedding
↓
FAISS
↓
Top-K nearest neighbors
```

---

## Task 4.3 — Search filtering

Support:

```text
all courses

specific course

specific lecture
```

Example:

```text
search:
"PCA"

filter:
Machine Learning
```

The result should not include unrelated courses.

---

# 16. Phase 5 — Reranking

## Task 5.1 — Reranker interface

Create an abstraction.

Input:

```text
query
candidate results
```

Output:

```text
reranked results
```

---

## Task 5.2 — Qwen3-VL reranker

Implement reranking for the top candidate results.

Recommended initial flow:

```text
retrieve Top 20
↓
rerank Top 20
↓
return Top 5
```

Make these configurable:

```text
RETRIEVAL_TOP_K
RERANK_TOP_K
FINAL_TOP_K
```

---

## Task 5.3 — Preserve both scores

Search results must expose:

```text
embedding similarity
reranker score
```

Do not overwrite one with the other.

This is required for later evaluation.

---

# 17. Phase 6 — Concept Extraction

This feature should remain simple.

Do not build a knowledge graph.

## Task 6.1 — Concept extractor

Given:

```text
lecture text
notes
available slide text if any
```

extract major concepts.

Example:

```text
Divide and Conquer
Karatsuba Multiplication
Recurrence Relation
Master Theorem
```

Store normalized concept names.

---

## Task 6.2 — Concept display

Each lecture page should show:

```text
Concepts

Divide and Conquer
Karatsuba
Recurrence Relations
```

Concepts do not initially need their own embedding index.

---

# 18. Phase 7 — Frontend

Use Streamlit.

Do not prioritize visual perfection.

Prioritize:

```text
clarity
fast navigation
stable functionality
```

---

## Task 7.1 — Home screen

Display:

```text
Lecture Memory
```

Then list courses.

Each course card should show:

```text
course name
course code
number of lectures
```

Actions:

```text
open
create course
```

---

## Task 7.2 — Course page

Display:

```text
course title
search box
lecture list
```

Allow:

```text
create lecture
upload PDF
```

---

## Task 7.3 — Lecture page

Display:

```text
lecture title
date
concepts
notes
slides
```

User can:

```text
add note
edit note
view slide
```

---

## Task 7.4 — Search result UI

Search result card must show:

```text
course
lecture
page number
slide preview
text preview
related note
embedding similarity
reranker score
```

Do not show excessively raw model output.

---

# 19. Phase 8 — Benchmark Dataset

This phase is essential.

Do not skip it.

The project must demonstrate that retrieval quality is measured rather than assumed.

Create:

```text
benchmark/queries.jsonl
```

Example:

```json
{
  "query": "slide explaining convex hull",
  "course": "Linear Optimization",
  "relevant_pages": [
    {
      "lecture_id": 4,
      "page_number": 12
    }
  ]
}
```

Another example:

```json
{
  "query": "where did we discuss three recursive multiplication calls?",
  "course": "CS344",
  "relevant_pages": [
    {
      "lecture_id": 3,
      "page_number": 17
    }
  ]
}
```

Target first benchmark size:

```text
50 queries minimum
100 preferred
```

Queries should be realistic.

Avoid artificially copying slide titles.

Good:

```text
graph showing convex and non-convex shapes
```

Bad:

```text
Convex Sets Definition
```

unless that is how a real user would search.

---

# 20. Phase 9 — Evaluation

Implement metrics:

```text
Recall@1
Recall@5
Recall@10
MRR
```

Optional later:

```text
nDCG
```

---

## Task 9.1 — Baseline retrieval

Implement at least one lexical baseline.

Preferred:

```text
BM25
```

The evaluation should compare:

```text
BM25
Qwen multimodal embedding
Qwen embedding + reranker
```

If text extraction is needed for BM25, keep that pipeline isolated from the visual retrieval pipeline.

---

## Task 9.2 — Evaluation runner

Command:

```bash
python scripts/evaluate.py
```

Output example:

```text
Method                 R@1    R@5    MRR
------------------------------------------------
BM25                    .51    .71    .60
Qwen Embedding          .68    .84    .75
Qwen + Reranker         .76    .91    .82
```

Also save results to:

```text
benchmark/results/
```

Prefer:

```text
CSV
JSON
```

---

## Task 9.3 — Latency evaluation

Also record:

```text
embedding latency
retrieval latency
reranking latency
total query latency
```

This helps show engineering tradeoffs.

---

# 21. Phase 10 — README

README is part of the product.

It must clearly explain:

## Problem

Students often remember seeing or learning something but cannot remember where it appeared.

Traditional keyword search performs poorly when information is contained in:

```text
diagrams
equations
visual layouts
screenshots
slides
personal shorthand notes
```

---

## Solution

Lecture Memory creates a searchable memory across:

```text
lecture slides
personal notes
course structure
concept metadata
```

using multimodal semantic retrieval and reranking.

---

## README sections

Recommended structure:

```text
Overview

Demo

Why Lecture Memory?

Architecture

Features

How It Works

Installation

Usage

Evaluation

Benchmark Results

Project Structure

Limitations

Roadmap
```

The README must include a system architecture diagram.

---

# 22. Explicit Non-Goals

Codex must NOT implement the following during V1.5 unless explicitly requested:

```text
audio recording
live transcription
Zoom integration
Google Meet integration
calendar integration
chatbot interface
LLM answer generation
agent workflows
MCP server
mobile app
browser extension
cloud deployment
multi-user accounts
authentication
knowledge graph
automatic flashcards
spaced repetition
YouTube ingestion
web crawling
fine-tuning
model training
OCR-heavy architecture
```

These are future extensions.

---

# 23. Future V2

Do NOT implement now.

V2 concept:

> Granola-style lecture capture.

Possible architecture:

```text
Lecture Session
     │
 ┌───┼────────────┐
 │   │            │
Audio Slides     Notes
 │   │            │
 ▼   ▼            ▼
Transcript     User shorthand
      \         /
       \       /
       AI synthesis
            │
            ▼
      Lecture Memory
```

Potential features:

```text
lecture recording
transcript
timestamped notes
AI-enhanced notes
professor emphasis
automatic summary
important moments
```

---

# 24. Future V3

Do NOT implement now.

V3 concept:

> Cross-course knowledge memory.

Example:

```text
PCA
├── Linear Algebra
│   └── SVD
│
├── Machine Learning
│   └── Dimensionality Reduction
│
└── Data Science
    └── Feature Engineering
```

Possible query:

```text
Where have I seen PCA before?
```

Possible result:

```text
Linear Algebra — Lecture 8
Machine Learning — Lecture 4
Data Science — Lecture 7
```

---

# 25. Engineering Quality Requirements

Code should follow these principles.

### Separation of concerns

Model inference should not exist inside Streamlit files.

Database logic should not exist inside embedding classes.

Retrieval logic should not exist inside UI components.

---

### Reproducibility

The repository should contain everything needed to reproduce:

```text
environment
index creation
evaluation
benchmark results
```

except copyrighted/private lecture files.

---

### Logging

Use structured logging for important operations:

```text
PDF ingestion
model loading
embedding generation
index creation
search
reranking
errors
```

Avoid excessive debug noise.

---

### Failure handling

The application should fail gracefully when:

```text
invalid PDF uploaded
model unavailable
CUDA unavailable
index missing
database missing
file deleted
embedding cache corrupted
```

Return useful error messages.

Do not crash silently.

---

### Performance

Avoid loading the Qwen model repeatedly.

Model should load once per process where practical.

Cache:

```text
rendered pages
embeddings
FAISS index
```

---

# 26. Testing Requirements

Each module should have tests where reasonable.

Mandatory test coverage:

```text
course CRUD
lecture CRUD
note CRUD
PDF rendering
duplicate ingestion
embedding cache behavior
FAISS index
metadata mapping
search filtering
retrieval ranking
MRR
Recall@K
```

AI model integration tests may be marked separately because they are expensive.

Example:

```text
@pytest.mark.integration
```

Unit tests must not require downloading several GB of models.

Use mock embedding providers where necessary.

---

# 27. Git Commit Strategy

Prefer one logical commit per completed task.

Examples:

```text
feat: initialize project structure

feat: add course and lecture persistence

feat: implement PDF page rendering

feat: add embedding abstraction

feat: integrate Qwen3-VL embeddings

feat: add FAISS retrieval

feat: implement multimodal reranking

feat: add Streamlit search interface

feat: add retrieval benchmark

feat: add Recall@K and MRR evaluation
```

Avoid commits such as:

```text
update stuff
fix
changes
final
```

---

# 28. Definition of Done for Each Codex Task

A task is not complete just because code was written.

Every task must finish with:

```text
implementation complete
tests added
tests executed
tests passing
changed files reported
known limitations reported
no unrelated changes
```

If tests cannot run, Codex must explain exactly why.

---

# 29. Codex Response Format After Every Task

After completing one task, Codex should return:

```text
Completed:
Task X.X — <task name>

Changed:
- file A
- file B
- file C

Implemented:
<short explanation>

Tests:
<commands executed>

Result:
<pass/fail>

Known limitations:
<limitations>

Next recommended task:
Task X.X
```

Do not continue automatically.

---

# 30. First Codex Instruction

Start ONLY with:

## Task 0.1 — Project Initialization

Instruction to Codex:

```text
Read PROJECT_SPEC.md completely before making changes.

Implement only Task 0.1.

Create the initial Python project structure for Lecture Memory.

Requirements:

1. Use Python 3.11+.
2. Create pyproject.toml.
3. Configure pytest.
4. Configure ruff.
5. Create the base app, frontend, benchmark, scripts, tests, and data directories described in PROJECT_SPEC.md.
6. Add minimal __init__.py files where appropriate.
7. Add a minimal README.md describing the project in one paragraph.
8. Add .gitignore suitable for Python, local model caches, generated embeddings, rendered PDFs, SQLite files, and environment files.
9. Add .env.example.
10. Do not implement databases, PDF processing, embeddings, FAISS, Qwen, or frontend functionality yet.

Add one minimal smoke test confirming that the app package can be imported.

Run:

pytest
ruff check .

Fix any errors.

At the end, report:

- files created
- tests executed
- whether they passed
- any decisions that differ from PROJECT_SPEC.md

Stop after Task 0.1.
Do not begin Task 0.2.
```

---

# 31. Recommended Development Order

The required implementation sequence is:

```text
0.1 Project initialization
0.2 Configuration

1.1 Database
1.2 Course
1.3 Lecture
1.4 Notes

2.1 PDF renderer
2.2 SlidePage
2.3 Ingestion pipeline

3.1 Embedding interface
3.2 Qwen embedding adapter
3.3 Embedding cache
3.4 Slide embeddings
3.5 Note embeddings

4.1 FAISS abstraction
4.2 Unified retrieval
4.3 Search filtering

5.1 Reranker interface
5.2 Qwen reranker
5.3 Score preservation

6.1 Concept extraction
6.2 Concept display

7.1 Home UI
7.2 Course UI
7.3 Lecture UI
7.4 Search UI

8 Benchmark dataset

9.1 BM25 baseline
9.2 Evaluation runner
9.3 Latency benchmark

10 README + Demo
```

Do not reorder major phases unless there is a clear technical reason.

---

# 32. Final V1.5 Definition of Done

V1.5 is complete when a user can:

```text
create a course
create a lecture
upload a lecture PDF
have pages automatically rendered
have pages embedded
write lecture notes
search using natural language
retrieve relevant slides and notes
view slide previews
see reranked results
see related lecture metadata
see extracted concepts
```

And the repository can demonstrate:

```text
BM25 baseline
multimodal embedding retrieval
embedding + reranker retrieval

Recall@1
Recall@5
Recall@10
MRR

query latency
```

The finished repository should communicate three competencies:

```text
AI / ML
multimodal retrieval and reranking

Experimentation
benchmarking and quantitative evaluation

Software Engineering
end-to-end architecture, persistence, caching, testing and UI
```

The project should remain small enough to understand end-to-end.

Do not sacrifice clarity for feature count.

The guiding principle is:

> Build a small, complete AI system rather than a large collection of AI features.
