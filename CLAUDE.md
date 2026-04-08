# CLAUDE.md

## Project

Document Research Assistant — a multi-document research assistant with configurable corpus support. See `project-plan.md` for full architecture, phasing, and rationale.

**Current phase: 1 (Vanilla RAG Baseline)**
Scope: dense retrieval + single Pydantic AI agent for generation, wired into eval harness. Establishing baseline scores.

## Commands

```bash
uv run python scripts/ingest.py        # Download, parse, embed, upsert EDGAR filings
uv run python scripts/run_eval.py      # Run full eval suite, report to Logfire
uv run pytest                           # Unit tests
uv run pytest -x -q                     # Fast fail
```

## Tech Stack Constraints

These are deliberate decisions — do not change without discussion.

- **Package management**: `uv` only. Use `uv run` for commands, `uv add`/`uv remove` for deps. Never edit pyproject.toml deps manually.
- **Embeddings**: `fastembed` (ONNX). Not sentence-transformers, not PyTorch. The embedding interface should be swappable via config but the default implementation uses fastembed.
- **Sparse vectors**: fastembed SPLADE or Qdrant BM25. Validate SPLADE performance in Phase 0 before committing.
- **Vector DB**: `qdrant-client`. Local mode (`path="./qdrant_data"`) for dev. Switching to Docker/Cloud is config-only via `QDRANT_MODE`.
- **Agent framework**: Pydantic AI directly. No orchestration protocol layer (QueryEngine/Retriever protocols were deliberately dropped). If a LangGraph comparison happens later, extract protocols then.
- **Graph API**: pydantic-graph class-based API (`BaseNode` subclasses). Not the beta builder API.
- **LLM (generation)**: `anthropic:claude-haiku-4-5-20251001`
- **LLM (eval judge)**: ideally a different model family from generation to avoid systematic bias. Default: `openai:gpt-5.4-nano`.
- **Evals**: `pydantic-evals`. RAGAS metrics implemented as custom evaluators within pydantic-evals, not as a separate framework.
- **Observability**: Logfire. `logfire.instrument_pydantic_ai()` for agent tracing.
- **Testing**: pytest with class-based tests. Shared fixtures in conftest.py. Qdrant in-memory mode for tests.
- **Reranking** (Phase 2+): ONNX cross-encoders. Same runtime as fastembed, no PyTorch dep.

## Architecture Invariants

- **Domain logic lives in the Corpus, not in agents.** Tools, system prompt (`instructions`), chunking strategy, metadata schema, and eval generation are all provided by the `Corpus` protocol implementation (e.g. `EdgarCorpus`). Agents read from the corpus — they don't hardcode domain knowledge.
- **Eval harness is corpus-agnostic.** pydantic-evals runs cases against a task function. The evaluators may be corpus-specific (e.g. `NumericMatch` for XBRL) but the harness itself doesn't know about EDGAR.
- **No premature directories.** Only create modules/directories needed for the current phase. Future structure is documented in the plan, not scaffolded as empty files.
- **Config via pydantic-settings.** All environment-specific values (API keys, Qdrant mode, model names, top_k) come from `Settings` loaded from env vars (no prefix). No hardcoded values in application code.

## Code Style

- Type hints on all function signatures, return types, and non-trivial variables.
- No docstrings, READMEs, or comments unless explicitly requested.
- Long-lived comments must be timeless — no references to "this refactor" or "we changed this because".
- Prefer clean implementations over backward compatibility. Flag breaking changes but default to cleanest design.
- Propose architectural refactors proactively if complexity is building.
- Flag uncertainty explicitly. Prefer "I'm not sure" over plausible-sounding guesses.
- Plan before building. Do not begin implementation until approach is agreed.

## Eval Dataset

- **Factual (auto-generated)**: latest fiscal year only per company-metric (~55 cases). Multi-year retrieval is covered by the temporal dataset.
- **Comparison (auto-generated)**: 2 pairs per concept (~20 cases).
- **Temporal / Multihop**: manually curated, 5-10 seeds each. Expand after Phase 1 when real failure modes are visible.
- **Stored as**: YAML in `src/research_assistant/eval/datasets/`, version controlled.
- **Reporting**: per-tier accuracy with 95% confidence intervals. Never a single aggregate score.

## Documentation

- **`project-plan.md`** — stable reference. Updated only for structural changes (tech stack swap, phase dropped, new corpus added). Not a log.
- **`docs/phases/phase-N.md`** — written at the end of each phase. Captures what was tried, eval results (with links to Logfire), what worked/didn't, decisions made, and anything that diverged from the plan. These are the living documents.
- **Eval results** live in Logfire, not duplicated in markdown. Phase docs reference Logfire experiment IDs.
- **When starting a new phase**, read the previous phase doc (`docs/phases/phase-{N-1}.md`) for context on baseline scores and lessons learned. Update the "Current phase" line at the top of this file.

## What Not To Build (Yet)

- FastAPI / serving layer
- Fine-tuning or custom model training
- GraphRAG / knowledge graphs
- `agents/graph/` directory (Phase 5-6)
- `agents/langgraph/` directory (Phase 7)
- `retrieval/hybrid.py` or `retrieval/reranker.py` (Phase 2)
