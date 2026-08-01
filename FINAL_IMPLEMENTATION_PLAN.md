# Smart AI Router: Final Engineering Blueprint & Implementation Plan

> **Document Status**: Final Architectural Roadmap  
> **Target Outcome**: Zero pending software engineering required prior to pure research experimentation.

---

## 1. Executive Summary & Research Philosophy

The **Smart AI Router** is a specialized, highly controlled experimentation platform designed to evaluate and benchmark context routing and compression principles under strictly standardized conditions.

### Fixed Controls vs. Experimental Variable

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   FIXED PIPELINE                                       │
│                                                                                        │
│   LexGLUE Dataset  ──► Preprocessing ──► [ ROUTING PRINCIPLE ] ──► Context Compression │
│                                                  ▲                                     │
│                                                  │                                     │
│                                          (Sole Variable)                               │
│                                                                                        │
│   Evaluation ◄── Gemini Flash (Fixed LLM) ◄──────┘                                     │
│       │                                                                                │
│       ▼                                                                                │
│   Auto-Report Generation (Markdown / PDF / CSV / JSON)                                 │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

- **Fixed Infrastructure**: Dataset (**LexGLUE**), Downstream LLM (**Gemini Flash API**), Evaluation Metrics, Latency & Cost Accounting, and Automatic Report Generation.
- **Experimental Variable**: **Routing Principle** (BM25, TF-IDF, Cosine Embedding, Logistic Regression, PyTorch Neural Router, Token Importance / Pruning, Attention / Gradient Methods).

---

## 2. Complete UI System Design & Workflow

The user interface is intentionally minimal, distraction-free, and streamlined so an experiment can be configured, executed, and analyzed in **fewer than 3 clicks**.

### 2.1 Navigation & Screen Architecture

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│  Smart AI Router |   [1. Home]  [2. Experiments]  [3. Benchmark All]             │
│                     [4. Reports]  [5. Models]       [6. Settings]                 │
└───────────────────────────────────────────────────────────────────────────────────┘
```

#### Page Breakdown & Wireframes

1. **Home (`/`)**
   - System overview, platform status, dataset integrity, active Gemini API status, quick metrics cards (Total Experiments, Active Router Models, Baseline Benchmarks).
2. **Experiments (`/experiments`)**
   - **Single Experiment Workflow**:
     - Step 1: Select Routing Principle (Dropdown: BM25, TF-IDF, Embedding, Logistic Regression, Neural Router, Custom).
     - Step 2: Configure Parameters (Compression Target %, Threshold / Top-K Policy).
     - Step 3: Click `[Run Experiment]`.
   - Live execution output console, progress indicator, real-time token/cost calculation display.
3. **Benchmark All (`/benchmark-all`)**
   - Single-click action: `[Run Full Comparative Suite]`.
   - Automatically runs all registered routing principles against the standardized LexGLUE test split with fixed Gemini Flash downstream evaluation.
   - Generates a consolidated comparative matrix report.
4. **Reports (`/reports`)**
   - Report browser, search, and comparative overlay tool.
   - Live view of Markdown reports, PDF download trigger, CSV/JSON data export buttons, and side-by-side metric comparison graphs (Precision vs. Recall vs. Downstream Accuracy vs. Token Cost Savings).
5. **Models (`/models`)**
   - Router model registry viewer (Logistic Regression weights, PyTorch Transformer checkpoints).
   - Single-click training triggers: `[Train Logistic Regression]`, `[Fine-Tune Neural Router]`.
6. **Settings (`/settings`)**
   - Configuration management: Gemini API Key (masked input), model directory paths, LexGLUE cache path, report output paths, log levels.

---

## 3. Subsystem Architectural Breakdown & Gap Analysis

```
┌─────────────────────────────┬──────────────────────────┬───────────────────────────────────────────┐
│ Subsystem                   │ Completion Status        │ Required Work to Complete Engineering     │
├─────────────────────────────┼──────────────────────────┼───────────────────────────────────────────┤
│ 1. Dataset Engine           │ 85% Completed            │ Finalize LexGLUE test split cache & schema│
│ 2. Scorer & Routing Layer   │ 90% Completed            │ Standardize interface for all principles   │
│ 3. Downstream Gemini LLM    │ 30% Completed            │ Implement official Gemini API provider    │
│ 4. Evaluation Engine        │ 70% Completed            │ Integrate downstream accuracy + judge     │
│ 5. Reporting & Exporting    │ 20% Completed            │ Build PDF, Markdown, CSV, JSON generators │
│ 6. Registry & Storage       │ 40% Completed            │ Implement SQLite/JSON experiment registry  │
│ 7. Web UI & Visualization   │ 10% Completed            │ Build FastAPI + React / Modern Web Dashboard│
└─────────────────────────────┴──────────────────────────┴───────────────────────────────────────────┘
```

---

## 4. Engineering Roadmap & Milestone Breakdown

```
Milestone 9: Gemini Flash Downstream Integration & Evaluator
     │
     ▼
Milestone 10: Registry, Storage, & Experiment Tracking Engine
     │
     ▼
Milestone 11: Automated Multi-Format Report & Export System
     │
     ▼
Milestone 12: Streamlined 6-Page UI & Visualization Platform
     │
     ▼
Milestone 13: Final Architecture Cleanup, Refactoring & Test Verification
```

---

### Milestone 9: Gemini Flash Downstream Integration & Evaluator

#### Purpose
Establish official **Gemini Flash API** downstream integration, token accounting, cost metrics, and accuracy evaluation against LexGLUE QA references.

#### Required New & Modified Files
- `evaluation/gemini_provider.py` `[NEW]`
  - `GeminiFlashProvider`: Inherits `BaseLLMProvider`. Executes requests to Gemini 1.5/2.0 Flash API via Google GenAI SDK.
  - Measures API latency, prompt tokens, completion tokens, estimated cost ($0.075 / 1M prompt tokens, $0.30 / 1M completion tokens).
- `evaluation/llm_eval.py` `[MODIFY]`
  - Integrate downstream generation + LLM-as-a-Judge semantic accuracy metrics.
- `tests/test_gemini_provider.py` `[NEW]`
  - Mocked integration tests verifying fallback, rate limiting, and token counter calculations.

---

### Milestone 10: Registry, Storage, & Experiment Tracking Engine

#### Purpose
Create a unified, persistent experiment and model registry store to record experiment configs, metrics, compression rates, token savings, and model checkpoints.

#### Required New & Modified Files
- `storage/experiment_registry.py` `[NEW]`
  - SQLite metadata database (`data/registry.db`) logging run IDs, timestamps, routing principles, hyperparams, and metric snapshots.
- `storage/artifact_store.py` `[NEW]`
  - Persistent storage for generated context logs, predictions, and model `.pkl` / `.pt` checkpoints.
- `tests/test_experiment_registry.py` `[NEW]`
  - Verification of query, insertion, and retrieval operations.

---

### Milestone 11: Automated Multi-Format Report & Export System

#### Purpose
Provide automated generation of comparative evaluation reports in Markdown, PDF, CSV, and JSON formats, complete with matplotlib vector graphics.

#### Required New & Modified Files
- `reporting/report_generator.py` `[NEW]`
  - Renders Markdown and PDF summary reports for single and comparative benchmarks.
- `reporting/exporters.py` `[NEW]`
  - CSV/JSON serialization of benchmark matrices.
- `reporting/plots.py` `[NEW]`
  - Programmatic generation of latency vs. accuracy scatter plots, context reduction bar charts, and cost efficiency curves.
- `tests/test_reporting.py` `[NEW]`

---

### Milestone 12: Streamlined 6-Page UI & Visualization Platform

#### Purpose
Develop the clean, low-friction Web UI exposing the 6 key screens (Home, Experiments, Benchmark All, Reports, Models, Settings).

#### Architecture
- **Backend API**: FastAPI framework (`ui/backend/app.py`) providing REST endpoints for experiment launching, benchmark sweeps, model training, and report fetching.
- **Frontend**: Vite + React + Vanilla CSS Modern Dashboard (`ui/frontend/`) with dark mode aesthetics and zero superfluous controls.

#### API Routes
- `GET /api/status`: System status and configuration.
- `POST /api/experiments/run`: Launch single experiment.
- `POST /api/experiments/benchmark-all`: Launch full comparative benchmark suite.
- `GET /api/reports`: List and fetch generated reports.
- `GET /api/models`: List available router models and trigger training.
- `POST /api/settings`: Save Gemini API keys and workspace settings.

---

### Milestone 13: Architecture Cleanup, Refactoring & Technical Debt Removal

#### Purpose
Ensure the repository has zero dead code, robust test coverage (>90%), strictly enforced typing, consistent coding conventions, and comprehensive documentation.

---

## 5. What Belongs to the "Research Phase" (Non-Framework Scope)

To maintain absolute architectural clarity, the following items are strictly categorized as **Research Phase** tasks and will NOT be baked into framework infrastructure:

1. **Novel Routing Algorithm Design**: Developing custom attention-weight extraction, gradient-based saliency routing, or hybrid token importance scoring algorithms.
2. **Model Training & Hyperparameter Tuning**: Conducting hyperparameter grid searches for custom PyTorch neural router architectures.
3. **Comparative Research Experiments**: Running experimental sweeps to gather empirical data for academic papers or benchmark evaluations.
4. **Academic Paper & Publication Writing**: Formulating theoretical conclusions, writing LaTeX manuscripts, and producing research figures from exported report data.

---

## 6. Implementation Order & Testing Strategy

```
Phase 1: Engineering Infrastructure (Milestones 9 - 11)
  ├── 1. Gemini Flash Provider & Cost/Token Accounting
  ├── 2. SQLite Experiment & Artifact Registry
  └── 3. Multi-Format Report Generator (Markdown/PDF/CSV/JSON)

Phase 2: UI & User Experience (Milestone 12)
  ├── 1. FastAPI REST Backend Endpoints
  └── 2. Streamlined 6-Page Web UI

Phase 3: Verification & Cleanup (Milestone 13)
  ├── 1. End-to-End Synthetic & Mock Test Suite Verification
  └── 2. Documentation Finalization
```

### Automated Testing Assurance
- Every milestone will be gated by `pytest` suite executions (`PYTHONPATH=. ./.venv/bin/pytest`).
- Mock providers will ensure complete offline CI testability without requiring live API keys or active internet access during test runs.

---

## 7. Approval & Sign-Off

Upon confirmation of this document (`FINAL_IMPLEMENTATION_PLAN.md`), software engineering work will proceed strictly according to this blueprint until the platform is fully completed and ready for research execution.
