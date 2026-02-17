# OpenJarvis Development Notes

Living document tracking implementation progress, testing state, lessons learned, dead ends, and practices for ongoing development. Updated across sessions.

---

## Current State (2026-02-16)

- **Version:** 1.0.0
- **All 6 roadmap phases complete** (Phase 0 through Phase 5)
- **Tests:** 520 passed, 8 skipped, 0 failures
- **Lint:** ruff clean (`select = ["E", "F", "I", "W"]`)
- **Source files:** 72 Python files in `src/openjarvis/`
- **Test files:** 74 Python files in `tests/`
- **Python:** 3.13 (compatible with 3.10+)
- **Package manager:** `uv` with `hatchling` build backend

### 8 Skipped Tests (Optional Dependencies)

| Test | Missing Dep | Install Extra |
|------|-------------|---------------|
| `tests/memory/test_bm25.py` | `rank_bm25` | `openjarvis[memory-bm25]` |
| `tests/memory/test_colbert.py` | `colbert` | `openjarvis[memory-colbert]` |
| `tests/memory/test_embeddings.py` | `sentence_transformers` | `openjarvis[memory-faiss]` |
| `tests/memory/test_faiss.py` | `faiss` | `openjarvis[memory-faiss]` |
| `tests/server/test_models_pydantic.py` | `pydantic` | `openjarvis[server]` |
| `tests/server/test_routes.py` | `fastapi` | `openjarvis[server]` |
| `tests/test_integration.py:165` | `fastapi` | `openjarvis[server]` |
| `tests/test_integration.py:190` | `fastapi` | `openjarvis[server]` |

---

## Phase Completion Log

| Phase | Version | Deliverables | Test Count (cumulative) |
|-------|---------|-------------|------------------------|
| Phase 0 | v0.1 | Scaffolding, registries, core types, config, CLI skeleton, event bus | ~60 |
| Phase 1 | v0.2 | Intelligence + Inference — `jarvis ask` end-to-end, heuristic router, engine discovery, basic telemetry | ~160 |
| Phase 2 | v0.3 | Memory — SQLite/FAISS/ColBERT/BM25/Hybrid backends, document ingest pipeline, context injection, `jarvis memory` CLI | ~270 |
| Phase 3 | v0.4 | Agents (Simple/Orchestrator/Custom/OpenClaw stub), tool system (Calculator/Think/Retrieval/LLM/FileRead), OpenAI-compatible API server, `jarvis serve` | ~360 |
| Phase 4 | v0.5 | Learning — HeuristicRouter, HeuristicRewardFunction, GRPORouterPolicy stub, TelemetryAggregator, `jarvis telemetry` CLI, `--router` CLI option | ~432 |
| Phase 5 | v1.0 | SDK (`Jarvis` class), OpenClaw infrastructure (protocol/transport/plugin), benchmarks (`jarvis bench`), Docker, docs | ~520 |

---

## Architecture Quick Reference

### Directory Layout

```
src/openjarvis/
├── __init__.py          # __version__ = "1.0.0", exports Jarvis, MemoryHandle
├── sdk.py               # Python SDK: Jarvis class + MemoryHandle
├── core/
│   ├── registry.py      # RegistryBase[T] + 7 typed registries
│   ├── types.py         # Message, Conversation, ModelSpec, ToolResult, TelemetryRecord
│   ├── config.py        # JarvisConfig dataclass hierarchy, TOML loader
│   └── events.py        # EventBus pub/sub (synchronous)
├── intelligence/        # ModelRegistry, HeuristicRouter, model catalog
├── learning/            # RouterPolicyRegistry, HeuristicRouter policy, GRPO stub
├── memory/              # SQLite/FAISS/ColBERT/BM25/Hybrid backends, chunking, ingest
├── agents/              # Simple/Orchestrator/Custom/OpenClaw agents + protocol/transport
├── engine/              # Ollama/vLLM/llama.cpp/Cloud engine wrappers
├── tools/               # Calculator/Think/Retrieval/LLM/FileRead tools
├── bench/               # Latency/Throughput benchmarks, BenchmarkSuite
├── telemetry/           # TelemetryStore, TelemetryAggregator, instrumented_generate
├── server/              # FastAPI OpenAI-compatible API server
└── cli/                 # Click CLI: init, ask, serve, model, memory, telemetry, bench
```

### 7 Registries

All use `RegistryBase[T]` with `@XRegistry.register("name")` or `register_value()`:

1. `ModelRegistry` — `ModelSpec` objects
2. `EngineRegistry` — `InferenceEngine` implementations
3. `MemoryRegistry` — `MemoryBackend` implementations
4. `AgentRegistry` — `BaseAgent` implementations
5. `ToolRegistry` — `BaseTool` implementations
6. `RouterPolicyRegistry` — `RouterPolicy` implementations
7. `BenchmarkRegistry` — `BaseBenchmark` implementations

---

## Patterns and Practices

### The `ensure_registered()` Pattern

**Problem:** The `_clean_registries` autouse fixture in `tests/conftest.py` calls `.clear()` on every registry before each test. Module-level `@XRegistry.register("name")` decorators only fire once at import time (Python caches modules in `sys.modules`). After registry clearing, the decorations never re-fire, leaving registries empty for subsequent tests.

**Solution:** Use lazy registration via `ensure_registered()`:

```python
# src/openjarvis/bench/latency.py
_registered = False

def ensure_registered() -> None:
    global _registered
    if _registered:
        return
    from openjarvis.core.registry import BenchmarkRegistry
    if not BenchmarkRegistry.contains("latency"):
        BenchmarkRegistry.register_value("latency", LatencyBenchmark)
    _registered = True
```

Then in `__init__.py`:
```python
def ensure_registered() -> None:
    from openjarvis.bench.latency import ensure_registered as _reg_latency
    _reg_latency()
```

And in test files, use an autouse fixture:
```python
@pytest.fixture(autouse=True)
def _register_latency():
    from openjarvis.bench import ensure_registered
    ensure_registered()
```

**Where this pattern is used:** `bench/latency.py`, `bench/throughput.py`, `learning/heuristic_policy.py`, `learning/grpo_policy.py`, `learning/heuristic_reward.py`

**Where this pattern is NOT needed:** Agents, engines, memory backends, and tools use `@register` decorators that work fine because their test files explicitly import and re-register as needed, or the test module import triggers registration.

### Test Infrastructure

- **`tests/conftest.py`** — `_clean_registries` autouse fixture clears all 7 registries + clears `EventBus` default listeners before each test. Critical for test isolation.
- **Mock engine pattern** — Almost every test that touches the engine layer uses a `MagicMock()` with `.engine_id`, `.health()`, `.list_models()`, `.generate()` stubbed:
  ```python
  def _make_engine(content="Hello"):
      engine = MagicMock()
      engine.engine_id = "mock"
      engine.health.return_value = True
      engine.list_models.return_value = ["test-model"]
      engine.generate.return_value = {
          "content": content,
          "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
          "model": "test-model",
          "finish_reason": "stop",
      }
      return engine
  ```
- **CLI tests** use Click's `CliRunner` with `patch("openjarvis.cli.X.get_engine", ...)` to mock the engine layer.
- **Memory tests** use `tmp_path` fixture for SQLite DB paths and test files.
- **Optional dep tests** use `pytest.importorskip("module_name")` at module level.

### Config Defaults

`JarvisConfig()` with no arguments produces sane defaults:
- Engine: auto-discover (Ollama, vLLM, llama.cpp, cloud in priority order)
- Memory: `sqlite` backend, `~/.openjarvis/memory.db`
- Agent: `simple` (no Node.js dependency)
- Intelligence: `qwen3:8b` default, `qwen3:0.6b` fallback
- Telemetry: enabled, `~/.openjarvis/telemetry.db`
- Learning: `heuristic` default policy

### File Naming Conventions

- ABCs and shared dataclasses: `_stubs.py` (e.g., `agents/_stubs.py`, `bench/_stubs.py`, `tools/_stubs.py`)
- Internal helpers: `_discovery.py`, `_base.py` (underscore prefix)
- CLI commands: `*_cmd.py` (e.g., `bench_cmd.py`, `telemetry_cmd.py`, `memory_cmd.py`)
- Test files mirror source: `tests/agents/test_openclaw.py` tests `src/openjarvis/agents/openclaw.py`

### Import Structure

- Package `__init__.py` files import submodules to trigger registration
- Try/except around optional dependency imports:
  ```python
  try:
      from openjarvis.engine.ollama import OllamaEngine  # noqa: F401
  except ImportError:
      pass
  ```
- Top-level `openjarvis/__init__.py` exports: `Jarvis`, `MemoryHandle`, `__version__`

---

## Dead Ends and Gotchas

### 1. `@register` Decorator vs. `ensure_registered()`

**Dead end:** Initially used `@BenchmarkRegistry.register("latency")` class decorator in `bench/latency.py`. This caused ~10 test failures because:
- Registry cleared between tests by `conftest.py`
- Module already in `sys.modules`, so `import openjarvis.bench` is a no-op on second import
- Registry stays empty after clearing

**Fix:** Switched to `ensure_registered()` pattern (see above). This is the pattern already used by `learning/` modules.

**Rule of thumb:** If a module is imported at package init time AND its registry gets cleared in tests, use `ensure_registered()`. If registration only happens in test fixtures or explicit calls, `@register` is fine.

### 2. Chunk Attribute Names

`memory/chunking.py` `Chunk` dataclass uses `content` (not `text`). `ChunkConfig` uses `chunk_overlap` (not `overlap`). Easy to get wrong because these aren't obvious from the field names alone. Always read `_stubs.py` or the actual dataclass before using.

### 3. Test Content Size for Chunking

`ChunkConfig.min_chunk_size=50` tokens by default. A test string like `"This is test content."` produces 0 chunks. Use at least ~100 words:
```python
words = " ".join(f"word{i}" for i in range(100))
```

### 4. Version String Locations

Version is defined in **three places** that must stay in sync:
1. `src/openjarvis/__init__.py` — `__version__ = "1.0.0"`
2. `pyproject.toml` — `version = "1.0.0"`
3. `src/openjarvis/server/app.py` — FastAPI `version="1.0.0"` constructor arg

Tests that check version: `tests/cli/test_cli.py::test_version_flag`

### 5. Server Import Guards

The `server/` module requires `fastapi`, `uvicorn`, `pydantic`. These are behind the `[server]` optional extra. All test files that touch server code use `pytest.importorskip("fastapi")`. The server `__init__.py` wraps imports in try/except.

### 6. `patch()` Targets for Engine Mocking

When mocking `get_engine` in CLI tests, the patch target must be the *importing module*, not the source module:
```python
# CORRECT — patches where it's imported
patch("openjarvis.cli.bench_cmd.get_engine", return_value=("mock", engine))

# WRONG — patches the source, doesn't affect the already-imported reference
patch("openjarvis.engine._discovery.get_engine", return_value=("mock", engine))
```

Same for SDK tests: `patch("openjarvis.sdk.get_engine", ...)`.

### 7. EventBus Clearing

`EventBus()` creates a new instance each time, but `EventBus._default_listeners` is a class variable. The `conftest.py` fixture resets it. If tests subscribe to events, subscriptions won't persist across tests.

---

## Post-v1.0: Unimplemented Ideas from VISION.md

These are mentioned in `VISION.md` but not in the roadmap phases. They represent future work:

### Learning / Router
- [ ] Learned router via GRPO (Group Relative Policy Optimization) — `GRPORouterPolicy` is a stub
- [ ] Preference learning from user feedback
- [ ] Continual fine-tuning on accumulated trajectories
- [ ] Multi-objective optimization: quality vs. latency vs. energy vs. cost

### Memory
- [ ] ConversationMemory — sliding window with automatic summarization of older turns
- [ ] Personal Notes — user-created persistent notes and preferences
- [ ] Episodic Memory — records of past interactions, tool uses, and outcomes
- [ ] Vector DB adapters (Qdrant, ChromaDB) for users with existing infrastructure

### Tools
- [ ] WebSearch tool (Tavily, SearXNG, DuckDuckGo)
- [ ] CodeInterpreter tool (sandboxed Python execution)
- [ ] FileWrite tool (safe file writing with path validation)
- [ ] MCP (Model Context Protocol) compatibility

### Engines
- [ ] SGLang engine backend (structured generation, constrained decoding)
- [ ] MLX engine backend (Apple Silicon native, Metal acceleration)
- [ ] Complete vLLM integration (tensor parallelism config, multi-GPU)

### OpenClaw
- [ ] Full OpenClaw gateway integration (WebSocket, `:18789`)
- [ ] OpenClaw skill composition
- [ ] Context compaction in OpenClaw agent
- [ ] `openjarvis-openclaw` as separate plugin package (currently inline)

### Infrastructure
- [ ] Documentation site (MkDocs or similar)
- [ ] Getting started guide
- [ ] Plugin development guide
- [ ] API reference docs
- [ ] CI/CD pipeline
- [ ] PyPI publishing

---

## Testing Recipes

### Run all tests
```bash
uv sync --extra dev
uv run pytest tests/ -v --tb=short
```

### Run a specific module's tests
```bash
uv run pytest tests/bench/ -v
uv run pytest tests/sdk/ -v
uv run pytest tests/agents/test_openclaw.py -v
```

### Run with optional deps (server)
```bash
uv sync --extra dev --extra server
uv run pytest tests/server/ -v  # No longer skipped
```

### Lint
```bash
uv run ruff check src/ tests/
uv run ruff check src/ tests/ --fix  # Auto-fix
```

### Quick smoke test
```bash
uv run jarvis --version          # 1.0.0
uv run jarvis --help             # All subcommands
python -c "from openjarvis import Jarvis; print(Jarvis)"
```

---

## Adding New Components

### New Benchmark

1. Create `src/openjarvis/bench/my_benchmark.py`:
   ```python
   from openjarvis.bench._stubs import BaseBenchmark, BenchmarkResult

   class MyBenchmark(BaseBenchmark):
       @property
       def name(self) -> str: return "my-bench"
       @property
       def description(self) -> str: return "Description"
       def run(self, engine, model, *, num_samples=10) -> BenchmarkResult: ...

   _registered = False
   def ensure_registered():
       global _registered
       if _registered: return
       from openjarvis.core.registry import BenchmarkRegistry
       if not BenchmarkRegistry.contains("my-bench"):
           BenchmarkRegistry.register_value("my-bench", MyBenchmark)
       _registered = True
   ```
2. Import in `bench/__init__.py` `ensure_registered()`
3. Add test file `tests/bench/test_my_benchmark.py` with autouse fixture calling `ensure_registered()`

### New Tool

1. Create `src/openjarvis/tools/my_tool.py`:
   ```python
   from openjarvis.core.registry import ToolRegistry
   from openjarvis.tools._stubs import BaseTool, ToolSpec

   @ToolRegistry.register("my-tool")
   class MyTool(BaseTool):
       @property
       def spec(self) -> ToolSpec: ...
       def execute(self, input: str, **params) -> str: ...
   ```
2. Import in `tools/__init__.py`
3. Add test file `tests/tools/test_my_tool.py`

### New Memory Backend

1. Create `src/openjarvis/memory/my_backend.py`:
   ```python
   from openjarvis.core.registry import MemoryRegistry
   from openjarvis.memory._stubs import MemoryBackend, RetrievalResult

   @MemoryRegistry.register("my-backend")
   class MyBackend(MemoryBackend):
       def store(self, content, *, source="", metadata=None) -> str: ...
       def retrieve(self, query, top_k=5) -> list[RetrievalResult]: ...
       def delete(self, doc_id) -> bool: ...
       def clear(self) -> None: ...
   ```
2. Import in `memory/__init__.py` with try/except for optional deps
3. Add test file with `pytest.importorskip()` if using optional deps
4. Add optional dep group in `pyproject.toml` if needed

### New Agent

1. Create `src/openjarvis/agents/my_agent.py`:
   ```python
   from openjarvis.agents._stubs import AgentResult, BaseAgent
   from openjarvis.core.registry import AgentRegistry

   @AgentRegistry.register("my-agent")
   class MyAgent(BaseAgent):
       agent_id = "my-agent"
       def __init__(self, engine, model, *, bus=None, **kwargs): ...
       def run(self, input, context=None, **kwargs) -> AgentResult: ...
   ```
2. Import in `agents/__init__.py`
3. Add test file `tests/agents/test_my_agent.py`

### New Engine

1. Create `src/openjarvis/engine/my_engine.py`:
   ```python
   from openjarvis.core.registry import EngineRegistry
   from openjarvis.engine._stubs import InferenceEngine

   @EngineRegistry.register("my-engine")
   class MyEngine(InferenceEngine):
       engine_id = "my-engine"
       def generate(self, messages, *, model, **kwargs) -> dict: ...
       def stream(self, messages, *, model, **kwargs): ...
       def list_models(self) -> list[str]: ...
       def health(self) -> bool: ...
   ```
2. Import in `engine/__init__.py` with try/except
3. Add to `_discovery.py` engine priority list if auto-discoverable

---

## Session Log

### Session 1 (2026-02-16) — Phase 5 Implementation

**Scope:** Full Phase 5 (v1.0) — SDK, OpenClaw, Benchmarks, Docker, Docs

**Work completed:**
- Step 1: Added `BenchmarkRegistry` to `core/registry.py`, updated `conftest.py`
- Step 2: Created `bench/` package — `_stubs.py`, `latency.py`, `throughput.py`, `__init__.py`; CLI `bench_cmd.py`
- Step 3: Created `sdk.py` — `Jarvis` class + `MemoryHandle`; updated `__init__.py` exports
- Step 4: Created OpenClaw infra — `openclaw_protocol.py`, `openclaw_transport.py`, `openclaw_plugin.py`; rewrote `openclaw.py` from stub
- Step 5: Created `Dockerfile`, `Dockerfile.gpu`, `docker-compose.yml`, `deploy/systemd/openjarvis.service`, `deploy/launchd/com.openjarvis.plist`
- Step 6: Version bump to 1.0.0, updated `README.md`, `CLAUDE.md`

**Bugs fixed during implementation:**
1. Ruff lint: 17 issues (E501, I001, F401, F841) — all fixed
2. Registry clearing broke `@register` decorators — switched to `ensure_registered()` for bench modules
3. `ChunkConfig(overlap=...)` should be `ChunkConfig(chunk_overlap=...)` — fixed
4. `chunk.text` should be `chunk.content` — fixed
5. Test content too short for chunking (0 chunks produced) — used 100 words

**Final: 520 passed, 8 skipped, 0 failures, ruff clean**
