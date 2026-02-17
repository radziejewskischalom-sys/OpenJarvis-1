# OpenJarvis

**Your AI stack, your rules.**

A modular, pluggable AI assistant backend. Compose your own stack across five pillars — Intelligence, Learning, Memory, Agents, and Inference — then swap any piece without touching the rest.

> **Status: v1.0** — All five pillars implemented. SDK, benchmarks, OpenClaw infrastructure, and Docker deployment ready.

## What is this?

OpenJarvis lets you build a personal AI assistant from composable parts:

- **Intelligence** — multi-model management with automatic routing (Qwen3, GPT OSS, Kimi-K2.5, Claude, GPT-5, Gemini)
- **Memory** — persistent, searchable storage with multiple backends (SQLite, FAISS, ColBERTv2, BM25, hybrid)
- **Agents** — pluggable reasoning and tool use (OpenClaw Pi agent, simple, orchestrator, custom)
- **Inference** — hardware-aware engine selection (vLLM, SGLang, Ollama, llama.cpp, MLX)
- **Learning** — router that improves over time (heuristic now, learned later)

## Quick Start — Python SDK

```python
from openjarvis import Jarvis

j = Jarvis()
response = j.ask("What is the meaning of life?")
print(response)

# With a specific model and agent
response = j.ask("Explain gravity", model="qwen3:8b", agent="orchestrator")

# Memory operations
j.memory.index("./docs/")
results = j.memory.search("machine learning")

j.close()
```

## Quick Start — CLI

```bash
jarvis ask "Hello, what can you do?"
jarvis ask --agent orchestrator --tools calculator,think "What is 2+2?"
jarvis bench run -n 5 --json
jarvis model list
jarvis memory index ./docs/
jarvis serve --port 8000
```

## Docker

```bash
docker compose up -d          # Starts Jarvis + Ollama
curl http://localhost:8000/health
```

## Documentation

- **[VISION.md](VISION.md)** — Project vision, architecture, design principles
- **[ROADMAP.md](ROADMAP.md)** — Phased development plan with deliverables
- **[CLAUDE.md](CLAUDE.md)** — Developer reference for working with the codebase

## Quick orientation

```
src/openjarvis/
├── core/          # Registry, types, config, event bus
├── intelligence/  # Model management, routing
├── memory/        # Storage backends (SQLite, FAISS, ColBERT, BM25, hybrid)
├── agents/        # Agent implementations + tool system + OpenClaw
├── engine/        # Inference engine wrappers
├── learning/      # Router policy (heuristic, GRPO stub)
├── bench/         # Benchmarking framework (latency, throughput)
├── telemetry/     # Telemetry store + aggregator
├── server/        # OpenAI-compatible API server
├── cli/           # CLI entry points
└── sdk.py         # Python SDK (Jarvis class)
```

## Requirements

- Python 3.10+
- An inference backend: [Ollama](https://ollama.com), [vLLM](https://github.com/vllm-project/vllm), or [llama.cpp](https://github.com/ggerganov/llama.cpp)
- Node.js 22+ (only if using OpenClaw agent)

## License

TBD
