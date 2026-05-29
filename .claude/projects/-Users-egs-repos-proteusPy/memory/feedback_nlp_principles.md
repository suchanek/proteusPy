---
name: NLP-first, minimize inference
description: Core engineering principle — use local NLP/embedding models directly, avoid round-tripping through inference APIs
type: feedback
---

Prefer loading embedding/NLP models directly (e.g. via sentence-transformers) over calling inference APIs (ollama HTTP, nomic.ai API, OpenAI, etc.).

**Why:** Eliminates network overhead, removes daemon dependencies, runs faster in batch, and keeps compute local. User stated explicitly: "use the power of NLP when possible and minimize inference."

**How to apply:** When embedding, classification, or similarity work is needed, reach for sentence-transformers / HuggingFace first. Only fall back to an inference API if the model genuinely cannot run locally (size, licensing, etc.). Remove ollama/API shims when refactoring existing code.
