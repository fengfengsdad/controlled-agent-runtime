# Architecture

## Purpose

Controlled Agent Runtime turns an engineering-change requirement into a structured delivery plan with retrieval grounding, tool context, human approval, and an auditable control plane.

## Component Boundaries

```
Client
  │
  ▼
FastAPI (OpenAPI)
  │
  ▼
RuntimeService (orchestration owner)
  ├─ LangGraph: start → retrieve → tool → context → plan → groundedness → approval_gate
  ├─ HybridRetriever (filter → recall → fuse → rerank → diversity)
  │    ├─ VectorStore (chunk/embed/cosine)
  │    └─ Bm25Index (in-process sparse index)
  ├─ Context assembler (token budget, [S1] labels, drop counts)
  ├─ Groundedness check (inline citations, coverage, insufficient-evidence refusal)
  ├─ LLMClient (stub | openai_compatible)    ← model provider interface
  ├─ RequirementLookupTool via MCP stdio   ← real client/server JSON-RPC session
  ├─ CheckpointStore (SQLite)                ← restart recovery + idempotency
  └─ AuditStore (SQLite)                     ← evidence chain
```

## Retrieval pipeline

```
query
  → structured pre-filter (source / doc_type / classification / owner)
  → parallel recall at candidate_k
       ├─ dense: cosine over normalised embeddings
       └─ sparse: BM25 Okapi over the same chunk text
  → reciprocal rank fusion (rank-based, no score normalisation)
  → near-duplicate suppression (shingle Jaccard)
  → rerank (stub | llm | cross_encoder)
  → MMR selection with per-source cap
  → top_k citations carrying per-stage ranks and scores
  → token-budgeted context assembly with [S1]…[Sn] labels
  → generation constrained to those labels
  → post-generation groundedness check (coverage, hallucinated-label drop, refuse if below floor)
```

`RETRIEVAL_MODE` selects `hybrid` (default), `vector`, or `bm25`, and the rerank
stage is switchable per query. The single-retriever and no-rerank configurations
exist so each stage's contribution can be measured against a baseline instead of
asserted; `POST /v1/retrieval/search` runs the pipeline without generation for
inspection.

Every citation reports `fusion_score`/`fusion_rank` alongside
`rerank_score`/`rerank_rank`, so `score` (the final ordering value) never hides
what the previous stage thought.

### Reranker trade-offs

| Provider | Latency | Cost | Data residency | Use |
|----------|---------|------|----------------|-----|
| `stub` | negligible | none | in-process | CI and regression runs; deterministic lexical proxy, not a quality claim |
| `cross_encoder` | tens of ms, predictable | fixed compute | in-process | preferred when the corpus is regulated |
| `llm` | hundreds of ms | per token | leaves the process | intent-heavy queries where listwise judgement wins |

Design notes:

- **Filtering happens before scoring.** Once an out-of-scope chunk reaches the
  context window the model can leak its content even if the citation is stripped
  from the response, so access decisions cannot be a post-filter.
- **Fusion is rank-based.** BM25 scores and cosine similarities are not on a
  comparable scale, so a weighted blend needs a per-corpus alpha that drifts.
  The trade-off is losing score magnitude.
- **The sparse index is rebuilt in-process** from the chunk text already in the
  vector index, keyed off an index signature, so a reindex cannot leave a stale
  second artefact behind. Beyond MVP corpus sizes this belongs in OpenSearch.
- **Duplicates are removed before reranking,** not after. Overlapping chunk
  windows mean the same sentences legitimately appear in several chunks, and
  scoring the same passage twice wastes the rerank budget.
- **Rerank failures degrade to the fusion order** and are recorded in the trace.
  A flaky rerank call should cost relevance, not availability.
- **Diversity is enforced at selection time** via MMR plus a per-source cap, and
  the number of candidates the cap blocked is reported, so the recall/diversity
  trade-off is visible rather than silent.
- **Context is packed against an explicit token budget.** Leftover evidence is
  counted, not dropped silently, and the last included chunk is truncated on a
  sentence boundary. Labels `[S1]`…`[Sn]` are assigned here so generation and
  the later groundedness check share one mapping.
- **Unsupported answers are refused.** If the best retrieval score is below
  `RELEVANCE_FLOOR`, generation is skipped. After generation, hallucinated
  `[Sn]` labels are stripped and `citation_coverage` below `COVERAGE_FLOOR`
  returns `insufficient_evidence` rather than a guessed plan.
- `GET /v1/workflows/{id}/evidence` replays retrieval, packing, and per-claim
  support so a "why this answer" question is reconstructable from the audit
  trail.
- **Known gap:** the tokenizer has no subword matching, so `idempotency` does not
  lexically match `webhook_idempotency_ledger`. Dense recall covers this, which
  is part of why hybrid is the default. Asserted in `tests/test_retrieval.py`.
- **Known gap:** the `stub` reranker is a lexical proxy so that CI stays offline
  and deterministic. It is not a substitute for a cross-encoder in production and
  should not be read as one in evaluation numbers.

## MCP stdio tool path

1. Runtime spawns `python -m agent_runtime.tools.mcp_requirement_server`.
2. Client performs MCP handshake over newline-delimited JSON-RPC stdio.
3. Client calls `requirement_lookup` with `change_id`.
4. Server returns structured catalog data; audit records `transport=mcp-stdio`.

This keeps tool execution outside the API process boundary while remaining synthetic/read-only.

## Anti lock-in

- Orchestration depends on stable interfaces (`LLMClient`, `Embedder`, tool allowlist), not a single vendor SDK.
- Default local mode uses deterministic stubs so CI and demos do not require paid APIs.
- Managed model/embedding endpoints are adapters behind the same contracts.

## Data flow

1. Validate and redact inbound requirement.
2. Emit `workflow_started`.
3. Retrieve top-k chunks through the hybrid pipeline; emit `retrieval_completed`
   with the mode, corpus size, count filtered out, per-retriever candidate
   counts, and the selected chunks with their per-retriever ranks.
4. Invoke read-only requirement tool; emit `tool_invoked`.
5. Assemble labelled context under the token budget; emit `context_assembled`
   with tokens used, dropped counts, and `[Sn]` labels. If max retrieval score
   is below the relevance floor, emit `groundedness_checked` + `workflow_completed`
   with status `insufficient_evidence` and skip generation.
6. Generate plan JSON constrained to those labels; emit `plan_generated`.
7. Verify inline citations, drop hallucinated labels, compute coverage; emit
   `groundedness_checked`. Coverage below the floor completes as
   `insufficient_evidence` rather than an unsupported plan.
8. Gate writes via approval; emit `approval_resolved` + `workflow_completed`.

## Trust boundaries

| Zone | Trust | Controls |
|------|-------|----------|
| Public API | Untrusted input | size limit, injection reject, redaction |
| Orchestrator | Trusted code | bounded graph, typed state |
| Tools | Allowlisted only | no arbitrary tool execution |
| Model provider | External | prompts versioned; secrets redacted before send |
| Persistence | Local disk/SQLite | workflow-scoped audit + checkpoints |
