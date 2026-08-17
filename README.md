# Controlled Agent Runtime

Engineering-change delivery agent runtime.

把合成的工程变更需求转为结构化交付计划，覆盖：

- LangGraph 编排（retrieve → tool → plan → approval）
- **混合检索**：BM25 稀疏召回 + 向量稠密召回 → RRF 融合，支持结构化预过滤（source / doc_type / classification / owner）
- **多阶段重排**：近重复抑制 → 重排（`stub` 确定性离线 / `llm` listwise / `cross_encoder` 本地 bge）→ MMR 多样性 + 单文档上限
- RAG（本地语料 chunk + embedding + 引用，引用附带每个阶段的排名与分数）
- LLM 可切换：`stub`（默认，可复现） / `openai_compatible`
- SQLite checkpoint（重启可恢复）+ 六段审计链
- **真实 MCP stdio client/server**：只读 `requirement_lookup` 工具经子进程 JSON-RPC 会话调用
- 安全控制：payload 限制、prompt-injection 拒绝、secret redaction、写操作审批
- FastAPI OpenAPI、Docker、离线评测 harness

## 环境要求

- Python 3.9+（推荐 3.11+；本机若只有 3.9 也可跑）
- 可选：Docker / OpenAI-compatible API Key

```bash
git clone https://github.com/fengfengsdad/controlled-agent-runtime.git
cd controlled-agent-runtime
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env

# 索引语料并启动 API
python -c "from agent_runtime.graph.runtime import get_runtime; print(get_runtime().reindex_corpus())"
uvicorn agent_runtime.api.main:app --host 0.0.0.0 --port 8080
```

打开 http://127.0.0.1:8080/docs

### 发起一次 workflow

```bash
curl -s http://127.0.0.1:8080/v1/workflows \
  -H 'Content-Type: application/json' \
  -d '{
    "requirement": "Add idempotent retry handling for payment webhook failures.",
    "change_id": "CHG-1001",
    "auto_approve": true
  }' | python -m json.tool
```

### MCP stdio tool

默认 `TOOL_TRANSPORT=mcp`：Runtime 会拉起独立 MCP server 子进程，经 stdin/stdout 完成：

`initialize` → `notifications/initialized` → `tools/list` → `tools/call(requirement_lookup)`

```bash
# 也可单独启动 server（调试用）
python -m agent_runtime.tools.mcp_requirement_server
```

若只需本地直读目录（不走 MCP），设置：

```env
TOOL_TRANSPORT=local
```

### 单独调检索（不走生成）

```bash
curl -s http://127.0.0.1:8080/v1/retrieval/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "how should idempotency and approval gates work together",
    "mode": "hybrid",
    "filters": {"classifications": ["internal"]}
  }' | python -m json.tool
```

`mode` 可取 `hybrid`（默认）/ `vector` / `bm25`，`rerank` 可传 `false` 关掉重排做基线对比。

返回的 `trace` 记录每个阶段做了什么：语料规模、预过滤掉多少、每路检索器候选数、RRF 参数、
去重掉多少、用了哪个 reranker、重排了多少条、被单文档上限挡掉多少、MMR 的 λ。
每条 citation 同时带 `vector_rank` / `bm25_rank` / `fusion_rank` / `rerank_rank` 和各阶段分数——
`score` 是最终排序值，但 `fusion_score` 始终保留重排前的判断，所以重排的效果是可度量的。

### Context 预算与接地生成

生成前会按 token 预算把选中的 chunk 打上 `[S1]`…`[Sn]` 标签；装不下的 chunk 会计入 `dropped_count` 并写入审计，最后一条按句边界截断而不是硬切。生成后会校验这些标签：幻觉引用会被剔除，`citation_coverage` 或检索分数低于阈值时返回 `insufficient_evidence`，而不是给出没有证据的答案。

```bash
curl -s http://127.0.0.1:8080/v1/workflows/{id}/evidence | python -m json.tool
```

### 本地 cross-encoder 重排（可选）

```bash
pip install -e ".[rerank]"   # 拉入 sentence-transformers
RERANKER_PROVIDER=cross_encoder uvicorn agent_runtime.api.main:app
```

依赖缺失时会自动降级到确定性的 `stub` reranker，不会让检索不可用。

## 测试与评测

```bash
pytest -q
python -m agent_runtime.eval_cli                            # 默认强制离线 stub，不受 .env 影响
python -m agent_runtime.eval_cli --mode vector --no-rerank  # 基线：单路召回、不重排
python -m agent_runtime.eval_cli --mode hybrid --rerank     # 完整管道
python -m agent_runtime.eval_cli --live                     # 使用 .env 中配置的真实 provider
```

评测默认走离线 stub，避免本地 `.env` 里指向付费模型的配置把回归套件变成一次计费的网络测试。

### Docker

```bash
docker compose up --build
```

## 切换真实 LLM

编辑 `.env`：

```env
LLM_PROVIDER=openai_compatible
EMBEDDING_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
PROMPT_VERSION=v1
```

兼容任何 OpenAI-compatible 网关（含部分国内代理）。

## 架构与运维

- [docs/architecture.md](docs/architecture.md)
- [docs/runbook.md](docs/runbook.md)
