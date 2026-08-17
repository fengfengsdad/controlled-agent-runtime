# Runbook

## Local start

```bash
source .venv/bin/activate
uvicorn agent_runtime.api.main:app --port 8080
curl -s localhost:8080/health
```

## Reindex corpus after editing `data/corpus/`

```bash
curl -s -X POST localhost:8080/v1/admin/reindex
```

## Approval path

1. Start workflow with `"auto_approve": false`
2. Status becomes `awaiting_approval`
3. Approve:

```bash
curl -s -X POST localhost:8080/v1/workflows/{id}/approval \
  -H 'Content-Type: application/json' \
  -d '{"approved": true, "reviewer": "yiyi"}'
```

## Inspect audit chain and evidence

```bash
curl -s localhost:8080/v1/workflows/{id}/audit | python -m json.tool
curl -s localhost:8080/v1/workflows/{id}/evidence | python -m json.tool
```

`/evidence` replays what was retrieved, what the token budget dropped, which
`[Sn]` labels were assigned, and which chunk supports each claim — or why the
run refused with `insufficient_evidence`.

## Restart recovery

Checkpoints are written after each graph node to `CHECKPOINT_DB`.
After process restart, `GET /v1/workflows/{id}` returns the last persisted state.

## Common failures

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| 400 prompt-injection | adversarial wording | rewrite requirement |
| plan empty / 500 with openai provider | missing/invalid API key | check `.env` |
| no citations | empty corpus / not indexed | add docs + `/v1/admin/reindex` |
| `insufficient_evidence` | retrieval score or citation coverage below floor | inspect `/evidence`; lower floors only if the miss is a packing bug |
| duplicate business effect feared | missing idempotency key | pass stable `idempotency_key` |

## Docker

```bash
docker compose up --build
```

Healthcheck probes `/health`.
