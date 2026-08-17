# Agent Runtime Runbook Notes

## Restart Recovery
Workflow state is checkpointed to SQLite after each graph node. After process restart, clients can resume by workflow_id.

## Prompt Injection
Reject payloads matching known injection patterns before orchestration begins.

## Observability
OpenTelemetry spans are exported to console in local mode. Prefer OTLP exporter in shared environments.
