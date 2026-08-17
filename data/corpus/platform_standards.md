# Engineering Change Delivery Standards

## Idempotency
All mutating APIs must accept an idempotency key. Repeated calls with the same key must not create duplicate side effects.

## Approval Gate
Write tools require human approval before execution. Read-only tools may run without approval.

## Audit
Emit structured audit events for: workflow start, retrieval, planning, tool invocation, approval, and completion.

## Payment Changes
Payment journey changes require regression coverage for fraud, timeout, and retry paths before rollout.
