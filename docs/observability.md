# Span schema

One vendor-neutral span schema (OpenTelemetry names), sent to Langfuse when its keys are set and a no-op
otherwise.

## Semantic conventions

- OpenTelemetry GenAI semantic conventions from `open-telemetry/semantic-conventions-genai`. That repo has no
  tagged release and every `gen_ai.*` attribute is "Development", so pin it by commit SHA: **not yet recorded**.
  Re-check on every SDK upgrade.
- `gen_ai.system` is deprecated; use `gen_ai.provider.name`.
- Code uses only the OpenTelemetry API (`resolv/observability.py`). The Langfuse Python SDK 4.x is the exporter:
  constructing its client registers a span processor on the global TracerProvider. No keys, no provider, and
  every span is a no-op with zero network calls.

## Span tree

```
invoke_agent resolv                      (one per resolution, root)
├── chat <extractor model>               (json / legacy_line extractor; the adk backend emits ADK's own spans here)
├── chat <agent model>                   (one per loop step)
├── execute_tool lookup_order
├── chat <agent model>
├── execute_tool issue_refund
└── chat <agent model>                   (final plain-text reply)
```

| Span name | Kind | Required attributes | Resolv-specific attributes |
|---|---|---|---|
| `invoke_agent resolv` | INTERNAL | `gen_ai.operation.name=invoke_agent`, `gen_ai.agent.name=resolv`, `gen_ai.conversation.id=<case_id>` | `resolv.caller.hash` (sha256 of caller_id, first 12 hex), `resolv.outcome` (resolved / escalated / denied), `resolv.steps`, `resolv.usd`, `resolv.cache_hit_ratio`, `resolv.budget_exceeded` |
| `chat <model>` | CLIENT | `gen_ai.operation.name=chat`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model` (served), `gen_ai.request.temperature`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.response.finish_reasons` | `resolv.role` (extractor / agent / simulator / judge), `resolv.cached_input_tokens` (absent when the provider doesn't report it), `resolv.usd`, `resolv.prompt.sha` |
| `execute_tool <tool>` | INTERNAL | `gen_ai.operation.name=execute_tool`, `gen_ai.tool.name`, `gen_ai.tool.call.id` | `resolv.policy.action`, `resolv.policy.rule_id`, `resolv.audit.record_id`, `resolv.audit.seq`, `resolv.audit.mac` (when the audit record carries them), `resolv.order.id` |

## Langfuse mapping

Set alongside the schema attributes so Langfuse can file traces and chart cost (attribute names to verify
against the Langfuse OTel docs on each upgrade):

| Where | Attribute | Value |
|---|---|---|
| root | `langfuse.session.id` | case_id |
| root | `langfuse.user.id` | caller hash |
| root | `langfuse.environment` | `RESOLV_ENV` |
| root | `langfuse.release` | `GIT_SHA` |
| root | `langfuse.trace.tags` | `prompt:<sha>` |
| chat | `langfuse.observation.type` | `generation` |
| chat | `langfuse.observation.model.name` | requested model |
| chat | `langfuse.observation.usage_details` / `cost_details` | JSON of tokens / USD |
| chat | `langfuse.observation.prompt.name` / `.version` | linked prompt (version pushed by `observability.init()`) |

Scores on the trace (after `evals/monitor.record`): `unauthorized`, `reply_grounded`, `looked_up_first`
(BOOLEAN) and `steps` (NUMERIC).

## Metrics derived from spans

- `cost_per_resolution` = sum of `resolv.usd` over the case's chat spans
- `step_latency_ms` = chat span duration; `tool_latency_ms` = execute_tool span duration
- `cache_hit_ratio` = sum of `resolv.cached_input_tokens` / sum of `gen_ai.usage.input_tokens`

## Content capture

Off by default. `gen_ai.input.messages` is set only when `RESOLV_TRACE_CONTENT=1`, and always passes through
`resolv/harness/pii.py` masking (emails, phones, card numbers, simple street addresses). The Langfuse client also gets
the same masking function as its `mask=` hook. The standard `OTEL_INSTRUMENTATION_GENAI_CAPTURE_*` variable name
is not settled upstream, so Resolv uses its own flag. Eval sweeps are untraced unless `RESOLV_TRACE_EVAL=1`.

## Backend notes

Not yet written: what Langfuse made easy or hard compared with the other two backends.
