# Threat model: prompt injection and money movement

**Claim:** prompt injection can make Resolv's agent say the wrong thing or refuse a valid refund.
It cannot move more money than the policy allows, and it cannot read another customer's order.
This document makes that argument against the actual code and names what is still at risk.

Grounded in: `resolv/agents/loop.py` (the loop), `resolv/harness/tools.py` and `resolv/harness/policy.py` (the only code
that acts), `resolv/harness/ledger.py` and `resolv/harness/audit.py` (the record), `resolv/api/main.py`, `resolv/api/auth.py`,
`resolv/api/escalations.py` (the HTTP surface), and `web/console/` (the approver UI).

## 1. Assets

| Asset | Where it lives | Worst outcome |
|---|---|---|
| Money (refunds) | `refunds` table, `resolv/integrations/payments.py` | refund paid that policy forbids, or paid twice |
| Other customers' order data | `data/db/orders.json` via `resolv/harness/validity.get_order` | one customer reads another's order |
| The system prompt | `resolv/agents/loop.SYSTEM` | leaked verbatim (low value: there are no secrets in it) |
| Audit integrity | `audit` table, keyed hash chain (`AUDIT_HMAC_KEY`) | a refund with no record, or an edited record |
| Provider budget and quota | LLM provider account | one caller exhausts the daily quota or the USD budget |
| Approver attention | `escalations` queue, `/console` | an approver is tricked into approving the wrong thing |

## 2. Actors and trust boundaries

| Actor | Trust | What they control |
|---|---|---|
| Honest customer | authenticated by order proof, untrusted text | the complaint message |
| Malicious customer | same as above | the message, including injection payloads |
| Third-party text (carrier notes, pasted emails) | untrusted, indirect | text quoted inside the message |
| Approver (possibly tired or socially engineered) | holds an approver token | approve/deny on `/escalations` |
| Insider with DB/file access | outside the app's control | `data/resolv.db`, env secrets |
| LLM provider | trusted for availability, not for correctness | every model output, retention of prompts |

```
  customer / third-party text (UNTRUSTED)
        |
        v
  [resolv/api/main.py]  rate limit -> token -> caller_id  (TRUST BOUNDARY 1: identity is decided here, not by text)
        |
        v
  [resolv/agents/loop.py]  extractor LLM (no tools) -> resolver LLM -> picks 1 of 3 tools + JSON args
        |                    ^ untrusted text reaches both LLMs (wrapped in <customer_message>)
        v
  [resolv/harness/tools.py]  case_id + caller_id bound in a closure; args type-checked   (TRUST BOUNDARY 2)
        |
        v
  [resolv/harness/policy.check_refund]  rules 0-7, deterministic, inside one ledger tx   (TRUST BOUNDARY 3)
        |                                   \
        v                                    v
  payments.refund (money)             escalations (action_json + action_hash)
                                              |
                                              v
                              [/escalations + /console] approver token (TRUST BOUNDARY 4)
                              approve re-runs tools.issue_refund -> policy again
```

## 3. Lethal trifecta

An agent is exploitable by injection when it has all three: access to **private data**, exposure to
**untrusted content**, and a way to **act externally or exfiltrate**. Resolv has all three:
order records, customer and carrier text, and refunds plus replies. So the design reduces each leg
rather than trusting the model.

- **Private data is scoped to the caller.** `caller_id` comes from the signed token
  (`resolv/api/auth.verify_token`), never from text. `resolv/agents/loop._bind` closes over `case_id` and
  `caller_id`. The tool schemas (`TOOL_SCHEMAS`) do not include them, so the model cannot name
  another customer. `tools.lookup_order` returns identical text for "missing" and "someone
  else's" orders, so IDs can't be probed. In any session the only private data the model can read
  is the caller's own.
- **External effect goes through a deterministic gate.** The model chooses one of three tools
  (`tools.TOOLS`). The arguments are typed: `order_id` string, `claim_type` enum, `amount_usd`
  number. `NaN`, `Infinity` and `bool` are rejected (`loop._parse_args`, policy rule 0).
  `policy.check_refund` computes eligibility from the **record**, never from the customer's
  stated amount or date. Rules 2 (owner), 3 (claim true against the record), 4 (window), 5 (not
  already refunded) and 6 (cap = fraction of the paid amount) run inside one
  `BEGIN IMMEDIATE` transaction, so they can't be raced. Above `AUTO_APPROVE_MAX_USD`, rule 7
  escalates. This is the action-selector pattern: the LLM selects an action, and code decides
  whether it happens.
- **Exfiltration channels are narrow.** The model's outputs are a plain-text reply and tool
  arguments. The reply goes to the customer who already owns the data. `resolv/harness/guardrails`
  blocks replies that leak the prompt (an 8-word verbatim run), contain PII patterns, or claim a
  payment the trail doesn't show. The console renders all server text with `textContent`, never
  `innerHTML`: no markup, no auto-loading image URLs, no XSS (`web/console/app.js`).

## 4. Rule of Two

The Rule of Two says an agent session should combine at most two of: untrusted input, sensitive
data or systems, and the ability to change state or communicate externally. A Resolv session
combines **untrusted input** with **an external effect** (refunds). The **sensitive-data** leg is
neutralised by caller scoping: the only data reachable is data the caller already owns. The
effect itself is bounded by code the model cannot influence.

The one-line version: *the model can choose to ask for a refund, but only a pure function of the
order record decides whether it is paid, and it can only see the caller's own orders.*

## 5. Dual-LLM / quarantine

- The **extractor** (`resolv/agents/extractor.py`) is a no-tools LLM. It reads the raw message and
  emits a typed `{order_id, claim_type}`. Its output is only a hint: the opening prompt says
  "Intake is often wrong — verify it."
- The **resolver** still reads the raw text, wrapped in `<customer_message>` tags. Typed closing
  tags are stripped (`loop._wrap`), and the system prompt declares the tagged content to be data.
  This is a partial quarantine, not a full one: a determined injection can still steer what the
  resolver *says* or *which tool it calls*. It cannot change what the gate allows.
- A stricter arm (the resolver sees only the extractor's struct, never the raw text) is not
  yet measured.

## 6. STRIDE by component

| Component | Spoofing | Tampering | Repudiation | Info disclosure | DoS | Elevation |
|---|---|---|---|---|---|---|
| API (`resolv/api/main.py`) | token required on `/resolve` | pydantic caps (4000-char message) | every call gets a `case_id` and trail | generic 401/503 text | per-IP rate limits, 90 s wall clock | none: no business logic here |
| Auth (`resolv/api/auth.py`) | HMAC proof per order, HMAC tokens, constant-time compare | signed payload | `jti` per token | unknown order and wrong proof give the same 401 | `/token` 5/min/IP | approver tokens minted only by local CLI, never over HTTP |
| Loop (`resolv/agents/loop.py`) | ids bound in closure | args parsed strictly | invalid calls recorded as denied attempts | prompt-leak guardrail | step, tool-call, token, USD and wall-clock budgets; loop detection | 3 tools only |
| Harness (`policy`, `tools`) | caller must own order (rule 2) | one `BEGIN IMMEDIATE` tx per refund | attempts recorded, denials included | lookup hides foreign orders | n/a | cap from record; rule 7 human gate |
| Ledger (`resolv/harness/ledger.py`) | n/a | keyed hash chain, `verify_chain` | append-only; `clear` refuses outside test/eval | local file | single SQLite writer | insider with file access (residual) |
| Approvals (`resolv/api/escalations.py`, console) | approver token; approver ≠ caller | approval bound to `action_hash`; approve re-runs the gate | `approver_id` on every record | approver sees only escalated cases | 24 h expiry | can only approve the exact stored action |
| Tracing (Langfuse) | keys in env | n/a | trace per run | PII masked before the LLM when `REDACT_BEFORE_LLM=1` (default in prod) | async export | n/a |
| PSP (`resolv/integrations/payments.py`) | n/a | idempotency key per refund | saga states: pending, settled, failed, unknown | n/a | unknown outcomes reconciled | n/a |

## 7. OWASP mapping

The red-team table that maps injection cases (direct, indirect via quoted carrier text, fake tool
results, fake approvals) to OWASP LLM Top 10 and agentic (ASI) categories belongs in
`docs/eval/REDTEAM.md`, generated from `evals/injection.py`. That file does not exist yet. Its
pass rates are not yet measured here.

## 8. Residual risks

- **Wrong denials (availability).** An injection, or plain model error, can talk the agent out
  of a valid refund or into a refusal. The gate stops over-payment, not under-payment. The
  over-block metric measures this. Its current value is not stated here.
- **Approver social engineering (ASI09).** Free text in an escalation (the `reason`, and the
  trail's `result` strings) can contain agent-written or customer-influenced text. Mitigations:
  the console shows the raw bound action and its hash first, labels free text as untrusted, warns
  when the text names a dollar amount that differs from the bound amount, and requires a
  confirmation that repeats the raw amount and order. The server rejects an approver approving
  their own refund. A careless approver can still approve a bad-but-in-policy refund.
- **Review escalations carry no requester.** `loop._bind` calls `tools.escalate_to_human` without
  `caller_id`, so model-initiated review escalations have an empty `requested_by`. This is minor
  (no money is bound to them), but it weakens the record.
- **Provider-side data retention.** Customer text is sent to the LLM provider. Masking before the
  LLM is on by default only in prod.- **Single-node SQLite (availability).** One writer and one file. The in-memory rate limiter and
  dev-mode random secrets do not survive a restart or span processes.
- **Insider with DB access.** The hash chain detects edits only when someone runs
  `verify_chain` against an exported head. It does not prevent edits.
- **Prompt injection is not solved.** Public guidance from both national cyber agencies and model
  vendors treats it as a standing risk to design around, not a bug that is fixed. This design
  assumes the model *will* be steered and limits what steering can buy.
