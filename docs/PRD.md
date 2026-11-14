# PRD: Resolv, a refund agent that can only act inside a policy harness

## Problem

Refund and late-delivery complaints are high-volume, repetitive, and touch money. Human agents
are slow and inconsistent. A plain LLM agent is fast, but it can be talked into paying what
policy forbids, or into revealing another customer's order. The support lead needs automation
that resolves the routine cases, provably cannot overpay, and puts the rest in front of a person
with the evidence attached.

## Users

| User | What they need |
|---|---|
| **Customer** | a correct answer quickly. If a refund is owed, the money; if not, a plain reason |
| **Support lead** | resolve rate, cost and incidents they can trust; a way to freeze refunds |
| **Approver** | a short queue of large refunds, each showing the exact action, the order record and the trail; one click to decide |

## Jobs to be done

- *When my order was late, never arrived, or was canceled and charged,* I want to explain it once
  in my own words and get what I'm owed, *so I don't have to argue with a form.*
- *When a refund is large or disputed,* I want to see exactly what would be paid and why, *so I
  approve or deny without re-investigating the case.*
- *When something goes wrong,* I want every attempt (including the refused ones) on a tamper-evident
  record, *so I can explain any refund to finance or an auditor.*

## Success metrics

| Metric | Definition | Target | Current |
|---|---|---|---|
| Resolve rate | share of owed claims paid correctly, with a confidence interval, over k trials | set after the first measurement | see `docs/eval/METRICS.md`; not restated here |
| Unauthorized refunds | refunds the policy forbids (over cap, wrong owner, outside window, duplicate) | **0** | must stay 0 |
| Over-block $ withheld | owed dollars refused or escalated when they should have been paid | as low as possible; tracked | not yet measured |
| Approval decided ≤ 24 h | share of approval escalations decided before expiry | 100% | not yet measured |
| Cost per resolution | USD per `/resolve` (`config.price_usd`) | tracked per provider | see `evals/cost.py` output; not restated here |
| First-action p95 | time to the first tool call | ≤ 6 s | not yet measured on the deploy |

## Scope

- One HTTP API: `/token` (order proof → session token), `/resolve`, and `/escalations` for
  approvers.
- Three tools: `lookup_order`, `issue_refund` (policy-gated, saga to the payment provider), and
  `escalate_to_human`.
- A deterministic policy: claim must be true against the record, within the window, not already
  refunded, under the cap; a human must approve above `AUTO_APPROVE_MAX_USD`.
- The approvals console (`/console`) and an append-only, hash-chained audit trail.

## Non-goals

- **RAG / knowledge base.** Answers come from the order record and the policy, not from documents.
- **Multi-agent orchestration.** One loop, three tools.
- **A real payments integration.** The payment provider is a stub with real saga semantics
  (pending, settled, failed, unknown, reconcile).
- **Multi-tenant.** One retailer, one policy, one database.

## Risks

| Risk | Mitigation | Residual |
|---|---|---|
| Prompt injection moves money | deterministic gate over typed args; caller scoping; see `docs/THREAT_MODEL.md` | wrong denials remain possible |
| Approver approves a bad action | console shows the raw bound action and hash; mismatch banner; approver ≠ caller | a careless approver |
| Provider outage or quota | circuit breaker, generic 503, provider switch | no automatic failover |
| Model drift changes behaviour | regression set and drift checks in `evals/` | detection, not prevention |
| Single SQLite node | backups and a chain head per backup (`docs/RUNBOOK.md`) | downtime on host loss |

## Rollout plan

1. **Shadow mode.** The agent runs on real complaints, but tool calls are recorded and not
   executed. Humans resolve as today. A shadow switch does not exist in the code yet. Compare the agent's decisions with the humans'. Exit when
   unauthorized = 0 and the decision agreement is acceptable to the support lead (threshold set
   before the run starts).
2. **Assisted.** Every refund is escalated: X = $0, so a human approves each one in the console.
   This measures approver minutes per decision and the share of agent proposals that approvers
   accept.
3. **Auto under $X.** Raise X to the owner's chosen threshold ($200 unless the
   measured load says otherwise). Keep the kill switch and the 24 h approval expiry. Review the
   metrics above weekly.

Rolling back from any step means lowering X (or freezing refunds). It needs no code change.
