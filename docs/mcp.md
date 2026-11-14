<!--
# docs/mcp.md - how to use Resolv's read-only MCP server

- What the three tools do, why there is no refund tool, and setup for Claude Desktop and Claude Code.
- For operators and reviewers who want to ask an MCP client about orders, policy and the approval queue.
-->

# Read-only MCP server

`resolv/mcp_server.py` exposes three tools over stdio with the `mcp` Python SDK (`MCPServer`):

| Tool | Arguments | Returns |
|---|---|---|
| `lookup_order` | `order_id` | the order record (amount paid, promised and delivered dates, status), or `{"error": "unknown_order"}` |
| `check_refund` | `order_id`, `claim_type`, `amount_usd`, optional `caller_id` | the policy verdict: `action` (allow / deny / escalate), `rule_id`, `reason` |
| `list_escalations` | `status` (default `pending`), `limit` (default 50, max 200) | the approval queue, as the `/console` list shows it |

`check_refund` is a dry run of `resolv/harness/policy.py` `check_refund`, the same function every real refund
passes through. It reads the order's live refunds, so a second refund is denied (`already_refunded`) just as it
would be for real. `caller_id` defaults to the order's owner, as if the customer had asked.

**Read-only by design.** There is no `issue_refund` tool. Money moves only through `resolv/harness/tools.py`,
behind the agent loop and the approvals console, where every call is recorded in the audit chain. The server
writes no records. The kill switch and shadow mode (`docs/RUNBOOK.md`) act on real refunds, not on this
verdict.

## Run it

From the repo root, with the project's dependencies installed (`pip install -r requirements.txt`):

```bash
RESOLV_ENV=dev python -m resolv.mcp_server   # speaks MCP on stdin/stdout; no LLM key needed
```

It reads the order data in `data/db/` and the ledger at `RESOLV_DB` (default `data/resolv.db`, relative to the
working directory).

## Claude Code

From the repo root:

```bash
claude mcp add resolv --env RESOLV_ENV=dev -- "$PWD/venv/bin/python" -m resolv.mcp_server
```

Then ask, for example: "Can ORD-1000 get a $50 late-delivery refund?" or "What is waiting for approval?"

## Claude Desktop

Add this to `claude_desktop_config.json` (Settings, Developer, Edit Config) and restart Claude Desktop. Claude
Desktop does not start in the repo folder, so give absolute paths:

```json
{
  "mcpServers": {
    "resolv": {
      "command": "/absolute/path/to/Resolv/venv/bin/python",
      "args": ["-m", "resolv.mcp_server"],
      "env": {
        "RESOLV_ENV": "dev",
        "PYTHONPATH": "/absolute/path/to/Resolv",
        "RESOLV_DB": "/absolute/path/to/Resolv/data/resolv.db"
      }
    }
  }
}
```

## Tests

`tests/test_mcp_server.py` calls every tool through the SDK's in-process client (no network, no key) and checks
that `check_refund` on a payable refund leaves zero refund rows.
