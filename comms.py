"""Email / SMS / Slack send tools — moved from mcp/comms_server.py.

Lives at the top level, not under harness/, because it's a side-effecting action
(send a message), not a decision. harness/ is where Resolv decides what to do;
this is where it actually does it. SEND_MODE=draft (the default) always logs to an
in-memory store instead of sending, so the whole pipeline is safe to run end-to-end
with no outbound side effects. Real SendGrid/Twilio/Slack calls slot in behind these
same signatures when SEND_MODE=live, so callers never change.
"""
from config import SEND_MODE

_DRAFT_LOG: list[dict] = []


async def send_email(to: str, subject: str, body: str, from_name: str = "Resolv Supply Chain") -> dict:
    """Sends email. In draft mode (default), logs only and does not send."""
    if SEND_MODE != "live":
        _DRAFT_LOG.append({"to": to, "subject": subject, "body": body})
        return {"sent": False, "mode": "draft"}
    raise NotImplementedError("Live email sending (SendGrid) not wired yet — runs in draft mode.")


async def send_sms(to: str, message: str) -> dict:
    if SEND_MODE != "live":
        _DRAFT_LOG.append({"to": to, "sms": message})
        return {"sent": False, "mode": "draft"}
    raise NotImplementedError("Live SMS sending (Twilio) not wired yet — runs in draft mode.")


async def post_slack(channel: str, message: str, blocks: list | None = None) -> dict:
    """Draft mode: logs the message locally instead of posting. Swap in the real
    Slack SDK call here behind the same signature to go live."""
    _DRAFT_LOG.append({"channel": channel, "message": message})
    return {"ok": True, "mode": "draft"}


def get_draft_log() -> list[dict]:
    """Test/debug helper: everything that would have been sent in live mode."""
    return list(_DRAFT_LOG)
