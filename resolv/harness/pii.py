"""
# resolv/harness/pii.py

**What:** regex masking for emails, phone numbers, card numbers and simple street addresses.
**Why:** customer text must not reach traces or logs with PII in it. Regex keeps it dependency-free and
deterministic, and order ids (ORD-1234) and dollar amounts never match. It will miss PII a trained recognizer
would catch (names, free-form addresses, non-US phone layouts); that limit is accepted, not hidden.
**Used by:** resolv/observability.py (span content and the Langfuse mask), resolv/harness/guardrails.py, resolv/agents/loop.py.
"""
import re

EMAIL = re.compile(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]+")
# 13-19 digits with optional space/dash separators: card length, longer than any order id or amount.
CARD = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
# North-American style 10 digits with an optional country code; not preceded by $ or a digit, so money is safe.
PHONE = re.compile(r"(?<![\d$.])(?:\+\d{1,3}[ .-]?)?\(?\b\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b")
# "221 Baker Street": house number, one to three capitalised words, a street suffix. Case-sensitive on purpose.
ADDRESS = re.compile(
    r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl)\b"
)

# Order matters: cards before phones, so a card number is never half-masked as a phone.
_PATTERNS = [("EMAIL", EMAIL), ("CARD", CARD), ("PHONE", PHONE), ("ADDRESS", ADDRESS)]


# Replace every match with its kind in brackets, e.g. "[EMAIL]".
def mask(text: str) -> str:
    out = text or ""
    for kind, pattern in _PATTERNS:
        out = pattern.sub(f"[{kind}]", out)
    return out


# The kinds of PII found in the text, in pattern order, e.g. ["EMAIL", "PHONE"].
def find(text: str) -> list[str]:
    return [kind for kind, pattern in _PATTERNS if pattern.search(text or "")]


# Mask every string inside nested dicts/lists/tuples; other values pass through unchanged.
def mask_any(data):
    if isinstance(data, str):
        return mask(data)
    if isinstance(data, dict):
        return {k: mask_any(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [mask_any(v) for v in data]
    return data
