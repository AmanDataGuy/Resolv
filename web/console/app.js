// web/console/app.js
//
// What: the console's logic: sign in with an approver token, poll the pending queue, show one escalation,
// approve (with an Idempotency-Key) or deny it.
// Why: plain fetch + DOM, no framework. Every server string goes through textContent, never innerHTML, so agent or
// customer text can't inject markup or exfiltrate via images/links. The token lives in sessionStorage (cleared on
// tab close); an XSS would expose it, which is why nothing here renders HTML.
// Used by: web/console/index.html.

const POLL_MS = 10000;
const $ = (id) => document.getElementById(id);
let current = null;      // the escalation detail on screen
let approveKey = null;   // {escId, key}: one Idempotency-Key per approve intent, reused on a network retry
let pollTimer = null;

// The stored approver token, or "".
function token() {
  try { return sessionStorage.getItem("resolv_token") || ""; } catch { return ""; }
}

// Call the API with the bearer token; returns parsed JSON or throws an Error with the server's detail.
async function api(path, options = {}) {
  const headers = { "Authorization": "Bearer " + token(), ...(options.headers || {}) };
  if (options.body) headers["Content-Type"] = "application/json";
  const resp = await fetch(path, { ...options, headers });
  const data = await resp.json().catch(() => ({}));
  if (resp.status === 401 || resp.status === 403) signOut(data.detail || "Not an approver token.");
  if (!resp.ok) {
    const err = new Error(typeof data.detail === "string" ? data.detail : "HTTP " + resp.status);
    err.status = resp.status;
    throw err;
  }
  return data;
}

// Make an element with text content (never HTML).
function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}

// Show or clear the error banner.
function showError(message) {
  $("error").textContent = message || "";
  $("error").hidden = !message;
}

// Show exactly one of the three views.
function show(view) {
  for (const id of ["login", "queue", "detail"]) $(id).hidden = id !== view;
  $("logout").hidden = view === "login";
}

// Human-readable span between now and an ISO time ("3h 4m").
function span(iso, sign) {
  let s = Math.max(0, Math.round(sign * (Date.parse(iso) - Date.now()) / 1000));
  const h = Math.floor(s / 3600);
  s -= h * 3600;
  return h ? `${h}h ${Math.floor(s / 60)}m` : `${Math.floor(s / 60)}m`;
}

// Format a dollar amount, or "-" when the escalation carries no action.
function money(v) {
  return v === null || v === undefined ? "-" : "$" + Number(v).toFixed(2);
}

// Fetch and render the pending queue.
async function loadQueue() {
  try {
    const items = await api("/escalations?status=pending");
    const rows = $("rows");
    rows.replaceChildren();
    for (const e of items) {
      const tr = el("tr");
      const open = el("button", e.id, "link");
      open.addEventListener("click", () => openDetail(e.id));
      const first = el("td");
      first.append(open);
      tr.append(first);
      for (const v of [e.order_id, e.kind, money(e.amount), e.reason, span(e.created_at, -1), span(e.expires_at, 1)]) {
        tr.append(el("td", v));
      }
      rows.append(tr);
    }
    $("empty").hidden = items.length > 0;
    $("updated").textContent = "updated " + new Date().toLocaleTimeString();
    showError("");
  } catch (err) {
    showError("Could not load the queue: " + err.message);
  }
}

// Dollar amounts mentioned in a piece of text.
function amountsIn(text) {
  return [...String(text || "").matchAll(/\$\s?(\d[\d,]*(?:\.\d{1,2})?)/g)].map((m) => Number(m[1].replace(/,/g, "")));
}

// Render the audit trail table.
function renderTrail(trail) {
  $("d-trail").replaceChildren(...(trail || []).map((r, i) => {
    const tr = el("tr");
    for (const v of [i + 1, r.tool, r.order_id, r.rule_id, r.ok ? "yes" : "no", r.caller_id, r.approver_id || "", r.result]) {
      tr.append(el("td", v));
    }
    return tr;
  }));
}

// Open one escalation: raw action, order facts, policy reasons, untrusted text, trail.
async function openDetail(escId) {
  stopPolling();
  showError("");
  try {
    current = await api("/escalations/" + encodeURIComponent(escId));
  } catch (err) {
    showError("Could not load " + escId + ": " + err.message);
    startPolling();
    return;
  }
  const esc = current.escalation;
  const action = current.action_json ? JSON.parse(current.action_json) : null;
  $("d-title").textContent = `${esc.id} - ${esc.order_id || "no order"} (${esc.kind}, ${esc.status})`;
  $("d-action").textContent = action ? JSON.stringify(action, null, 2) : "No bound action (review escalation).";
  $("d-hash").textContent = current.action_hash || "-";
  $("d-order").textContent = JSON.stringify(current.order || {}, null, 2);
  $("d-reason").textContent = esc.reason || "";

  $("d-policy").replaceChildren(...(current.policy_explanation || []).map((p) => el("li", `${p.rule_id}: ${p.reason}`)));
  renderTrail(current.trail);

  // ASI09: if the free text names an amount that is not the bound amount, warn the approver.
  const bound = action ? action.amount_cents / 100 : null;
  const odd = bound === null ? [] : amountsIn(esc.reason).filter((a) => Math.abs(a - bound) >= 0.005);
  const mismatch = $("mismatch");
  mismatch.hidden = odd.length === 0;
  mismatch.textContent = odd.length
    ? `The text mentions ${odd.map(money).join(", ")} but the bound action is ${money(bound)}. Decide on the bound action.`
    : "";

  const pending = esc.status === "pending";
  $("approve").disabled = !pending || esc.kind !== "approval";
  $("deny").disabled = !pending;
  $("note").value = "";
  $("d-result").hidden = true;
  show("detail");
}

// The note, or null (with an error shown) when it is shorter than 5 characters.
function note() {
  const text = $("note").value.trim();
  if (text.length < 5) {
    showError("Write a note of at least 5 characters.");
    return null;
  }
  return text;
}

// Approve after a confirmation that repeats the raw amount and order.
async function approve() {
  const text = note();
  if (text === null) return;
  const esc = current.escalation;
  const action = JSON.parse(current.action_json);
  const question = `Approve refund of ${money(action.amount_cents / 100)} on ${action.order_id} (${action.claim_type})?`;
  if (!window.confirm(question)) return;
  if (!approveKey || approveKey.escId !== esc.id) approveKey = { escId: esc.id, key: crypto.randomUUID() };
  await decide(`/escalations/${encodeURIComponent(esc.id)}/approve`, text, { "Idempotency-Key": approveKey.key });
}

// Deny after a confirmation.
async function deny() {
  const text = note();
  if (text === null) return;
  const esc = current.escalation;
  if (!window.confirm(`Deny escalation ${esc.id} on ${esc.order_id || "no order"}?`)) return;
  await decide(`/escalations/${encodeURIComponent(esc.id)}/deny`, text, {});
}

// Send a decision; show the server's result and the new trail. A network failure keeps the approve key for retry.
async function decide(path, text, headers) {
  $("approve").disabled = $("deny").disabled = true;
  try {
    const out = await api(path, { method: "POST", headers, body: JSON.stringify({ note: text }) });
    approveKey = null;
    current.trail = out.trail;
    $("d-result").textContent = JSON.stringify({ result: out.result, escalation: out.escalation }, null, 2);
    $("d-result").hidden = false;
    renderTrail(out.trail);
    showError("");
  } catch (err) {
    if (err.status) approveKey = null;  // the server answered: a new click is a new intent
    showError("Decision failed: " + err.message + (err.status ? "" : " (network; retrying reuses the same key)"));
    $("approve").disabled = current.escalation.kind !== "approval";
    $("deny").disabled = false;
  }
}

// Poll the queue every POLL_MS while it is on screen.
function startPolling() {
  stopPolling();
  show("queue");
  loadQueue();
  pollTimer = setInterval(loadQueue, POLL_MS);
}

// Stop polling.
function stopPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
}

// Forget the token and return to the login view.
function signOut(message) {
  stopPolling();
  try { sessionStorage.removeItem("resolv_token"); } catch { /* storage blocked */ }
  $("who").textContent = "";
  show("login");
  if (message) showError(message);
}

$("login-form").addEventListener("submit", (event) => {
  event.preventDefault();
  try { sessionStorage.setItem("resolv_token", $("token").value.trim()); } catch { /* storage blocked */ }
  $("token").value = "";
  startPolling();
});
$("logout").addEventListener("click", () => signOut(""));
$("back").addEventListener("click", startPolling);
$("approve").addEventListener("click", approve);
$("deny").addEventListener("click", deny);

if (token()) startPolling(); else show("login");
