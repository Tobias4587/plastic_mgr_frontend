# app_cards_voice.py
import os, io, wave, json, hashlib, re, requests
import streamlit as st
from openai import OpenAI
from urllib.parse import quote
from pathlib import Path
from dotenv import load_dotenv, dotenv_values  # ← add dotenv_values
from streamlit_mic_recorder import mic_recorder
import hashlib
from openai import BadRequestError  # add at the imports top


# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Plastic Manager – Cards", layout="wide")


# Load .env that sits next to this file (Frontend/.env), independent of cwd
ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)

OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or dotenv_values(ENV_PATH).get("OPENAI_API_KEY")
    or os.getenv("OPENAI_APIKEY")  # optional legacy name
)
if not OPENAI_API_KEY:
    st.error(f"Missing OPENAI_API_KEY. Expected at: {ENV_PATH}")
    st.caption("Ensure file is named '.env' (no .txt), UTF-8, line: OPENAI_API_KEY=sk-...")
    st.stop()


BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5000")
REQ_TIMEOUT = 30
ASST_MODEL = os.getenv("ASST_MODEL", "gpt-4o")
TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
TRANSCRIBE_LANGUAGE = (os.getenv("TRANSCRIBE_LANGUAGE") or "").strip()
TERMINAL_STATES = {"completed", "failed", "cancelled", "expired"}
client = OpenAI(api_key=OPENAI_API_KEY)



#define default card for doc upload
DEFAULT_INBOX_CARD_ID = os.getenv("DEFAULT_INBOX_CARD_ID", "C_000")


# ----------------------------
# Helpers
# ----------------------------
def _q(s: str) -> str:
    return quote(s or "", safe="")

SYSTEM_PROMPT = SYSTEM_PROMPT = """
You are a Cards Assistant for a simple finance tracker.
- Create/list/update cards via tools with fields: business_partner, description, type, quantity, amount, date.
- Create/list/update items on a card with fields: description, quantity, amount, date.
- Card IDs follow C_###; Item IDs follow Item_### (assigned by backend).
- You can resolve where queued attachments should be stored via the tool 'store_attachment_intent'.
- When the user asks to VIEW attachments (EN: attachments, files, docs; DE: Anhang, Anhänge, Datei, Dokumente, Beilagen, Attachments),
  you MUST call exactly one of: list_card_attachments(card_id) or list_item_attachments(card_id,item_id) BEFORE answering.
  Never infer attachment presence without a tool call.
- If you are not sure which card/item, ask one targeted question to disambiguate.
- Be concise and only ask for missing essentials.
- Always call tools when available.

Attachment policy:
- If the user expresses intent to attach/upload previously staged files (e.g., “upload now”, “attach now”, “bitte an C_014 anhängen”,
  “attach to C_014”, “add to card 14”, “an karte 14”), you MUST call the tool `commit_staged`:
  - If they specify a card: commit_staged(card_id=<that card>, item_id if they gave one).
  - Otherwise, use the CURRENT_TARGET from the dynamic context if it exists; if not, ask one clarifying question.
- Never claim files were attached unless `commit_staged` returns success. Summarize with the returned filenames/links.
- If there are zero staged files in the dynamic context and the user asks to attach, tell them there are no staged files and instruct them to stage first.
"""





def open_uploader():
    st.session_state["uploader_open"] = True

def close_uploader():
    st.session_state["uploader_open"] = False
    st.session_state.pop("attach_target", None)

def current_target_str(card_id: str | None, item_id: str | None) -> str:
    if not card_id:
        return "(no target)"
    return card_id + (f"/{item_id}" if item_id else "")



def _drain_active_runs(thread_id: str):
    """
    Make sure there are no active runs on the thread.
    Handles queued/in_progress/requires_action (incl. tool calls) until all runs are terminal.
    """
    # If the SDK supports listing runs:
    try:
        runs = client.beta.threads.runs.list(thread_id=thread_id).data
    except Exception:
        runs = []

    # If we have a run id in session, include it first to guarantee we drain it
    ids = []
    if st.session_state.get("active_run_id"):
        ids.append(st.session_state["active_run_id"])
    ids.extend([r.id for r in runs if r.id not in ids])

    for run_id in ids:
        while True:
            run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
            status = getattr(run, "status", "")
            if status in ("queued", "in_progress"):
                run = client.beta.threads.runs.poll(thread_id=thread_id, run_id=run_id)
                continue
            if status == "requires_action":
                outs = []
                for tc in run.required_action.submit_tool_outputs.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments or "{}")
                    out = call_tool(name, args)
                    outs.append({"tool_call_id": tc.id, "output": out or ""})
                run = client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id, run_id=run_id, tool_outputs=outs
                )
                # loop back to poll the same run_id again
                continue
            # terminal (completed/failed/…) -> stop draining this run
            break

    # clear our local flag if we had one
    st.session_state.pop("active_run_id", None)



# --- Backend fetchers with basic pagination ---
def fetch_cards(offset=0, limit=50):
    try:
        r = requests.get(f"{BACKEND_URL}/cards", params={"offset": offset, "limit": limit}, timeout=REQ_TIMEOUT)
        j = r.json()
        # Support both old (list) and new (paged) backends
        if isinstance(j, dict) and "data" in j:
            return j.get("data", []), j.get("total", len(j.get("data", []))), j.get("offset", offset), j.get("limit", limit)
        elif isinstance(j, list):
            return j, len(j), 0, len(j)
        return [], 0, 0, limit
    except Exception:
        return [], 0, 0, limit

def fetch_items(card_id, offset=0, limit=50):
    try:
        cid = normalize_card_id(card_id)
        r = requests.get(f"{BACKEND_URL}/cards/{_q(cid)}/items", params={"offset": offset, "limit": limit}, timeout=REQ_TIMEOUT)
        j = r.json()
        if isinstance(j, dict) and "data" in j:
            return j.get("data", []), j.get("total", len(j.get("data", [])))
        elif isinstance(j, list):
            return j, len(j)
        return [], 0
    except Exception:
        return [], 0

TRANSCRIBE_ALLOWED = {
    s.strip().lower() for s in os.getenv("TRANSCRIBE_ALLOWED", "en,fr,de").split(",") if s.strip()
}
FORCE_LANG = (os.getenv("TRANSCRIBE_LANGUAGE") or "en").strip().lower()
if FORCE_LANG not in TRANSCRIBE_ALLOWED:
    raise ValueError(f"TRANSCRIBE_LANGUAGE '{FORCE_LANG}' not in allowed set {sorted(TRANSCRIBE_ALLOWED)}")

def transcribe(wav: bytes) -> str:
    tr = client.audio.transcriptions.create(
        model=TRANSCRIBE_MODEL,
        file=("speech.wav", wav, "audio/wav"),
        response_format="json",   # NOT 'verbose_json'
        language=FORCE_LANG
    )
    return (getattr(tr, "text", None) or "").strip()



def normalize_card_id(v: str) -> str:
    s = str(v).strip().upper()
    m = re.search(r'(\d+)$', s)
    if m:
        n = m.group(1)
        pad = 3 if int(n) <= 999 else 4
        return f"C_{n.zfill(pad)}"
    return s


def _compose_dynamic_instructions() -> str:
    names = []
    # We didn't store names, but staged tokens exist; the model only needs to know count.
    # If you want names, save them alongside tokens when staging.
    staged_tokens = st.session_state.get("staged_tokens") or []
    staged_count = len(staged_tokens)

    curr_card = st.session_state.get("current_card_id") or ""
    curr_item = st.session_state.get("current_item_id") or ""

    return (
        "Dynamic context for this user turn:\n"
        f"- STAGED_FILES_COUNT: {staged_count}\n"
        f"- CURRENT_TARGET_CARD: {curr_card}\n"
        f"- CURRENT_TARGET_ITEM: {curr_item}\n"
        "If the user asks to attach or upload staged files and STAGED_FILES_COUNT>0, "
        "call commit_staged with the specified card/item; else with CURRENT_TARGET_* if present; "
        "else ask one clarifying question.\n"
    )


def render_attachments(card_id: str, item_id: str | None = None, show_previews: bool = False):
    """Render attachments compactly. Only show image previews on demand (button) or if show_previews=True."""
    try:
        url = (f"{BACKEND_URL}/cards/{_q(card_id)}/items/{_q(item_id)}/attachments"
               if item_id else f"{BACKEND_URL}/cards/{_q(card_id)}/attachments")
        r = requests.get(url, timeout=REQ_TIMEOUT)
        if r.status_code != 200:
            st.info("No attachments found."); return
        att = r.json() or []
        if not isinstance(att, list) or not att:
            st.info("No attachments found."); return

        for a in att:
            raw = (a.get("url") or a.get("path") or "").strip()
            if raw.startswith(("http://","https://")):
                file_url = raw
            elif raw.startswith("sandbox:/"):
                file_url = f"{BACKEND_URL.rstrip('/')}/{raw.split('sandbox:/',1)[1].lstrip('/')}"
            elif raw.startswith("/"):
                file_url = f"{BACKEND_URL.rstrip('/')}{raw}"
            else:
                file_url = f"{BACKEND_URL.rstrip('/')}/{raw.lstrip('/')}"

            fn = a.get("filename", "(unnamed)")
            mime = a.get("mime", "")
            uploaded = (a.get("uploaded_at","") or "").split("T")[0]

            # One line: clickable name + meta + 'Open' + optional 'Preview' button
            col1, col2, col3, col4 = st.columns([0.6, 0.15, 0.15, 0.10])
            with col1:
                st.markdown(f"• [{fn}]({file_url})  \n_{mime}_  ·  {uploaded}")
            with col2:
                st.link_button("Open", file_url, use_container_width=True)
            with col3:
                wants_preview = False
                if mime.startswith("image/"):
                    wants_preview = st.button("Preview", key=f"prev_{card_id}_{item_id}_{fn}", use_container_width=True)
                else:
                    st.write("")  # spacer
            with col4:
                pass

            # Only show image when explicitly requested (or forced by flag)
            if (show_previews and mime.startswith("image/")) or wants_preview:
                st.image(file_url, clamp=True)
    except Exception as e:
        st.warning(f"Could not load attachments: {e}")

# ----------------------------
# Tools schema for the Assistant
# ----------------------------
def tools_schema():
    return [
        # Cards
        {"type": "function", "function": {
            "name": "create_card", "description": "Create a new card",
            "parameters": {"type": "object", "properties": {
                "business_partner": {"type": "string"},
                "description": {"type": "string"},
                "type": {"type": "string"},  # sale | procurement | expense
                "quantity": {"type": "number"},
                "unit": {"type": "string"},
                "amount": {"type": "number"},
                "currency": {"type": "string"},
                "date": {"type": "string", "format": "date"}
            }}
        }},
        {"type": "function", "function": {
            "name": "list_cards", "description": "List cards (optional filters)",
            "parameters": {"type": "object", "properties": {
                "business_partner": {"type": "string"},
                "type": {"type": "string"},
                "date": {"type": "string"}
            }}
        }},
        {"type": "function", "function": {
            "name": "update_card", "description": "Update a card",
            "parameters": {"type": "object", "properties": {
                "card_id": {"type": "string"},
                "business_partner": {"type": "string"},
                "description": {"type": "string"},
                "type": {"type": "string"},
                "quantity": {"type": "number"},
                "unit": {"type": "string"},
                "amount": {"type": "number"},
                "currency": {"type": "string"},
                "date": {"type": "string", "format": "date"}
            }, "required": ["card_id"]}
        }},
        # Items
        {"type": "function", "function": {
            "name": "add_item", "description": "Add item to a card",
            "parameters": {"type": "object", "properties": {
                "card_id": {"type": "string"},
                "description": {"type": "string"},
                "quantity": {"type": "number"},
                "amount": {"type": "number"},
                "date": {"type": "string", "format": "date"}
            }, "required": ["card_id", "description"]}
        }},
        {"type": "function", "function": {
            "name": "list_items", "description": "List items of a card",
            "parameters": {"type": "object", "properties": {
                "card_id": {"type": "string"}
            }, "required": ["card_id"]}
        }},
        {"type": "function", "function": {
            "name": "update_item", "description": "Update an item on a card",
            "parameters": {"type": "object", "properties": {
                "card_id": {"type": "string"},
                "item_id": {"type": "string"},
                "updates": {"type": "object", "properties": {
                    "description": {"type": "string"},
                    "quantity": {"type": "number"},
                    "amount": {"type": "number"},
                    "date": {"type": "string", "format": "date"}
                }}
            }, "required": ["card_id", "item_id", "updates"]}
        }},
        # Attachment destination resolver
        {"type": "function", "function": {
            "name": "store_attachment_intent",
            "description": "Resolve where to store most recent user attachments",
            "parameters": {"type": "object", "properties": {
                "card_id": {"type": "string"},
                "item_id": {"type": "string", "description": "optional"}
            }, "required": ["card_id"]}
        }},
        {"type":"function","function":{
            "name":"list_card_attachments",
            "description":"List attachments for a card",
            "parameters":{"type":"object","properties":{"card_id":{"type":"string"}},"required":["card_id"]}
        }},
        {"type":"function","function":{
            "name":"list_item_attachments",
            "description":"List attachments for an item of a card",
            "parameters":{"type":"object","properties":{
                "card_id":{"type":"string"},
                "item_id":{"type":"string"}
            },"required":["card_id","item_id"]}
        }},
        {"type": "function", "function": {
            "name": "commit_staged",
            "description": "Commit previously staged uploads to a specific card/item",
            "parameters": {"type": "object", "properties": {
                "card_id": {"type": "string"},
                "item_id": {"type": "string", "description": "optional"}
            }, "required": ["card_id"]}
        }}
    ]


# Create once (top, after OpenAI client):
if "asst_id" not in st.session_state:
    a = client.beta.assistants.create(
        name="Cards Assistant",
        instructions=SYSTEM_PROMPT,
        model=os.getenv("ASST_MODEL", "gpt-4o-mini"),  # faster for routing
        tools=tools_schema(),
    )
    st.session_state["asst_id"] = a.id

# ----------------------------
# Backend bridge
# ----------------------------
def call_tool(name, args):
    if name == "create_card":
        r = requests.post(f"{BACKEND_URL}/cards", json=args, timeout=REQ_TIMEOUT)
    elif name == "list_cards":
        r = requests.get(f"{BACKEND_URL}/cards", params=args, timeout=REQ_TIMEOUT)
    elif name == "update_card":
        body = dict(args)
        cid = normalize_card_id(body.pop("card_id"))
        st.session_state["current_card_id"] = cid
        r = requests.patch(f"{BACKEND_URL}/cards/{_q(cid)}", json=body, timeout=REQ_TIMEOUT)
    elif name == "add_item":
        body = dict(args)
        cid = normalize_card_id(body.pop("card_id"))
        st.session_state["current_card_id"] = cid
        r = requests.post(f"{BACKEND_URL}/cards/{_q(cid)}/items", json=body, timeout=REQ_TIMEOUT)
    elif name == "list_items":
        cid = normalize_card_id(args["card_id"])
        r = requests.get(f"{BACKEND_URL}/cards/{_q(cid)}/items", timeout=REQ_TIMEOUT)
        st.session_state["current_card_id"] = cid
        st.session_state["current_item_id"] = None
    elif name == "update_item":
        body = dict(args)
        cid = normalize_card_id(body.pop("card_id"))
        iid = body.pop("item_id")
        st.session_state["current_card_id"] = cid
        st.session_state["current_item_id"] = iid
        r = requests.patch(f"{BACKEND_URL}/cards/{_q(cid)}/items/{_q(iid)}", json=body, timeout=REQ_TIMEOUT)
    elif name == "store_attachment_intent":
        # Apply resolved targets to queued uploads and push to backend
        applied = []
        if "pending_uploads" in st.session_state:
            for p in list(st.session_state["pending_uploads"]):
                p["card_id"] = normalize_card_id(args.get("card_id"))
                p["item_id"] = args.get("item_id")
                try:
                    res = upload_to_backend(p)
                    if isinstance(res, dict) and res.get("status") == "uploaded":
                        applied.append(p["name"])
                        st.session_state["pending_uploads"].remove(p)
                except Exception:
                    pass
        return json.dumps({"stored": applied})
    
    elif name == "list_card_attachments":
        cid = normalize_card_id(args["card_id"])
        st.session_state["current_card_id"] = cid
        st.session_state["current_item_id"] = None
        resp = requests.get(f"{BACKEND_URL}/cards/{_q(cid)}/attachments", timeout=REQ_TIMEOUT)
        data = resp.json() if resp.status_code == 200 else []
        # UI render (compact)
        #render_attachments(cid, show_previews=False)
        for a in data:
            a.setdefault("url", f"{BACKEND_URL.rstrip('/')}{a.get('path','')}")
        return json.dumps({"card_id": cid, "count": len(data), "attachments": data})

    elif name == "list_item_attachments":
        cid = normalize_card_id(args["card_id"]); iid = args["item_id"]
        st.session_state["current_card_id"] = cid
        st.session_state["current_item_id"] = iid
        resp = requests.get(f"{BACKEND_URL}/cards/{_q(cid)}/items/{_q(iid)}/attachments", timeout=REQ_TIMEOUT)
        data = resp.json() if resp.status_code == 200 else []
        #render_attachments(cid, iid, show_previews=False)
        for a in data:
            a.setdefault("url", f"{BACKEND_URL.rstrip('/')}{a.get('path','')}")
        return json.dumps({"card_id": cid, "item_id": iid, "count": len(data), "attachments": data})

    elif name == "commit_staged":
        cid = normalize_card_id(args["card_id"])
        iid = args.get("item_id")

        # keep UI context in sync
        st.session_state["current_card_id"] = cid
        st.session_state["current_item_id"] = iid

        # commit tokens we staged earlier
        metas, errs = commit_staged(cid, iid)

        # render authoritative list in the UI (compact, no previews)
        render_attachments(cid, iid, show_previews=False)

        # return a compact text summary for the assistant’s reply
        if metas:
            dest = cid + (f"/{iid}" if iid else "")
            lines = [f"Committed {len(metas)} attachment(s) to **{dest}**:"]
            for m in metas:
                url = m.get("url") or f"{BACKEND_URL.rstrip('/')}{m.get('path','')}"
                fn  = m.get("filename", "(unbenannt)")
                lines.append(f"- [{fn}]({url})")
            return "\n".join(lines)
        else:
            return "No staged attachments to commit."
        
    else:
        return None
    return r.text if r is not None else None

def run_assistant(user_text: str):
    # Ensure thread
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = client.beta.threads.create().id
    tid = st.session_state.thread_id

    # Drain any active/blocked runs before we add a new message
    _drain_active_runs(tid)

    # Add the user message (with a safety retry if API says a run is active)
    try:
        client.beta.threads.messages.create(thread_id=tid, role="user", content=user_text)
    except BadRequestError as e:
        # If another run became active between drain and now, drain again once, then retry
        if "while a run" in str(e):
            _drain_active_runs(tid)
            client.beta.threads.messages.create(thread_id=tid, role="user", content=user_text)
        else:
            raise

    # pass dynamic state so the model knows about staged files + current target
    extra = _compose_dynamic_instructions()

    # Start one run and remember its id
    run = client.beta.threads.runs.create(
        thread_id=tid,
        assistant_id=st.session_state["asst_id"],
        instructions=extra,   # <-- the important line
    )
    st.session_state["active_run_id"] = run.id

    # Drive this run to a terminal state (handle tool calls as needed)
    while True:
        run = client.beta.threads.runs.poll(thread_id=tid, run_id=run.id)
        status = getattr(run, "status", "")
        if status == "requires_action":
            outs = []
            for tc in run.required_action.submit_tool_outputs.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                out = call_tool(name, args)
                outs.append({"tool_call_id": tc.id, "output": out or ""})
            run = client.beta.threads.runs.submit_tool_outputs(
                thread_id=tid, run_id=run.id, tool_outputs=outs
            )
            continue
        if status in ("queued", "in_progress"):
            continue
        break

    # Clear active flag
    st.session_state.pop("active_run_id", None)

    # Return last assistant message
    msgs = client.beta.threads.messages.list(thread_id=tid)
    ai_msgs = [m for m in msgs.data if m.role == "assistant" and m.content]
    return (ai_msgs[0].content[0].text.value if ai_msgs else "OK")

# ----------------------------
# Upload plumbing
# ----------------------------
def upload_to_backend(entry: dict):
    cid = normalize_card_id(entry.get("card_id")) if entry.get("card_id") else None
    item_id = entry.get("item_id")
    if not cid:
        return {"status": "queued_no_target"}

    url = f"{BACKEND_URL}/cards/{_q(cid)}/attachments"
    files = {"file": (entry["name"], entry["bytes"], entry.get("type") or "application/octet-stream")}
    data = {}
    if item_id:
        data["item_id"] = item_id
    r = requests.post(url, files=files, data=data, timeout=REQ_TIMEOUT)
    try:
        return r.json()
    except Exception:
        return {"error": r.text, "status_code": r.status_code}

def stage_file_locally(f):
    return {
        "name": getattr(f, "name", "file"),
        "type": getattr(f, "type", "application/octet-stream"),
        "bytes": f.getvalue(),
    }

def commit_staged_to(card_id: str, item_id: str | None, tokens: list[str]):
    results = []
    for t in tokens:
        payload = {"token": t, "card_id": card_id}
        if item_id: payload["item_id"] = item_id
        r = requests.post(f"{BACKEND_URL}/attachments/commit", json=payload, timeout=REQ_TIMEOUT)
        try:
            results.append(r.json())
        except Exception:
            results.append({"_status_code": r.status_code, "_raw": r.text})
    return results

def commit_staged(card_id: str, item_id: str | None = None):
    """
    Commit all staged upload tokens from session to the given card/item.
    Returns (metas, errors); also clears the staged tokens from session.
    """
    tokens = st.session_state.get("staged_tokens", []) or []
    if not tokens:
        return [], []

    cid = normalize_card_id(card_id)
    metas = []
    errors = []

    for t in tokens:
        payload = {"token": t, "card_id": cid}
        if item_id:
            payload["item_id"] = item_id
        try:
            r = requests.post(f"{BACKEND_URL}/attachments/commit", json=payload, timeout=REQ_TIMEOUT)
            if r.status_code == 200:
                # expect {"status":"committed","meta":{...}}
                js = r.json()
                metas.append(js.get("meta", {}))
            else:
                errors.append({"token": t, "status": r.status_code, "raw": r.text})
        except Exception as e:
            errors.append({"token": t, "status": "client", "raw": str(e)})

    # IMPORTANT: clear staged tokens so we don’t double-commit
    st.session_state["staged_tokens"] = []
    return metas, errors


# ----------------------------
# UI
# ----------------------------
st.title("Plastic Manager — Cards")
st.caption("Create and manage cards & items (voice or text).")

# Chat history
if "chat" not in st.session_state:
    st.session_state.chat = []
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# --- UI context state ---
st.session_state.setdefault("current_card_id", None)
st.session_state.setdefault("current_item_id", None)
st.session_state.setdefault("cards_offset", 0)
st.session_state.setdefault("camera_on", False)


# #Show current context above chat input
# if st.session_state.get("current_card_id"):
#     ctx = st.session_state["current_card_id"]
#     if st.session_state.get("current_item_id"):
#         ctx += f" / {st.session_state['current_item_id']}"
#     st.caption(f"Current context: {ctx} (uploads go here). If none, uploads go to {DEFAULT_INBOX_CARD_ID}.")



# # --- Minimal uploader: pick file or take photo; always confirm destination ---
# if st.session_state.get("uploader_open"):
#     with st.expander("Add attachment", expanded=True):
#         # Small, simple controls
#         files = st.file_uploader("Select files", accept_multiple_files=True)

#         # Camera is OFF by default; only render when user asks
#         c1, c2, c3 = st.columns([0.22, 0.18, 0.60])
#         with c1:
#             if not st.session_state["camera_on"]:
#                 if st.button("Use camera", key="btn_camera_on", use_container_width=True):
#                     st.session_state["camera_on"] = True
#                     st.rerun()
#             else:
#                 if st.button("Close camera", key="btn_camera_off", use_container_width=True):
#                     st.session_state["camera_on"] = False
#                     st.rerun()
#         with c2:
#             if st.button("Cancel", key="btn_upload_cancel", use_container_width=True):
#                 st.session_state["camera_on"] = False
#                 close_uploader()
#                 st.rerun()

#         photo = None
#         if st.session_state["camera_on"]:
#             photo = st.camera_input("Take a picture")

#         st.divider()

#         # Destination is the current context (if any)
#         target_card = st.session_state.get("current_card_id")
#         target_item = st.session_state.get("current_item_id")
#         st.caption(f"Destination: {current_target_str(target_card, target_item)}")

#         # Upload button
#         do_upload = st.button("Upload", key="btn_upload_now", use_container_width=False)

#         if do_upload:
#             # If no card context and you don't have an Inbox card, stop and ask the user
#             if not target_card:
#                 target_card = os.getenv("DEFAULT_INBOX_CARD_ID", "C_000")  # rely on backend-initialized Inbox
#                 target_item = None

#             # Collect payloads
#             to_upload = []
#             if files:
#                 to_upload += files
#             if photo:
#                 to_upload.append(photo)

#             if not to_upload:
#                 st.info("Please select a file or take a photo.")
#                 st.stop()

#             uploaded_count = 0
#             uploaded = []        # <-- NEW: collect metas for chat message
#             last_meta = None
#             for f in to_upload:
#                 entry = {
#                     "name": getattr(f, "name", "photo.jpg"),
#                     "type": getattr(f, "type", "image/jpeg"),
#                     "bytes": f.getvalue(),
#                     "card_id": target_card,
#                     "item_id": target_item or None
#                 }
#                 try:
#                     res = upload_to_backend(entry)
#                 except Exception as e:
#                     st.error(f"Upload failed: {e}")
#                     continue

#                 if isinstance(res, dict) and res.get("status") == "uploaded":
#                     uploaded_count += 1
#                     last_meta = res.get("meta")

#             if uploaded_count > 0:
#                 dest = current_target_str(target_card, target_item)

#                 # Compose a short assistant message with links (if backend returned absolute URLs)
#                 lines = [f"**Upload erfolgreich.** {uploaded_count} Datei(en) gespeichert unter **{dest}**:"]
#                 for m in uploaded:  # uploaded is a list of res["meta"]
#                     url = (m.get("url") or f"{BACKEND_URL.rstrip('/')}{m.get('path','')}")
#                     fn  = m.get("filename","(unbenannt)")
#                     lines.append(f"- [{fn}]({url})")

#                 st.session_state.chat.append({
#                     "role": "assistant",
#                     "content": "\n".join(lines)
#                 })

#                 # Keep the in-place success toast as well
#                 st.success(f"Uploaded {uploaded_count} file(s) to {dest}.")

#                 # Show authoritative list immediately (compact; no auto image previews)
#                 render_attachments(target_card, target_item, show_previews=False)

#                 # Close uploader and camera, then re-run for a clean UI
#                 st.session_state["camera_on"] = False
#                 close_uploader()
#                 st.rerun()

#             else:
#                 st.warning("Nothing uploaded. Please try again.")


# --- Minimal uploader: stage only ---
if st.session_state.get("uploader_open"):
    with st.expander("Add attachment", expanded=True):
        files = st.file_uploader("Select files", accept_multiple_files=True, key="uploader_files")
        use_cam = st.toggle("Use camera", value=False)
        photo = st.camera_input("Take a picture") if use_cam else None

        # st.divider()
        # target_card = st.session_state.get("current_card_id")
        # target_item = st.session_state.get("current_item_id")
        # st.caption(f"Current target (for later commit): {current_target_str(target_card, target_item)}")

        if st.button("Stage file(s)", key="btn_stage"):
            to_stage = []
            if files: to_stage += files
            if photo: to_stage.append(photo)
            if not to_stage:
                st.info("Bitte zuerst eine Datei auswählen oder ein Foto aufnehmen.")
                st.stop()

            staged_tokens, staged_names, errors = [], [], []

            for f in to_stage:
                # inside: for f in to_stage:
                fname = getattr(f, "name", "photo.jpg")
                ftype = getattr(f, "type", "application/octet-stream")
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/uploads",
                        files={"file": (fname, f.getvalue(), ftype)},
                        timeout=REQ_TIMEOUT,
                    )
                except Exception as e:
                    errors.append(f"{fname}: client error {e}")
                    continue

                # Try to parse JSON for details
                try:
                    js = resp.json()
                except Exception:
                    js = {"_raw": resp.text}

                if resp.status_code in (200, 201) and js.get("status") == "staged":
                    meta = js.get("meta") or {}
                    token = meta.get("token")
                    if token:                                  # <-- tiny check
                        staged_tokens.append(token)
                        staged_names.append(meta.get("filename", fname))
                    else:
                        errors.append(f"{fname}: missing token in response {js}")  # <-- graceful diag
                else:
                    errors.append(f"{fname}: HTTP {resp.status_code} -> {js}")

            if staged_tokens:
                st.session_state.setdefault("staged_tokens", [])
                st.session_state["staged_tokens"].extend(staged_tokens)

                # Chat confirmation + proposal
                target_card = st.session_state.get("current_card_id")
                target_item = st.session_state.get("current_item_id")
                dest = current_target_str(target_card, target_item)
                st.session_state.chat.append({
                    "role": "assistant",
                    "content": (
                        f"**Staging erfolgreich.** {len(staged_tokens)} Datei(en) vorgemerkt: "
                        + ", ".join(staged_names)
                        + f". Soll ich auf **{dest}** anhängen? Tippen Sie **'upload now'**."
                    ),
                })

                # Close uploader for compact UI
                st.session_state["uploader_open"] = False
                st.rerun()
            else:
                st.warning("Staging fehlgeschlagen.")
                # Show diagnostics both in UI and chat
                if errors:
                    err_text = "\n".join(f"- {e}" for e in errors)
                    with st.expander("Staging diagnostics", expanded=True):
                        st.code(err_text)
                    st.session_state.chat.append({
                        "role": "assistant",
                        "content": "**Staging fehlgeschlagen.** Details:\n" + err_text,
                    })




# --- Sticky bottom toolbar (below content, above chat input) ---
st.markdown("""
<style>
.toolbar { position: sticky; bottom: 0; z-index: 999; background: white;
           padding: .5rem .75rem; border-top: 1px solid #eee; }
.toolbar .stButton>button { padding: .25rem .6rem; font-size: .9rem; }
.toolbar .st-emotion-cache-ocqkz7 { margin: 0; } /* tighten columns */
</style>
""", unsafe_allow_html=True)

audio = None  # ensure defined

with st.container():
    st.markdown('<div class="toolbar">', unsafe_allow_html=True)
    c_plus, c_mic, c_reset, _ = st.columns([0.12, 0.18, 0.22, 0.48])

    with c_plus:
        if st.button("➕", help="Add attachment (file or camera)", key="btn_plus_footer"):
            open_uploader()

    with c_mic:
        audio = mic_recorder(
            start_prompt="🎙️ Record",
            stop_prompt="⏹️ Stop",
            format="wav",
            just_once=True,                 # single-shot; prevents double-capture
            use_container_width=True,
            key="mic_toolbar",              # ← unique key
        )

    with c_reset:
        if st.button("Reset session", key="btn_reset_footer"):
            st.session_state.clear()
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# Process mic capture (single place)
if audio and audio.get("bytes"):
    wav = bytes(audio["bytes"])
    if wav:
        h = hashlib.md5(wav).hexdigest()
        if st.session_state.get("last_mic_hash") != h:
            st.session_state["last_mic_hash"] = h
            try:
                text = transcribe(wav)
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                text = ""
            if text:
                st.session_state.chat.append({"role":"user","content":text})
                st.session_state["pending"] = {"type":"voice","payload":text}
                # re-arm mic component cleanly
                st.session_state.pop("mic_toolbar", None)
                st.rerun()



# Chat input
user_msg = st.chat_input("Type a message…")
if user_msg:
    st.session_state.chat.append({"role": "user", "content": user_msg})
    txt = user_msg.strip()

    # Try to set context from free text like: "Karte 7", "Card 19", "C_019", etc.
    m_ctx = re.search(r"(?i)\b(?:karte|card)?\s*([cC]?\s*_?\s*\d{1,5})\b", user_msg.strip())
    if m_ctx:
        raw = re.sub(r"\s", "", m_ctx.group(1))
        cid = normalize_card_id(raw)     # "7" -> "C_007"
        st.session_state["current_card_id"] = cid
        st.session_state["current_item_id"] = None   # reset item on card change


    # 0) Optional: "show/zeige attachments ... with preview/vorschau" (context-based)
    #    e.g., "zeige anhänge mit vorschau" or "show attachments with preview"
    m_prev_ctx = re.search(r"(?i)\b(attachments|anh(?:ä|ae)nge).*(preview|vorschau)", txt)
    if m_prev_ctx and st.session_state.get("current_card_id"):
        cid = st.session_state["current_card_id"]
        iid = st.session_state.get("current_item_id")
        with st.chat_message("assistant"):
            st.markdown(
                f"Attachments (preview) for **{cid}**" + (f"/**{iid}**" if iid else "")
            )
            render_attachments(cid, iid, show_previews=True)
        st.stop()

    # 1) Explicit: "show/zeige attachments [of card] <id>[/<item>]" (no preview)
    m = re.search(
        r"(?i)\b(?:show|zeige|zeigen)\s+(?:mir\s+)?(?:die\s+)?(?:attachments|anh(?:ä|ae)nge)"
        r"(?:\s+(?:of|von|der|der\s+karte|card))?\s+([a-z_]*\d+)"
        r"(?:\s*/\s*([a-z0-9_]+))?\b",
        txt
    )
    if m:
        raw_id = m.group(1)
        cid = normalize_card_id(raw_id)           # "18" -> "C_018"
        iid = m.group(2) if m.lastindex and m.group(2) else None
        with st.chat_message("assistant"):
            st.markdown(f"Showing attachments for **{cid}**" + (f"/**{iid}**" if iid else ""))
            render_attachments(cid, iid, show_previews=False)  # compact by default
        st.stop()

    # 2) Explicit: "show/zeige attachments ... with preview/vorschau for card <id>"
    m_prev_id = re.search(
        r"(?i)\b(?:show|zeige|zeigen)\s+(?:mir\s+)?(?:die\s+)?(?:attachments|anh(?:ä|ae)nge).*(preview|vorschau).*"
        r"(?:of|von|der|der\s+karte|card)\s+([a-z_]*\d+)(?:\s*/\s*([a-z0-9_]+))?\b",
        txt
    )
    if m_prev_id:
        cid = normalize_card_id(m_prev_id.group(2))
        iid = m_prev_id.group(3) if m_prev_id.lastindex and m_prev_id.group(3) else None
        with st.chat_message("assistant"):
            st.markdown(f"Showing attachments (preview) for **{cid}**" + (f"/**{iid}**" if iid else ""))
            render_attachments(cid, iid, show_previews=True)
        st.stop()

    # Commit staged files on command
    if re.search(r"(?i)\b(upload\s*now|jetzt\s*hochladen|jetzt\s*anh[aä]ngen)\b", txt):
        # Require staged tokens first
        if not (st.session_state.get("staged_tokens") or []):
            st.session_state.chat.append({
                "role": "assistant",
                "content": "Es sind derzeit keine Dateien vorgemerkt. Bitte zuerst Datei(en) **stagen**."
            })
            st.stop()

        card_id = st.session_state.get("current_card_id")
        item_id = st.session_state.get("current_item_id")

        if not card_id:
            st.warning("Kein Ziel konfiguriert. Bitte Karte angeben (z. B. 'an C_013').")
            st.stop()

        metas, errs = commit_staged(card_id, item_id)   # clears staged_tokens internally

        if metas:
            # keep context aligned with destination
            st.session_state["current_card_id"] = card_id
            if not item_id:
                st.session_state["current_item_id"] = None

            dest = card_id + (f"/{item_id}" if item_id else "")
            lines = [f"**Anhänge gespeichert** unter **{dest}**:"]
            for m in metas:
                url = m.get("url") or f"{BACKEND_URL.rstrip('/')}{m.get('path','')}"
                fn  = m.get("filename","(unbenannt)")
                lines.append(f"- [{fn}]({url})")

            st.session_state.chat.append({"role": "assistant", "content": "\n".join(lines)})

            # show authoritative list immediately (compact)
            render_attachments(card_id, item_id, show_previews=False)
            st.rerun()

        else:
            # Optional: include brief diagnostics
            if errs:
                err_lines = "; ".join([f"{e.get('token','?')}: {e.get('status')} {e.get('raw','')}" for e in errs])
                st.session_state.chat.append({"role":"assistant","content": f"Kein Anhang übernommen. Fehler: {err_lines}"})
            else:
                st.session_state.chat.append({"role":"assistant","content":"Kein Anhang wurde übernommen. Prüfen Sie bitte die Staging-Schritte."})
            st.stop()




    # Otherwise, send to assistant (LLM path)
    st.session_state["pending"] = {"type": "text", "payload": user_msg}
    st.rerun()







# Process pending (single place)
if st.session_state.get("pending"):
    to_process = st.session_state.pop("pending")["payload"]

    # Try uploading any queued files where a target is already known
    if st.session_state.get("pending_uploads"):
        still_queued = []
        for p in st.session_state["pending_uploads"]:
            if p.get("card_id"):
                try:
                    upload_to_backend(p)
                except Exception:
                    still_queued.append(p)
            else:
                still_queued.append(p)
        st.session_state["pending_uploads"] = still_queued

    # Let the assistant handle the user text (may call store_attachment_intent)
    reply = run_assistant(to_process)
    st.session_state.chat.append({"role": "assistant", "content": reply})
    st.rerun()

