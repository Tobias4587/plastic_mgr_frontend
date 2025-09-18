import os, io, wave, json, hashlib, time, re
import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from audiorecorder import audiorecorder

# ---------- Config ----------
st.set_page_config(page_title="Plastic Manager – Cards", layout="wide")
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5000")
ASST_MODEL = os.getenv("ASST_MODEL", "gpt-4o")
TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")  # or "whisper-1"
TRANSCRIBE_LANGUAGE = (os.getenv("TRANSCRIBE_LANGUAGE") or "").strip()

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in .env")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Minimal system instructions (embedded; no sidebar) ----------
SYSTEM_PROMPT = (
    "You are a Cards Assistant for a simple finance tracker.\n"
    "- Create/list/update cards via tools with fields: business_partner, description, type, quantity, amount, date.\n"
    "- Create/list/update items on a card with fields: description, quantity, amount, date.\n"
    "- Card IDs follow C_###; Item IDs follow Item_### (assigned by backend).\n"
    "- Be concise. Only ask for truly missing required info (e.g., description or card_id).\n"
    "- Always call tools when available."
)

# ---------- Tool schemas ----------
def tools_schema():
    return [
        # Cards
        {"type":"function","function":{
            "name":"create_card","description":"Create a new card",
            "parameters":{"type":"object","properties":{
                "business_partner":{"type":"string"},
                "description":{"type":"string"},
                "type":{"type":"string"},          # e.g., sale | procurement | expense
                "quantity":{"type":"number"},
                "unit":{"type":"string", "description":"e.g., kg (default)"},
                "amount":{"type":"number"},
                "currency":{"type":"string", "description":"e.g., CFA (default)"},
                "date":{"type":"string","format":"date"}
            }}}},

        {"type":"function","function":{
            "name":"list_cards","description":"List cards (optional filters)",
            "parameters":{"type":"object","properties":{
                "business_partner":{"type":"string"},
                "type":{"type":"string"},
                "date":{"type":"string"}
            }}}},

        {"type":"function","function":{
            "name":"update_card","description":"Update a card",
            "parameters":{"type":"object","properties":{
                "card_id":{"type":"string"},
                "business_partner":{"type":"string"},
                "description":{"type":"string"},
                "type":{"type":"string"},
                "quantity":{"type":"number"},
                "unit":{"type":"string"},
                "amount":{"type":"number"},
                "currency":{"type":"string"},
                "date":{"type":"string","format":"date"}
            },"required":["card_id"]}}},

        # Items
        {"type":"function","function":{
            "name":"add_item","description":"Add item to a card",
            "parameters":{"type":"object","properties":{
                "card_id":{"type":"string"},
                "description":{"type":"string"},
                "quantity":{"type":"number"},
                "amount":{"type":"number"},
                "date":{"type":"string","format":"date"}
            },"required":["card_id","description"]}}},

        {"type":"function","function":{
            "name":"list_items","description":"List items of a card",
            "parameters":{"type":"object","properties":{
                "card_id":{"type":"string"}
            },"required":["card_id"]}}},

        {"type":"function","function":{
            "name":"update_item","description":"Update an item on a card",
            "parameters":{"type":"object","properties":{
                "card_id":{"type":"string"},
                "item_id":{"type":"string"},
                "updates":{"type":"object","properties":{
                    "description":{"type":"string"},
                    "quantity":{"type":"number"},
                    "amount":{"type":"number"},
                    "date":{"type":"string","format":"date"}
                }}
            },"required":["card_id","item_id","updates"]}}}
    ]

# ---------- OpenAI Assistant (stateless calls) ----------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = client.beta.threads.create().id




# (Re)create a fresh thread if missing
if "thread_id" not in st.session_state:
    st.session_state.thread_id = client.beta.threads.create().id
if "chat" not in st.session_state:
    st.session_state.chat = []



def audio_to_wav_bytes(seg) -> bytes:
    mono16 = seg.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
        wf.writeframes(mono16.raw_data)
    return buf.getvalue()

def transcribe(wav: bytes) -> str:
    kwargs = {"model": TRANSCRIBE_MODEL, "file": ("speech.wav", wav, "audio/wav")}
    if TRANSCRIBE_LANGUAGE: kwargs["language"] = TRANSCRIBE_LANGUAGE
    tr = client.audio.transcriptions.create(**kwargs)
    return (tr.text or "").strip()

def normalize_card_id(v: str) -> str:
    s = str(v).strip().upper()
    m = re.search(r'(\d+)$', s)
    if m:
        n = m.group(1).zfill(3 if int(m.group(1)) <= 999 else 4)
        return f"C_{n}"
    return s

# ---------- Tool caller bridging to backend ----------
def call_tool(name, args):
    if name == "create_card":
        r = requests.post(f"{BACKEND_URL}/cards", json=args)
    elif name == "list_cards":
        r = requests.get(f"{BACKEND_URL}/cards", params=args)
    elif name == "update_card":
        args = dict(args)
        args["card_id"] = normalize_card_id(args["card_id"])
        r = requests.patch(f"{BACKEND_URL}/cards/{args.pop('card_id')}", json=args)
    elif name == "add_item":
        args = dict(args)
        cid = normalize_card_id(args.pop("card_id"))
        r = requests.post(f"{BACKEND_URL}/cards/{cid}/items", json=args)
    elif name == "list_items":
        cid = normalize_card_id(args["card_id"])
        r = requests.get(f"{BACKEND_URL}/cards/{cid}/items")
    elif name == "update_item":
        args = dict(args)
        cid = normalize_card_id(args.pop("card_id"))
        iid = args.pop("item_id")
        r = requests.patch(f"{BACKEND_URL}/cards/{cid}/items/{iid}", json=args)
    else:
        return None
    return r.text if r is not None else None

def run_assistant(user_text: str):
    # Create a transient assistant each run to keep code simple and stateless
    a = client.beta.assistants.create(
        name="Cards Assistant",
        instructions=SYSTEM_PROMPT,
        model=ASST_MODEL,
        tools=tools_schema(),
    )
    client.beta.threads.messages.create(
        thread_id=st.session_state.thread_id, role="user", content=user_text
    )
    run = client.beta.threads.runs.create(
        thread_id=st.session_state.thread_id, assistant_id=a.id
    )
    run = client.beta.threads.runs.poll(thread_id=st.session_state.thread_id, run_id=run.id)

    if run.status == "requires_action":
        outs = []
        for tc in run.required_action.submit_tool_outputs.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            out = call_tool(name, args)
            outs.append({"tool_call_id": tc.id, "output": out or ""})
        run = client.beta.threads.runs.submit_tool_outputs(
            thread_id=st.session_state.thread_id, run_id=run.id, tool_outputs=outs
        )
        run = client.beta.threads.runs.poll(thread_id=st.session_state.thread_id, run_id=run.id)

    msgs = client.beta.threads.messages.list(thread_id=st.session_state.thread_id)
    ai_msgs = [m for m in msgs.data if m.role == "assistant" and m.content]
    return (ai_msgs[0].content[0].text.value if ai_msgs else "OK")

# ---------- UI (clean, no sidebar) ----------
st.title("Plastic Manager — Cards")
st.caption("Create and manage cards & items (voice or text).")

# --- Chat history ---
if "chat" not in st.session_state:
    st.session_state.chat = []
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- Reset button (optional) ---
left, _ = st.columns([0.2, 0.8])
with left:
    if st.button("Reset session"):
        st.session_state.pop("thread_id", None)
        st.session_state.chat = []
        st.session_state.pop("pending", None)
        st.rerun()

# --- Chat input (single input; optimistic render) ---
user_msg = st.chat_input("Type a message…")
if user_msg:
    st.session_state.chat.append({"role": "user", "content": user_msg})
    st.session_state["pending"] = {"type": "text", "payload": user_msg}
    st.rerun()

# --- Mic (records, then transcribe -> optimistic render) ---
audio = audiorecorder("🎙️ Start", "⏹️ Stop", key="mic")
if len(audio) > 0:
    wav = audio_to_wav_bytes(audio)
    h = hashlib.md5(wav).hexdigest()
    if h != st.session_state.get("last_audio_hash"):
        st.session_state["last_audio_hash"] = h
        try:
            text = transcribe(wav)
            if text:
                st.session_state.chat.append({"role": "user", "content": text})
                st.session_state["pending"] = {"type": "voice", "payload": text}
                st.rerun()
        except Exception as e:
            st.error(f"Transcription failed: {e}")

# --- Process pending after optimistic render ---
if st.session_state.get("pending"):
    to_process = st.session_state.pop("pending")["payload"]
    reply = run_assistant(to_process)
    st.session_state.chat.append({"role": "assistant", "content": reply})
    st.rerun()



