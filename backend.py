# backend.py (Flask API to replace app_cards_voice.py logic)
import os, json, hashlib, re, requests
from pathlib import Path
from dotenv import load_dotenv, dotenv_values
from openai import OpenAI, BadRequestError
from flask import Flask, request, jsonify, send_from_directory
from functools import wraps
from werkzeug.utils import secure_filename
import base64
import io

# --- 0. Config & Setup (from app_cards_voice.py) ---
# Load .env
ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(f"Missing OPENAI_API_KEY. Expected at: {ENV_PATH}")

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5000")
REQ_TIMEOUT = 30
ASST_MODEL = os.getenv("ASST_MODEL", "gpt-4o")
TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
TRANSCRIBE_LANGUAGE = (os.getenv("TRANSCRIBE_LANGUAGE") or "").strip()
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Initializing Assistant/Thread (Global State Management) ---
# NOTE: In a real-world app, thread management should be persistent (e.g., DB)
GLOBAL_STATE = {}

def initialize_assistant():
    if "asst_id" not in GLOBAL_STATE:
        # Load tools_schema from the original app_cards_voice.py content
        def tools_schema():
            # ... (Content of tools_schema() from app_cards_voice.py goes here) ...
            return [
                {"type": "function", "function": {
                    "name": "create_card", "description": "Create a new card",
                    "parameters": {"type": "object", "properties": {
                        "business_partner": {"type": "string"},
                        "description": {"type": "string"},
                        "type": {"type": "string"},
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
                    "name": "get_card",
                    "description": "Fetch a single card by its ID, including all attributes",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "card_id": {
                                "type": "string",
                                "description": "Card ID such as 'C_002' or a user phrase like 'card 2' (normalize to C_002)."
                            }
                        },
                        "required": ["card_id"]
                    }
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

        # SYSTEM_PROMPT = """
        # You are a Cards Assistant for a simple finance tracker.
        # - Create/list/update cards via tools with fields: business_partner, description, type, quantity, amount, date.
        # - Create/list/update items on a card with fields: description, quantity, amount, date.
        # - Card IDs follow C_###; Item IDs follow Item_### (assigned by backend). If the user mentions card 3 he is referring to c_003, if he mentions card 15 it would correspond to c_015.
        # - You can resolve where queued attachments should be stored via the tool 'store_attachment_intent'.
        # - When the user asks to VIEW attachments (EN: attachments, files, docs; DE: Anhang, Anhänge, Datei, Dokumente, Beilagen, Attachments),
        #   you MUST call exactly one of: list_card_attachments(card_id) or list_item_attachments(card_id,item_id) BEFORE answering.
        #   Never infer attachment presence without a tool call.
        # - If you are not sure which card/item, ask one targeted question to disambiguate.
        # - Be concise and only ask for missing essentials.
        # - Always call tools when available.
        # - When the user says things like "upload now", "attach to C_013", or "bitte an C_13",
        #   call the tool commit_staged(card_id, [item_id]) using the current or specified target.
        # - Do not claim files are attached unless the commit_staged tool returns success.
        # - You MUST speak and respond ONLY in **English**. Do not use any other language for your output unless user is speaking or writing in FRENCH or GERMAN.
        #     - When the user refers to previously staged files (e.g. "upload now", "attach the screenshot", "bitte an Karte 5 anhängen"):
        #     - If exactly one target card is clear (e.g. "card 5" / "Karte 5"), call commit_staged(card_id="C_005", item_id if specified).
        #     - If the target card or item is ambiguous, ask a short clarification question.
        # - Assume that staged attachments are managed server-side; do not ask the user to repeat file names.
        # """


        SYSTEM_PROMPT = """
        You are a Cards Assistant for a simple finance tracker.

        CORE FUNCTIONALITY
        - Create, list and update cards via tools with fields: business_partner, description, type, quantity, amount, date, unit, currency.
        - Create, list and update items on a card with fields: description, quantity, amount, date.
        - Card IDs follow C_###; Item IDs follow Item_### (assigned by backend).
        If the user mentions "card 3" this corresponds to C_003; "card 15" corresponds to C_015.

        CARD VIEW / DETAILS BEHAVIOUR
        - When the user asks to SEE or SHOW a specific card (e.g. "show card 2", "zeige Karte 2", "montre la carte 2"):
        1) Normalize the mentioned card number to a proper card ID (e.g. "card 2" -> "C_002").
        2) Retrieve that card using the available tools (e.g. get_card or list_cards).
        3) You MUST show ALL attributes of the card itself, at least:
            - card_id
            - business_partner
            - description
            - type
            - amount and currency
            - quantity and unit
            - date
            - and any other stored fields present in the tool output.
        4) Even if there are no items or attachments, ALWAYS show the card attributes.

        - When the user asks for DETAILS, ALL DETAILS, or a FULL OVERVIEW of a card or a list of cards
        (e.g. "show all details of card 2", "details for cards 2 and 3",
        "alle Details von Karte 2", "tous les détails de la carte 2"):
        1) For each requested card, first show ALL attributes of the card as above.
        2) Additionally, for each card:
            - Call list_items(card_id) and provide an overview of items:
            * If there are items, summarise them (count and key fields per item).
            * If there are no items, explicitly state that there are no items.
            - Call list_card_attachments(card_id) and provide an overview of attachments:
            * If there are attachments, summarise them (count and filenames or identifiers).
            * If there are no attachments, explicitly state that there are no attachments.
        3) Never answer with only "no items" or "no attachments" when the user requested a card or card details;
            always include the card attributes as well.

        ATTACHMENTS
        - The user can upload and stage files. You do NOT see the raw files, only metadata via tools.
        - When the user asks to VIEW attachments
        * English: attachments, attachment, files, file, docs, documents
        * German: Anhang, Anhänge, Datei, Dateien, Dokument, Dokumente, Beilage, Beilagen, Attachments
        * French: pièce jointe, pièces jointes, fichier, fichiers, document, documents
        you MUST call exactly one of:
            * list_card_attachments(card_id)
            * list_item_attachments(card_id, item_id)
        BEFORE answering. Never assume attachments exist without a tool call.

        - When the user refers to previously staged files (e.g. "upload now", "attach the screenshot",
        "bitte an Karte 5 anhängen", "ajoute la capture à la carte 5"):
        * If exactly one target card is clear (e.g. "card 5" / "Karte 5" / "carte 5"),
            call commit_staged(card_id="C_005", and item_id if specified).
        * If the target card or item is ambiguous, ask ONE short clarification question.
        - Assume that staged attachments are managed server-side; do not ask the user to repeat file names.

        ERROR HANDLING
        - Tool outputs may contain fields like "status_code" and "error".
        - If a tool output indicates:
        * status_code = 404 or an error such as "card not found":
            - Clearly tell the user that the requested card or item does not exist.
            - Do NOT call this a technical problem.
        * status_code = 500 or an error like "Failed to connect to Card Backend":
            - You may mention a technical problem, but be brief and factual.
        - Never invent technical errors. Only mention a technical problem if it is clearly shown by the tool output.

        LANGUAGE BEHAVIOUR
        - You MUST speak and respond ONLY in English, French or German.
        - Always respond in the SAME language as the last user message:
        * English user message -> respond in English.
        * German user message -> respond in German.
        * French user message -> respond in French.
        - Do NOT translate the user's content unless explicitly asked.
        - Do NOT switch languages on your own; follow the user's choice.

        GENERAL STYLE
        - Be concise and business-like, but friendly.
        - Ask only for missing essentials when necessary (e.g. missing card_id, ambiguous target card).
        - Always use the available tools (create_card, list_cards, get_card, update_card, add_item, list_items,
        update_item, store_attachment_intent, list_card_attachments, list_item_attachments, commit_staged)
        instead of inventing data.
        """





        a = client.beta.assistants.create(
            name="Cards Assistant",
            instructions=SYSTEM_PROMPT,
            model=os.getenv("ASST_MODEL", "gpt-4o-mini"),
            tools=tools_schema(),
        )
        GLOBAL_STATE["asst_id"] = a.id
        GLOBAL_STATE["thread_id"] = client.beta.threads.create().id

# --- Helper functions (from app_cards_voice.py) ---
def _q(s: str) -> str:
    return requests.utils.quote(s or "", safe="")

def normalize_card_id(v: str) -> str:
    s = str(v).strip().upper()
    m = re.search(r'(\d+)$', s)
    if m:
        n = m.group(1)
        pad = 3 if int(n) <= 999 else 4
        return f"C_{n.zfill(pad)}"
    return s

def _drain_active_runs(thread_id: str, call_tool_func):
    """
    Handles queued/in_progress/requires_action until all runs are terminal.
    NOTE: Simplified for Flask, assuming no concurrent access.
    """
    try:
        runs = client.beta.threads.runs.list(thread_id=thread_id).data
    except Exception:
        runs = []

    ids = [r.id for r in runs]

    for run_id in ids:
        while True:
            run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
            status = getattr(run, "status", "")
            if status in ("queued", "in_progress"):
                import time; time.sleep(0.5) # Wait a bit for non-poll method
                continue
            if status == "requires_action":
                outs = []
                for tc in run.required_action.submit_tool_outputs.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments or "{}")
                    out = call_tool_func(name, args)
                    outs.append({"tool_call_id": tc.id, "output": out or ""})
                run = client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id, run_id=run_id, tool_outputs=outs
                )
                continue
            break


def call_tool(name, args):
    """
    Executes a tool call. Unlike Streamlit, this returns the output string directly.
    The side effects (like updating context) are handled in the front-end logic.
    """
    if name == "create_card":
        r = requests.post(f"{BACKEND_URL}/cards", json=args, timeout=REQ_TIMEOUT)
    elif name == "list_cards":
        r = requests.get(f"{BACKEND_URL}/cards", params=args, timeout=REQ_TIMEOUT)
    elif name == "get_card":
        cid = normalize_card_id(args["card_id"])
        r = requests.get(f"{BACKEND_URL}/cards/{_q(cid)}", timeout=REQ_TIMEOUT)
        print("DEBUG get_card:", cid, r.status_code, r.text[:200])
    elif name == "update_card":
        body = dict(args)
        cid = normalize_card_id(body.pop("card_id"))
        r = requests.patch(f"{BACKEND_URL}/cards/{_q(cid)}", json=body, timeout=REQ_TIMEOUT)
    elif name == "add_item":
        body = dict(args)
        cid = normalize_card_id(body.pop("card_id"))
        r = requests.post(f"{BACKEND_URL}/cards/{_q(cid)}/items", json=body, timeout=REQ_TIMEOUT)
    elif name == "list_items":
        cid = normalize_card_id(args["card_id"])
        r = requests.get(f"{BACKEND_URL}/cards/{_q(cid)}/items", timeout=REQ_TIMEOUT)
    elif name == "update_item":
        body = dict(args)
        cid = normalize_card_id(body.pop("card_id"))
        iid = body.pop("item_id")
        r = requests.patch(f"{BACKEND_URL}/cards/{_q(cid)}/items/{_q(iid)}", json=body, timeout=REQ_TIMEOUT)
    elif name == "store_attachment_intent":
        # This function only signals the target. The actual staging/commit is frontend-controlled.
        cid = normalize_card_id(args["card_id"])
        return json.dumps({"status": "target_set", "card_id": cid, "item_id": args.get("item_id")})
    
    elif name == "list_card_attachments":
        cid = normalize_card_id(args["card_id"])
        resp = requests.get(f"{BACKEND_URL}/cards/{_q(cid)}/attachments", timeout=REQ_TIMEOUT)
        data = resp.json() if resp.status_code == 200 else []
        for a in data:
            a.setdefault("url", f"{BACKEND_URL.rstrip('/')}{a.get('path','')}")
        # The frontend will use the `attachments` data to render the UI, not the LLM's text.
        return json.dumps({"card_id": cid, "count": len(data), "attachments": data})

    elif name == "list_item_attachments":
        cid = normalize_card_id(args["card_id"]); iid = args["item_id"]
        resp = requests.get(f"{BACKEND_URL}/cards/{_q(cid)}/items/{_q(iid)}/attachments", timeout=REQ_TIMEOUT)
        data = resp.json() if resp.status_code == 200 else []
        for a in data:
            a.setdefault("url", f"{BACKEND_URL.rstrip('/')}{a.get('path','')}")
        return json.dumps({"card_id": cid, "item_id": iid, "count": len(data), "attachments": data})

    elif name == "commit_staged":
        # args: { card_id: "...", item_id: "..." (optional) }
        tid = GLOBAL_STATE["thread_id"]
        staged = GLOBAL_STATE.get("staged_tokens", {}).get(tid, [])

        results = []
        for s in staged:
            payload = {
                "token": s["token"],
                "card_id": normalize_card_id(args["card_id"]),
            }
            if args.get("item_id"):
                payload["item_id"] = args["item_id"]
            r = requests.post(f"{BACKEND_URL}/attachments/commit", json=payload, timeout=REQ_TIMEOUT)
            r.raise_for_status()
            results.append(r.json())

        # Clear staged tokens for this thread after commit
        GLOBAL_STATE["staged_tokens"][tid] = []

        return json.dumps({"status": "committed", "results": results})


    else:
        return None

    try:
        r.raise_for_status()
        return r.text
    except requests.exceptions.HTTPError:
        return json.dumps({"error": r.text, "status_code": r.status_code})
    except Exception:
        return json.dumps({"error": "Failed to connect to Card Backend.", "status_code": 500})


# --- 1. Flask App ---
app = Flask(__name__, static_folder=".") # Serve files from current directory

# Allow Cross-Origin for local testing (remove in production)
from flask_cors import CORS; CORS(app)

@app.before_request
def before_request_func():
    initialize_assistant()

@app.route("/<filename>")
def serve_static_file(filename):
    """
    Serves files (like styles.css) that are in the same directory as index.html
    and are requested directly by the browser (e.g., /styles.css).
    """
    # The "." is safe because we restricted static_folder to the current directory earlier, 
    # but we explicitly tell it to look in the current directory anyway.
    return send_from_directory(".", filename)


@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/api/chat", methods=["POST"])
def handle_chat():
    tid = GLOBAL_STATE["thread_id"]
    data = request.get_json()
    user_text = data.get("message")
    
    if not user_text:
        return jsonify({"error": "Missing message"}), 400

    # 1. Add user message
    try:
        client.beta.threads.messages.create(thread_id=tid, role="user", content=user_text)
    except BadRequestError:
        _drain_active_runs(tid, call_tool) # Retry if a run is active
        client.beta.threads.messages.create(thread_id=tid, role="user", content=user_text)

    # 2. Start and run
    run = client.beta.threads.runs.create(thread_id=tid, assistant_id=GLOBAL_STATE["asst_id"])

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
            import time; time.sleep(0.5)
            continue
        break
    
    # 3. Get reply
    msgs = client.beta.threads.messages.list(thread_id=tid)
    ai_msgs = [m for m in msgs.data if m.role == "assistant" and m.content]
    reply = (ai_msgs[0].content[0].text.value if ai_msgs else "OK")
    
    return jsonify({"reply": reply})

@app.route("/api/transcribe", methods=["POST"])
def handle_transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]

    try:
        audio_bytes = audio_file.read()
        # print("DEBUG: received audio bytes:", len(audio_bytes))
        # with open("debug_input.wav", "wb") as f:
        #     f.write(audio_bytes)
        # print("DEBUG: saved debug_input.wav")

        # Use a BytesIO buffer as file-like object
        buf = io.BytesIO(audio_bytes)
        buf.name = "speech.wav"

        # kwargs = {
        #     "model": TRANSCRIBE_MODEL,   # gpt-4o-mini-transcribe
        #     "file": buf,
        #     "temperature": 0,
        #     "prompt": (
        #         "User speaks short English.French or German commands to a Plastic manager card app. "
        #         "Typical phrases: 'show card 5', 'show card five', 'open card 3', "
        #         "'list cards', 'create card', 'delete card 2'. "
        #         "Transcribe exactly what the user says."
        #     ),
        # # }

        # if TRANSCRIBE_LANGUAGE:
        #     kwargs["language"] = TRANSCRIBE_LANGUAGE  # 'en' in your .env
        # tr = client.audio.transcriptions.create(**kwargs)

        tr = client.audio.transcriptions.create(
            model=TRANSCRIBE_MODEL,          # whisper-1
            file=buf,
            language=TRANSCRIBE_LANGUAGE or None,
        )
        text = (tr.text or "").strip()

        print(f"DEBUG: RAW TRANSCRIPTION TEXT: {repr(text)}")
        return jsonify({"text": text})
    except Exception as e:
        print("Transcription error:", e)
        return jsonify({"error": str(e)}), 500


GLOBAL_STATE.setdefault("staged_tokens", {})  # per-thread mapping

@app.route("/api/register_staged", methods=["POST"])
def register_staged():
    data = request.get_json()
    tokens = data.get("tokens", [])
    filenames = data.get("filenames", [])
    tid = GLOBAL_STATE["thread_id"]

    # Store tokens server-side per thread
    GLOBAL_STATE["staged_tokens"].setdefault(tid, [])
    for t, fn in zip(tokens, filenames):
        GLOBAL_STATE["staged_tokens"][tid].append({"token": t, "filename": fn})

    # Optionally: add a message to the thread so LLM "knows" what’s available
    info = ", ".join(fn for fn in filenames)
    client.beta.threads.messages.create(
        thread_id=tid,
        role="user",
        content=f"[SYSTEM INFO] Staged attachments: {info}."
    )

    return jsonify({"status": "ok"})


# --- Helper endpoint for token commit (needed by the frontend) ---
@app.route("/api/commit_staged", methods=["POST"])
def commit_staged_api():
    """Client-side token commitment."""
    data = request.get_json()
    tokens = data.get("tokens", [])
    card_id = data.get("card_id")
    item_id = data.get("item_id")

    if not card_id or not tokens:
        return jsonify({"error": "Missing card_id or tokens"}), 400

    cid = normalize_card_id(card_id)
    results = []
    
    for t in tokens:
        payload = {"token": t, "card_id": cid}
        if item_id: payload["item_id"] = item_id
        try:
            r = requests.post(f"{BACKEND_URL}/attachments/commit", json=payload, timeout=REQ_TIMEOUT)
            r.raise_for_status()
            results.append(r.json())
        except requests.exceptions.RequestException as e:
            results.append({"token": t, "error": str(e), "status_code": getattr(r, 'status_code', 500)})

    # The frontend is responsible for clearing its local staged tokens.
    return jsonify({"status": "committed", "results": results})


if __name__ == "__main__":
    print("Starting Flask server...")
    print("Access the chat at: http://127.0.0.1:5001/")
    # NOTE: Running on port 5001 to avoid collision with the backend URL's default 5000
    app.run(port=5001, debug=True)