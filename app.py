import os
import io
import json
import tempfile
from datetime import datetime, timedelta
from dateutil import tz, parser as dtparser
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import plotly.express as px
import soundfile as sf
import streamlit as st
from streamlit_mic_recorder import mic_recorder
from openai import OpenAI

# ----------------------------
# Config & globals
# ----------------------------
st.set_page_config(page_title="Voice → Gantt", layout="wide")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
APP_TZ = tz.gettz(st.secrets.get("TZ", "Asia/Makassar"))
TODAY = datetime.now(APP_TZ).date()

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY. Set it in .streamlit/secrets.toml")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------
# Demo data loader (adapter)
# Replace this with your zip’s loading function if available.
# ----------------------------
def load_plan_dataframe():
    """
    Expected columns: id, start, finish, resource
    start/finish must be ISO strings or pandas Timestamps (timezone-aware).
    """
    if "plan_df" in st.session_state:
        return st.session_state.plan_df

    # Try to load from a known CSV path to match your project layout,
    # otherwise seed a tiny demo.
    csv_candidates = [
        "data/orders.csv",
        "orders.csv",
    ]
    for path in csv_candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    else:
        base = datetime.combine(TODAY, datetime.min.time()).replace(tzinfo=APP_TZ)
        df = pd.DataFrame([
            {"id": "O021", "start": base + timedelta(hours=8),  "finish": base + timedelta(hours=12), "resource": "Line A"},
            {"id": "O022", "start": base + timedelta(hours=13), "finish": base + timedelta(hours=18), "resource": "Line A"},
            {"id": "O023", "start": base + timedelta(hours=9),  "finish": base + timedelta(hours=15), "resource": "Line B"},
            {"id": "O024", "start": base + timedelta(days=1, hours=8),  "finish": base + timedelta(days=1, hours=12), "resource": "Line B"},
        ])

    # Normalize types
    for col in ["start", "finish"]:
        if df[col].dtype == object:
            df[col] = pd.to_datetime(df[col]).dt.tz_localize(APP_TZ, nonexistent="shift_forward", ambiguous="NaT", errors="coerce").fillna(
                pd.to_datetime(df[col], utc=True).dt.tz_convert(APP_TZ)
            )
        else:
            # assume naive -> localize
            df[col] = pd.to_datetime(df[col]).dt.tz_localize(APP_TZ, nonexistent="shift_forward", ambiguous="NaT", errors="coerce")

    st.session_state.plan_df = df
    return df

# ----------------------------
# Undo/Redo stacks
# ----------------------------
def init_history():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "future" not in st.session_state:
        st.session_state.future = []

def snapshot():
    # Store a deep copy (as json-serializable)
    df = st.session_state.plan_df.copy()
    payload = df.to_dict(orient="records")
    st.session_state.history.append(payload)
    st.session_state.future.clear()

def restore_from(payload):
    st.session_state.plan_df = pd.DataFrame(payload)
    # ensure tz-aware
    for col in ["start", "finish"]:
        st.session_state.plan_df[col] = pd.to_datetime(st.session_state.plan_df[col]).dt.tz_convert(APP_TZ)

def do_undo(steps=1):
    if not st.session_state.history:
        return False
    current = st.session_state.plan_df.to_dict(orient="records")
    for _ in range(steps):
        if not st.session_state.history: break
        prev = st.session_state.history.pop()
        st.session_state.future.append(current)
        restore_from(prev)
        current = prev
    return True

def do_redo(steps=1):
    if not st.session_state.future:
        return False
    current = st.session_state.plan_df.to_dict(orient="records")
    for _ in range(steps):
        if not st.session_state.future: break
        nxt = st.session_state.future.pop()
        st.session_state.history.append(current)
        restore_from(nxt)
        current = nxt
    return True

# ----------------------------
# Scheduling operations (adapter to your existing logic)
# ----------------------------
def move_order_by(order_id, delta_days=0, delta_hours=0, delta_minutes=0):
    df = st.session_state.plan_df
    mask = df["id"] == order_id
    if not mask.any():
        raise ValueError(f"Order {order_id} not found.")
    delta = relativedelta(days=delta_days, hours=delta_hours, minutes=delta_minutes)
    df.loc[mask, "start"] = df.loc[mask, "start"].apply(lambda x: x + delta)
    df.loc[mask, "finish"] = df.loc[mask, "finish"].apply(lambda x: x + delta)

def move_order_to(order_id, new_start_iso):
    df = st.session_state.plan_df
    mask = df["id"] == order_id
    if not mask.any():
        raise ValueError(f"Order {order_id} not found.")
    new_start = dtparser.isoparse(new_start_iso).astimezone(APP_TZ)
    durations = (df.loc[mask, "finish"] - df.loc[mask, "start"]).dt.total_seconds()
    duration = timedelta(seconds=float(durations.iloc[0]))
    df.loc[mask, "start"] = new_start
    df.loc[mask, "finish"] = new_start + duration

def shift_range_by(filter_dict, delta_days=0, delta_hours=0, delta_minutes=0):
    df = st.session_state.plan_df
    m = pd.Series(True, index=df.index)
    if filter_dict.get("orderIds"):
        m &= df["id"].isin(filter_dict["orderIds"])
    if filter_dict.get("resourceIds"):
        m &= df["resource"].isin(filter_dict["resourceIds"])
    if filter_dict.get("dateRange"):
        s = dtparser.isoparse(filter_dict["dateRange"]["startISO"]).astimezone(APP_TZ)
        e = dtparser.isoparse(filter_dict["dateRange"]["endISO"]).astimezone(APP_TZ)
        m &= (df["start"] < e) & (df["finish"] > s)
    if not m.any():
        raise ValueError("No tasks match the given range filter.")
    delta = relativedelta(days=delta_days, hours=delta_hours, minutes=delta_minutes)
    df.loc[m, "start"] = df.loc[m, "start"].apply(lambda x: x + delta)
    df.loc[m, "finish"] = df.loc[m, "finish"].apply(lambda x: x + delta)

# ----------------------------
# OpenAI prompts & tool schema
# ----------------------------
SYSTEM_PROMPT = """
You convert user scheduling commands into STRICT JSON operations for a Gantt planner.
You MUST return ONLY a JSON object with an "operations" array. Do not add prose.
Timezone: Asia/Makassar. Today is {today}.

Allowed operation shapes:
1) move_order_by:
{{
  "type": "move_order_by",
  "orderId": "O023",
  "delta": {{"days": 1, "hours": 0, "minutes": 0}}
}}
2) move_order_to:
{{
  "type": "move_order_to",
  "orderId": "O023",
  "start": "2025-08-25T09:00:00+08:00"
}}
3) shift_range_by:
{{
  "type": "shift_range_by",
  "filter": {{
    "resourceIds": ["Line A"],
    "orderIds": ["O021","O022"],
    "dateRange": {{"startISO":"2025-08-22T00:00:00+08:00","endISO":"2025-08-22T23:59:59+08:00"}}
  }},
  "delta": {{"hours": 2}}
}}
4) undo:
{{ "type": "undo", "steps": 1 }}
5) redo:
{{ "type": "redo", "steps": 1 }}

Rules:
- Resolve relative dates like "tomorrow" using the timezone above.
- Only reference known order IDs/resources when possible (see context).
- If the user requests a single shift like "move O023 by one day", produce a single operation.
- Keep numbers integers where possible.
"""

def build_context_snapshot(df: pd.DataFrame):
    # Small index of known entities to guide the model
    resources = sorted(df["resource"].dropna().unique().tolist())
    orders = df["id"].tolist()
    bounds = {
        "minStart": df["start"].min().isoformat(),
        "maxFinish": df["finish"].max().isoformat(),
    }
    return {"orders": orders, "resources": resources, "bounds": bounds, "today": str(TODAY)}

def interpret_with_llm(transcript: str, context_snapshot: dict) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(today=str(TODAY))},
        {"role": "user", "content": f"Context: {json.dumps(context_snapshot)}"},
        {"role": "user", "content": f'Command: """{transcript}"""'},
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    data = json.loads(content)
    ops = data.get("operations", [])
    if not isinstance(ops, list):
        raise ValueError("LLM did not return a valid operations array.")
    return ops

# ----------------------------
# Whisper transcription
# ----------------------------
def transcribe_with_whisper(wav_bytes: bytes, language_hint: str | None = None) -> dict:
    # Save to a NamedTemporaryFile for the SDK
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp.flush()
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            tr = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=language_hint  # None lets Whisper autodetect
            )
        return {"text": tr.text}
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# ----------------------------
# UI helpers
# ----------------------------
def render_gantt(df: pd.DataFrame):
    fig = px.timeline(
        df.sort_values("start"),
        x_start="start",
        x_end="finish",
        y="resource",
        color="id",
        hover_data=["id", "resource", "start", "finish"],
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

def apply_operations(ops: list[dict]) -> list[str]:
    """
    Apply a list of operations atomically (simple version).
    Returns a list of human-readable summaries for the toast/log.
    """
    if not ops:
        return ["No operations to apply."]
    # Snapshot for undo
    snapshot()
    msgs = []
    try:
        for op in ops:
            t = op.get("type")
            if t == "move_order_by":
                oid = op["orderId"]
                delta = op.get("delta", {})
                move_order_by(oid,
                              delta_days=int(delta.get("days", 0)),
                              delta_hours=int(delta.get("hours", 0)),
                              delta_minutes=int(delta.get("minutes", 0)))
                msgs.append(f"Moved {oid} by {delta.get('days',0)}d {delta.get('hours',0)}h {delta.get('minutes',0)}m")

            elif t == "move_order_to":
                oid = op["orderId"]
                start_iso = op["start"]
                move_order_to(oid, start_iso)
                msgs.append(f"Moved {oid} to {start_iso}")

            elif t == "shift_range_by":
                fdict = op.get("filter", {})
                delta = op.get("delta", {})
                shift_range_by(fdict,
                               delta_days=int(delta.get("days", 0)),
                               delta_hours=int(delta.get("hours", 0)),
                               delta_minutes=int(delta.get("minutes", 0)))
                msgs.append("Shifted range by "
                            f"{delta.get('days',0)}d {delta.get('hours',0)}h {delta.get('minutes',0)}m")

            elif t == "undo":
                steps = int(op.get("steps", 1))
                do_undo(steps)
                msgs.append(f"Undid {steps} step(s)")

            elif t == "redo":
                steps = int(op.get("steps", 1))
                do_redo(steps)
                msgs.append(f"Redid {steps} step(s)")

            else:
                raise ValueError(f"Unknown operation type: {t}")
        return msgs
    except Exception as e:
        # Roll back if something fails by undoing the snapshot we just pushed
        do_undo(1)
        raise e

# ----------------------------
# App
# ----------------------------
st.title("🎙️ Voice → Gantt (Whisper + LLM)")

with st.sidebar:
    st.markdown("### Voice settings")
    lang = st.selectbox("Language hint (optional)", ["auto", "en", "id", "de", "fr", "es"], index=0)
    st.caption("If set to 'auto', Whisper will detect the language.")
    st.divider()
    if st.button("Undo"):
        if do_undo(1):
            st.success("Undid last change.")
        else:
            st.info("Nothing to undo.")
    if st.button("Redo"):
        if do_redo(1):
            st.success("Redid change.")
        else:
            st.info("Nothing to redo.")
    st.divider()
    st.markdown("**What can I say?**")
    st.code('''
"move Order O023 by one day"
"move O023 to Aug 25 09:00"
"shift all on Line A today by 2 hours"
"undo"
'''.strip())

init_history()
df = load_plan_dataframe()
render_gantt(df)

st.markdown("#### Speak a command")
st.caption("Click to record, click again to stop. Review the transcript, then Apply.")
audio = mic_recorder(
    start_prompt="Start recording",
    stop_prompt="Stop",
    just_once=False,
    use_container_width=True,
    format="wav",
    key="mic",
)

col1, col2 = st.columns([1,1])

with col1:
    transcript_area = st.empty()
with col2:
    ops_area = st.empty()

transcript_text = ""

if audio and isinstance(audio, dict) and audio.get("bytes"):
    # audio["bytes"] is raw WAV bytes from the component
    wav_bytes = audio["bytes"]

    # Ensure valid WAV by decoding/encoding once via soundfile
    data, samplerate = sf.read(io.BytesIO(wav_bytes))
    buf = io.BytesIO()
    sf.write(buf, data, samplerate, format="WAV")
    clean_wav_bytes = buf.getvalue()

    with st.spinner("Transcribing with Whisper..."):
        tr = transcribe_with_whisper(clean_wav_bytes, None if lang == "auto" else lang)
        transcript_text = tr["text"].strip()
        transcript_area.text_area("Transcript", transcript_text, height=120)

    if transcript_text:
        with st.spinner("Interpreting command..."):
            ctx = build_context_snapshot(st.session_state.plan_df)
            try:
                ops = interpret_with_llm(transcript_text, ctx)
                ops_area.json({"operations": ops})
                if st.button("Apply to plan", type="primary"):
                    try:
                        msgs = apply_operations(ops)
                        st.success(" | ".join(msgs))
                        render_gantt(st.session_state.plan_df)
                    except Exception as e:
                        st.error(str(e))
            except Exception as e:
                st.error(f"Could not interpret the command: {e}")

# Manual testing without mic
st.markdown("#### Or type a command (debug)")
manual = st.text_input("Try: move O023 by one day")
if st.button("Interpret (typed)"):
    if not manual.strip():
        st.warning("Type something first.")
    else:
        ctx = build_context_snapshot(st.session_state.plan_df)
        try:
            ops = interpret_with_llm(manual.strip(), ctx)
            ops_area.json({"operations": ops})
        except Exception as e:
            st.error(f"Could not interpret: {e}")

