import os
import io
import json
import time
import random
import hashlib
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
from streamlit_plotly_events import plotly_events
from openai import OpenAI, RateLimitError

# =========================
# Config & globals
# =========================
st.set_page_config(page_title="FlowKa UC1 – Voice Gantt", layout="wide")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
APP_TZ = tz.gettz(st.secrets.get("TZ", "Asia/Makassar"))
TODAY = datetime.now(APP_TZ).date()

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY. Set it in .streamlit/secrets.toml")
    st.stop()

@st.cache_resource(show_spinner=False)
def get_openai_client(key: str):
    return OpenAI(api_key=key)

client = get_openai_client(OPENAI_API_KEY)

# Cooldown gate
COOLDOWN_SECONDS = 2.5
if "last_api_call_ts" not in st.session_state:
    st.session_state.last_api_call_ts = 0.0

def gate_api_call() -> bool:
    now = time.time()
    if now - st.session_state.last_api_call_ts < COOLDOWN_SECONDS:
        return False
    st.session_state.last_api_call_ts = now
    return True

# UI state
st.session_state.setdefault("show_filters", False)
st.session_state.setdefault("selected_group", None)  # order/project group
st.session_state.setdefault("transcript", "")
st.session_state.setdefault("ops_preview", None)
st.session_state.setdefault("audio_bytes", None)
st.session_state.setdefault("audio_hash", None)

# =========================
# Retry helper (exponential backoff + jitter)
# =========================
def with_retries(fn, *args, _max_attempts=5, _base=1.3, _jitter=0.5, **kwargs):
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except RateLimitError:
            attempt += 1
            if attempt >= _max_attempts:
                raise
            delay = _base * (2 ** (attempt - 1)) + random.uniform(0, _jitter)
            st.toast(f"Rate limited. Retrying in {delay:.1f}s…", icon="⌛")
            time.sleep(delay)
        except Exception as e:
            msg = str(e).lower()
            transient = any(t in msg for t in ["timeout", "temporarily", "server error", "503", "502", "504"])
            if not transient:
                raise
            attempt += 1
            if attempt >= _max_attempts:
                raise
            delay = _base * (2 ** (attempt - 1)) + random.uniform(0, _jitter)
            st.toast(f"Temporary error. Retrying in {delay:.1f}s…", icon="↻")
            time.sleep(delay)

# =========================
# Time helpers
# =========================
def ensure_app_tz(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    if getattr(s.dt, "tz", None) is None:
        s = s.dt.tz_localize(APP_TZ, nonexistent="shift_forward", ambiguous="NaT")
    else:
        s = s.dt.tz_convert(APP_TZ)
    return s

# =========================
# Data load (UC1 scooter)
# =========================
ALIASES = {
    "order_id": ["order_id", "order", "project", "job", "id", "ORDER_ID", "Order", "Project"],
    "operation_id": ["operation_id", "operation", "op", "OPERATION_ID"],
    "resource": ["resource", "machine", "line", "workcenter", "RESOURCE", "Machine", "Line"],
    "start": ["start", "start_time", "start_dt", "START", "Start"],
    "finish": ["finish", "end", "end_time", "finish_time", "FINISH", "End"],
}

def find_col(df, keys):
    for k in keys:
        if k in df.columns:
            return k
    return None

def load_uc1_scooter():
    # Try the scooter files first
    candidates = [
        ("data/scooter_schedule.csv", None),
        ("data/scooter_orders.csv", None),
        ("scooter_schedule.csv", None),
        ("scooter_orders.csv", None),
    ]
    found = [p for p, _ in candidates if os.path.exists(p)]
    if not found:
        # fallback small seed (so the app still opens)
        base = datetime.combine(TODAY, datetime.min.time()).replace(tzinfo=APP_TZ)
        df = pd.DataFrame([
            {"order_id": "O021", "operation_id": "Op1", "start": base + timedelta(hours=8),  "finish": base + timedelta(hours=12), "resource": "Line A"},
            {"order_id": "O022", "operation_id": "Op1", "start": base + timedelta(hours=13), "finish": base + timedelta(hours=18), "resource": "Line A"},
            {"order_id": "O023", "operation_id": "Op1", "start": base + timedelta(hours=9),  "finish": base + timedelta(hours=15), "resource": "Line B"},
            {"order_id": "O024", "operation_id": "Op1", "start": base + timedelta(days=1, hours=8), "finish": base + timedelta(days=1, hours=12), "resource": "Line B"},
        ])
    else:
        # Load whichever exists; if both exist we prefer schedule
        primary = "data/scooter_schedule.csv" if os.path.exists("data/scooter_schedule.csv") else found[0]
        df = pd.read_csv(primary)

    # Map columns
    col_order = find_col(df, ALIASES["order_id"])
    col_op = find_col(df, ALIASES["operation_id"])
    col_res = find_col(df, ALIASES["resource"])
    col_start = find_col(df, ALIASES["start"])
    col_finish = find_col(df, ALIASES["finish"])

    if not all([col_order, col_res, col_start, col_finish]):
        raise ValueError(
            "Could not resolve required columns. "
            "Expected columns like: order_id/order/project, resource/machine, start/start_time, finish/end_time."
        )

    # Build normalized DF
    out = pd.DataFrame({
        "order_id": df[col_order].astype(str),
        "resource": df[col_res].astype(str),
        "start": ensure_app_tz(df[col_start]),
        "finish": ensure_app_tz(df[col_finish]),
    })
    if col_op:
        out["operation_id"] = df[col_op].astype(str)
    else:
        # fabricate operation id if not present
        out["operation_id"] = "Op"

    # Add display id for color & tooltips
    out["id"] = out["order_id"] + ":" + out["operation_id"].astype(str)

    # Sort
    out = out.sort_values(["start", "resource"]).reset_index(drop=True)
    return out

@st.cache_data(show_spinner=False)
def get_plan_df():
    return load_uc1_scooter()

def get_filtered_df(df: pd.DataFrame, res_sel, date_range):
    m = pd.Series(True, index=df.index)
    if res_sel:
        m &= df["resource"].isin(res_sel)
    if date_range and all(date_range):
        s = datetime.combine(date_range[0], datetime.min.time(), tzinfo=APP_TZ)
        e = datetime.combine(date_range[1], datetime.max.time(), tzinfo=APP_TZ)
        m &= (df["start"] < e) & (df["finish"] > s)
    return df[m].copy()

# Keep working copy in session (for mutations)
def ensure_work_df():
    if "plan_df" not in st.session_state:
        st.session_state.plan_df = get_plan_df().copy()
    return st.session_state.plan_df

# =========================
# Undo / Redo
# =========================
def init_history():
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("future", [])

def snapshot():
    df = st.session_state.plan_df.copy()
    payload = df.to_dict(orient="records")
    st.session_state.history.append(payload)
    st.session_state.future.clear()

def restore_from(payload):
    df = pd.DataFrame(payload)
    for col in ["start", "finish"]:
        df[col] = ensure_app_tz(df[col])
    st.session_state.plan_df = df

def do_undo(steps=1):
    if not st.session_state.history:
        return False
    current = st.session_state.plan_df.to_dict(orient="records")
    for _ in range(steps):
        if not st.session_state.history:
            break
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
        if not st.session_state.future:
            break
        nxt = st.session_state.future.pop()
        st.session_state.history.append(current)
        restore_from(nxt)
        current = nxt
    return True

# =========================
# Scheduling ops (UC1 semantics)
# =========================
def move_order_by(order_id, delta_days=0, delta_hours=0, delta_minutes=0):
    df = st.session_state.plan_df
    m = df["order_id"] == str(order_id)
    if not m.any():
        raise ValueError(f"Order {order_id} not found.")
    delta = relativedelta(days=delta_days, hours=delta_hours, minutes=delta_minutes)
    df.loc[m, "start"] = df.loc[m, "start"].apply(lambda x: x + delta)
    df.loc[m, "finish"] = df.loc[m, "finish"].apply(lambda x: x + delta)

def move_order_to(order_id, new_start_iso):
    df = st.session_state.plan_df
    m = df["order_id"] == str(order_id)
    if not m.any():
        raise ValueError(f"Order {order_id} not found.")
    # Align whole order by delta preserving relative offsets between operations
    current_min = df.loc[m, "start"].min()
    new_start = dtparser.isoparse(new_start_iso).astimezone(APP_TZ)
    delta = new_start - current_min
    df.loc[m, "start"] = df.loc[m, "start"] + delta
    df.loc[m, "finish"] = df.loc[m, "finish"] + delta

def swap_orders(order_a, order_b):
    df = st.session_state.plan_df
    ma = df["order_id"] == str(order_a)
    mb = df["order_id"] == str(order_b)
    if not ma.any() or not mb.any():
        raise ValueError("Both orders must exist to swap.")
    # Compute anchor times
    a_min, a_max = df.loc[ma, "start"].min(), df.loc[ma, "finish"].max()
    b_min, b_max = df.loc[mb, "start"].min(), df.loc[mb, "finish"].max()
    # Swapping by translating sets to the other's anchor
    delta_a = b_min - a_min
    delta_b = a_min - b_min
    df.loc[ma, "start"] = df.loc[ma, "start"] + delta_a
    df.loc[ma, "finish"] = df.loc[ma, "finish"] + delta_a
    df.loc[mb, "start"] = df.loc[mb, "start"] + delta_b
    df.loc[mb, "finish"] = df.loc[mb, "finish"] + delta_b

def shift_range_by(filter_dict, delta_days=0, delta_hours=0, delta_minutes=0):
    df = st.session_state.plan_df
    m = pd.Series(True, index=df.index)
    if filter_dict.get("orderIds"):
        m &= df["order_id"].isin([str(x) for x in filter_dict["orderIds"]])
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

# =========================
# LLM interface (UC1 intents)
# =========================
SYSTEM_PROMPT = """
You are the command interpreter for a factory Gantt (UC1 scooter manufacturer).
Return ONLY JSON with an "operations" array. No prose. Timezone: Asia/Makassar. Today is {today}.

Allowed operations:
1) move_order_by:
{{
  "type": "move_order_by",
  "orderId": "O023",
  "delta": {{"days": 1, "hours": 0, "minutes": 0}}
}}
Use for: move / delay / advance (advance = negative delta).

2) move_order_to:
{{
  "type": "move_order_to",
  "orderId": "O023",
  "start": "2025-08-25T09:00:00+08:00"
}}
Use for: "move to" exact date/time.

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

4) swap_orders:
{{
  "type": "swap_orders",
  "orderA": "O023",
  "orderB": "O045"
}}
Use for: switch/swap order A with order B (swap positions).

5) undo:
{{ "type": "undo", "steps": 1 }}

6) redo:
{{ "type": "redo", "steps": 1 }}

Rules:
- Resolve relative dates like "tomorrow" using the given timezone and today's date.
- Order IDs should match known IDs in context. If the user says "this order" and a selected order is provided in context, use that.
- Use integers for days/hours/minutes when possible.
"""

def build_context_snapshot(df: pd.DataFrame):
    resources = sorted(df["resource"].dropna().unique().tolist())
    orders = sorted(df["order_id"].unique().tolist())
    bounds = {
        "minStart": df["start"].min().isoformat(),
        "maxFinish": df["finish"].max().isoformat(),
    }
    # if user has clicked a bar, include it for "this order"
    selected = st.session_state.get("selected_group")
    return {"orders": orders, "resources": resources, "bounds": bounds, "selectedOrder": selected, "today": str(TODAY)}

@st.cache_data(show_spinner=False, ttl=1800)
def cached_interpret(transcript: str, context_snapshot_json: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(today=str(TODAY))},
        {"role": "user", "content": f"Context: {context_snapshot_json}"},
        {"role": "user", "content": f'Command: """{transcript}"""'},
    ]
    def _call():
        return client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
    resp = with_retries(_call)
    content = resp.choices[0].message.content
    data = json.loads(content)
    ops = data.get("operations", [])
    if not isinstance(ops, list):
        raise ValueError("LLM did not return a valid operations array.")
    return ops

def interpret_with_llm(transcript: str, context_snapshot: dict):
    return cached_interpret(transcript, json.dumps(context_snapshot, separators=(",", ":")))

# =========================
# Whisper transcription (cached & trimmed)
# =========================
def _audio_hash(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _trim_to_seconds(wav_bytes: bytes, max_seconds: float = 15.0) -> bytes:
    data, samplerate = sf.read(io.BytesIO(wav_bytes), always_2d=True)
    mono = data.mean(axis=1)
    if mono.shape[0] > int(max_seconds * samplerate):
        mono = mono[: int(max_seconds * samplerate)]
    buf = io.BytesIO()
    sf.write(buf, mono, samplerate, format="WAV")
    return buf.getvalue()

@st.cache_data(show_spinner=False, ttl=1800)
def cached_transcribe(wav_bytes: bytes, language_hint: str | None):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp.flush()
        tmp_path = tmp.name
    try:
        def _call():
            with open(tmp_path, "rb") as f:
                return client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language=language_hint
                )
        tr = with_retries(_call)
        return {"text": tr.text}
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def transcribe_with_whisper(wav_bytes: bytes, language_hint: str | None = None) -> dict:
    return cached_transcribe(wav_bytes, language_hint)

# =========================
# Apply operations
# =========================
def apply_operations(ops) -> list:
    if not ops:
        return ["No operations to apply."]
    snapshot()
    msgs = []
    try:
        for op in ops:
            t = op.get("type")
            if t == "move_order_by":
                oid = op["orderId"] or st.session_state.get("selected_group")
                if not oid:
                    raise ValueError("No order specified for move.")
                delta = op.get("delta", {})
                move_order_by(
                    oid,
                    delta_days=int(delta.get("days", 0)),
                    delta_hours=int(delta.get("hours", 0)),
                    delta_minutes=int(delta.get("minutes", 0)),
                )
                msgs.append(f"Moved {oid} by {delta.get('days',0)}d {delta.get('hours',0)}h {delta.get('minutes',0)}m")

            elif t == "move_order_to":
                oid = op["orderId"] or st.session_state.get("selected_group")
                if not oid:
                    raise ValueError("No order specified for move.")
                move_order_to(oid, op["start"])
                msgs.append(f"Moved {oid} to {op['start']}")

            elif t == "shift_range_by":
                fdict = op.get("filter", {})
                delta = op.get("delta", {})
                shift_range_by(
                    fdict,
                    delta_days=int(delta.get("days", 0)),
                    delta_hours=int(delta.get("hours", 0)),
                    delta_minutes=int(delta.get("minutes", 0)),
                )
                msgs.append("Shifted range by "
                            f"{delta.get('days',0)}d {delta.get('hours',0)}h {delta.get('minutes',0)}m")

            elif t == "swap_orders":
                a, b = op.get("orderA"), op.get("orderB")
                if not a or not b:
                    raise ValueError("swap_orders requires orderA and orderB.")
                swap_orders(a, b)
                msgs.append(f"Swapped {a} ↔ {b}")

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
        do_undo(1)
        raise e

# =========================
# Rendering
# =========================
def render_gantt(df: pd.DataFrame, selected_group: str | None, height_px: int = 640):
    # Highlight whole project/order when selected
    df = df.copy()
    if selected_group:
        df["__sel__"] = np.where(df["order_id"] == selected_group, "Selected", "Other")
        color = "__sel__"
        category_orders = {"__sel__": ["Selected", "Other"]}
    else:
        color = "order_id"
        category_orders = None

    fig = px.timeline(
        df.sort_values("start"),
        x_start="start",
        x_end="finish",
        y="resource",
        color=color,
        hover_data=["order_id", "operation_id", "resource", "start", "finish"],
        category_orders=category_orders
    )
    fig.update_yaxes(autorange="reversed")

    # Muted others if a group is selected
    if selected_group:
        fig.update_traces(
            selector=dict(name="Other"),
            opacity=0.25
        )
    fig.update_layout(
        height=height_px,
        margin=dict(l=10, r=10, t=10, b=80),  # leave room for bottom bar
        showlegend=False
    )
    return fig

# =========================
# Minimalist UI layout
# =========================
# Global CSS: hide default sidebar toggle, create bottom mic bar, filter drawer button
st.markdown("""
<style>
/* Make main area tall and clean */
.block-container {padding-top: 0.6rem; padding-bottom: 5rem;} /* leave space for bottom bar */
/* Bottom mic bar */
#bottom-bar {
  position: fixed; left: 0; right: 0; bottom: 0;
  background: white; border-top: 1px solid #eee; padding: 10px 16px; z-index: 1000;
}
#bottom-inner {display: flex; gap: 8px; align-items: center; max-width: 1200px; margin: 0 auto;}
#utterance {flex: 1; border: 1px solid #ddd; border-radius: 10px; padding: 8px 12px; min-height: 44px;}
/* Floating filter toggle button (left) */
#filter-toggle {
  position: fixed; left: 6px; top: 70px; z-index: 1100;
  background: #ffffffdd; border: 1px solid #e5e5e5; border-radius: 999px; padding: 6px 10px;
}
</style>
""", unsafe_allow_html=True)

# Top row: only Gantt (85% screen height approx -> 640-760px typically)
init_history()
work_df = ensure_work_df()

# Filter drawer toggle
toggle_col = st.container()
with toggle_col:
    if st.button(("» Show filters" if not st.session_state.show_filters else "« Hide filters"), key="toggle_filters"):
        st.session_state.show_filters = not st.session_state.show_filters
        st.experimental_rerun()

# Filter drawer (left) – initially hidden
res_list = sorted(work_df["resource"].unique().tolist())
min_d = work_df["start"].min().date()
max_d = work_df["finish"].max().date()

if st.session_state.show_filters:
    with st.sidebar:
        st.markdown("### Filters")
        res_sel = st.multiselect("Resource", options=res_list, default=[])
        date_range = st.date_input("Date range", value=(min_d, max_d))
        if st.button("Apply filters"):
            st.session_state["filters"] = {"res_sel": res_sel, "date_range": date_range}
            st.toast("Filters applied", icon="✅")
else:
    # keep current filters if any
    pass

filters = st.session_state.get("filters", {"res_sel": [], "date_range": (min_d, max_d)})
view_df = get_filtered_df(work_df, filters.get("res_sel"), filters.get("date_range"))

# Gantt
gantt_height = int(st.session_state.get("gantt_height", 720))
fig = render_gantt(view_df, st.session_state.selected_group, height_px=gantt_height)
clicks = plotly_events(fig, click_event=True, hover_event=False, select_event=False, override_height=gantt_height, override_width="100%")

# click-to-highlight (whole project)
if clicks:
    # We get the y (resource) and x, but need to map back to order_id via hover text
    # Use nearest row by time and resource
    pt = clicks[0]
    # Try to read customdata or pointNumber mapping – fallback via time window
    # For robust mapping, pick the bar whose resource == pt['y'] and whose interval contains pt['x']
    resource_clicked = pt.get("y")
    x_clicked = pd.to_datetime(pt.get("x"))
    if resource_clicked and not pd.isna(x_clicked):
        cand = view_df[(view_df["resource"] == str(resource_clicked)) &
                       (view_df["start"] <= x_clicked.tz_convert(APP_TZ)) &
                       (view_df["finish"] >= x_clicked.tz_convert(APP_TZ))]
        if len(cand):
            sel_order = cand.iloc[0]["order_id"]
            st.session_state.selected_group = sel_order
            st.toast(f"Selected order: {sel_order}", icon="🎯")
            st.experimental_rerun()

# =========================
# Bottom mic bar (fixed)
# =========================
st.markdown('<div id="bottom-bar"><div id="bottom-inner">', unsafe_allow_html=True)

# Left: a simple text of transcript / manual text; Middle: mic; Right: buttons
colA, colB, colC = st.columns([6, 2, 2])

with colA:
    st.session_state.transcript = st.text_input(
        "Say or type a command", value=st.session_state.transcript, label_visibility="collapsed", key="utterance_input",
        placeholder='e.g., "move order O023 by one day" or "swap O023 with O031"'
    )

with colB:
    st.caption("Record / Stop")
    audio = mic_recorder(
        start_prompt="🎙️",
        stop_prompt="■",
        just_once=False,
        use_container_width=True,
        format="wav",
        key="mic",
    )
    if audio and isinstance(audio, dict) and audio.get("bytes"):
        raw_wav = audio["bytes"]
        try:
            trimmed = _trim_to_seconds(raw_wav, max_seconds=15.0)
        except Exception:
            trimmed = raw_wav
        h = _audio_hash(trimmed)
        if h != st.session_state.audio_hash:
            st.session_state.audio_bytes = trimmed
            st.session_state.audio_hash = h

with colC:
    st.write("")  # spacing
    go = st.button("Transcribe & interpret", type="primary")
    apply_btn = st.button("Apply")

st.markdown('</div></div>', unsafe_allow_html=True)

# Actions
if go:
    if st.session_state.audio_bytes is None and not st.session_state.transcript.strip():
        st.warning("Record or type a command first.")
    else:
        # Prefer audio → Whisper; else use typed
        if st.session_state.audio_bytes is not None:
            if not gate_api_call():
                st.info("Please wait a moment before submitting another command.")
            else:
                with st.spinner("Transcribing…"):
                    try:
                        tr = transcribe_with_whisper(st.session_state.audio_bytes, None)
                        st.session_state.transcript = tr["text"].strip()
                    except RateLimitError:
                        st.error("Hit Whisper rate limit. Try again in a few seconds.")
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")
        # Interpret
        if st.session_state.transcript.strip():
            ctx = build_context_snapshot(st.session_state.plan_df)
            if not gate_api_call():
                st.info("Please wait a moment before submitting another command.")
            else:
                with st.spinner("Interpreting…"):
                    try:
                        st.session_state.ops_preview = interpret_with_llm(st.session_state.transcript, ctx)
                        st.toast("Command ready to apply", icon="🧩")
                    except RateLimitError:
                        st.error("Hit Chat rate limit. Try again in a few seconds.")
                    except Exception as e:
                        st.error(f"Interpretation failed: {e}")

if apply_btn and st.session_state.ops_preview:
    try:
        msgs = apply_operations(st.session_state.ops_preview)
        st.success(" | ".join(msgs))
        # After apply, re-render with same filters, keep selection
        st.experimental_rerun()
    except Exception as e:
        st.error(str(e))
