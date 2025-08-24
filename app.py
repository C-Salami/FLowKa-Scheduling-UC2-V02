import os
import io
from datetime import datetime, timedelta
from dateutil import tz, parser as dtparser
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_plotly_events import plotly_events
from streamlit_mic_recorder import mic_recorder

# Speech-to-text (Google Web Speech via SpeechRecognition)
import speech_recognition as sr

# =========================
# Config & globals
# =========================
st.set_page_config(page_title="FlowKa UC1 – Voice Gantt", layout="wide")
APP_TZ = tz.gettz(st.secrets.get("TZ", "Asia/Makassar"))
TODAY = datetime.now(APP_TZ).date()

# Session state
st.session_state.setdefault("show_filters", False)
st.session_state.setdefault("selected_order", None)
st.session_state.setdefault("command_text", "")
st.session_state.setdefault("history", [])
st.session_state.setdefault("future", [])
st.session_state.setdefault("filters", None)
st.session_state.setdefault("ops_preview", None)

# =========================
# Helpers
# =========================
ALIASES = {
    "order_id": ["order_id", "order", "project", "job", "id", "ORDER_ID", "Order", "Project"],
    "operation_id": ["operation_id", "operation", "op", "OPERATION_ID"],
    "resource": ["resource", "machine", "line", "workcenter", "RESOURCE", "Machine", "Line"],
    "start": ["start", "start_time", "start_dt", "START", "Start"],
    "finish": ["finish", "end", "end_time", "finish_time", "FINISH", "End"],
}

def ensure_app_tz(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    if getattr(s.dt, "tz", None) is None:
        s = s.dt.tz_localize(APP_TZ, nonexistent="shift_forward", ambiguous="NaT")
    else:
        s = s.dt.tz_convert(APP_TZ)
    return s

def find_col(df, keys):
    for k in keys:
        if k in df.columns:
            return k
    return None

@st.cache_data(show_spinner=False)
def load_uc1_df():
    # Prefer UC1 scooter files
    paths = [
        "data/scooter_schedule.csv",
        "data/scooter_orders.csv",
        "scooter_schedule.csv",
        "scooter_orders.csv",
    ]
    found = None
    for p in paths:
        if os.path.exists(p):
            found = p
            break

    if not found:
        # Tiny fallback seed (so app opens even if files missing)
        base = datetime.combine(TODAY, datetime.min.time()).replace(tzinfo=APP_TZ)
        df = pd.DataFrame([
            {"order_id": "O021", "operation_id": "Op1", "start": base + timedelta(hours=8),  "finish": base + timedelta(hours=12), "resource": "Line A"},
            {"order_id": "O022", "operation_id": "Op1", "start": base + timedelta(hours=13), "finish": base + timedelta(hours=18), "resource": "Line A"},
            {"order_id": "O023", "operation_id": "Op1", "start": base + timedelta(hours=9),  "finish": base + timedelta(hours=15), "resource": "Line B"},
            {"order_id": "O024", "operation_id": "Op1", "start": base + timedelta(days=1, hours=8), "finish": base + timedelta(days=1, hours=12), "resource": "Line B"},
        ])
    else:
        df = pd.read_csv(found)

    # Map columns
    col_order  = find_col(df, ALIASES["order_id"])
    col_op     = find_col(df, ALIASES["operation_id"])
    col_res    = find_col(df, ALIASES["resource"])
    col_start  = find_col(df, ALIASES["start"])
    col_finish = find_col(df, ALIASES["finish"])

    if not all([col_order, col_res, col_start, col_finish]):
        raise ValueError("Missing required columns. Need order_id/order, resource, start, finish.")

    out = pd.DataFrame({
        "order_id": df[col_order].astype(str),
        "resource": df[col_res].astype(str),
        "start": ensure_app_tz(df[col_start]),
        "finish": ensure_app_tz(df[col_finish]),
    })
    if col_op:
        out["operation_id"] = df[col_op].astype(str)
    else:
        out["operation_id"] = "Op"

    out["id"] = out["order_id"] + ":" + out["operation_id"].astype(str)
    out = out.sort_values(["start", "resource"]).reset_index(drop=True)
    return out

def ensure_work_df():
    if "plan_df" not in st.session_state:
        st.session_state.plan_df = load_uc1_df().copy()
    return st.session_state.plan_df

def snapshot():
    df = st.session_state.plan_df.copy()
    st.session_state.history.append(df.to_dict(orient="records"))
    st.session_state.future.clear()

def restore_from(payload):
    df = pd.DataFrame(payload)
    for c in ["start", "finish"]:
        df[c] = ensure_app_tz(df[c])
    st.session_state.plan_df = df

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

# =========================
# Filters (UC1-like)
# =========================
def apply_filters(df: pd.DataFrame, f: dict):
    if not f:
        return df
    m = pd.Series(True, index=df.index)
    if f.get("resources"):
        m &= df["resource"].isin(f["resources"])
    if f.get("orders"):
        term = f["orders"].strip()
        if term:
            m &= df["order_id"].str.contains(term, case=False, na=False)
    if f.get("date_range"):
        d0, d1 = f["date_range"]
        if d0 and d1:
            s = datetime.combine(d0, datetime.min.time()).replace(tzinfo=APP_TZ)
            e = datetime.combine(d1, datetime.max.time()).replace(tzinfo=APP_TZ)
            m &= (df["start"] < e) & (df["finish"] > s)
    return df[m].copy()

# =========================
# Move operation (only)
# =========================
def move_order_by(order_id, delta_days=0, delta_hours=0, delta_minutes=0):
    df = st.session_state.plan_df
    m = df["order_id"] == str(order_id)
    if not m.any():
        raise ValueError(f"Order {order_id} not found.")
    delta = relativedelta(days=int(delta_days), hours=int(delta_hours), minutes=int(delta_minutes))
    df.loc[m, "start"]  = df.loc[m, "start"].apply(lambda x: x + delta)
    df.loc[m, "finish"] = df.loc[m, "finish"].apply(lambda x: x + delta)

def parse_move_command(cmd: str, selected_order: str | None):
    """
    Supported utterances:
      - move order O023 by 1 day
      - delay O023 by 2 hours
      - advance O045 by 30 minutes
      - move this order by 3 hours
    """
    import re
    text = (cmd or "").strip().lower()
    if not text:
        raise ValueError("Empty command.")

    # Determine sign: advance = negative, delay/move = positive
    sign = 1
    if "advance" in text or "bring forward" in text or "earlier" in text:
        sign = -1

    # find order id
    m_order = re.search(r"\b(o\d{2,5})\b", text)
    if m_order:
        order_id = m_order.group(1).upper()
    elif "this order" in text and selected_order:
        order_id = selected_order
    else:
        raise ValueError("No order id found. Say 'move O023 by 1 day' or click a bar then say 'move this order by 2 hours'.")

    # find quantity + unit
    m_qty = re.search(r"\bby\s+(-?\d+)\s*(day|days|hour|hours|minute|minutes|min|mins)\b", text)
    if not m_qty:
        m_qty = re.search(r"\bby\s+(-?\d+)\s*(d|h|m)\b", text)
    if not m_qty:
        raise ValueError("No amount found. Example: 'by 2 hours' or 'by 1 day'.")

    qty = int(m_qty.group(1)) * sign
    unit = m_qty.group(2)

    days = hours = minutes = 0
    if unit in ["day", "days", "d"]:
        days = qty
    elif unit in ["hour", "hours", "h"]:
        hours = qty
    elif unit in ["minute", "minutes", "min", "mins", "m"]:
        minutes = qty

    return {"orderId": order_id, "delta": {"days": days, "hours": hours, "minutes": minutes}}

# =========================
# Gantt rendering (Plotly; click→highlight whole order; distinct color per order)
# =========================
def render_gantt(df: pd.DataFrame, selected_order: str | None, height_px: int = 720):
    """
    Two layers so selection truly highlights:
      - others: low opacity
      - selected: full opacity, thicker border
    Also attach customdata=order_id so click mapping is 100% reliable.
    """
    df = df.copy()
    order_list = sorted(df["order_id"].unique().tolist())

    # Prepare customdata
    df["__order__"] = df["order_id"]

    if selected_order and selected_order in order_list:
        sel_mask = df["order_id"] == selected_order
        df_sel = df[sel_mask].copy()
        df_oth = df[~sel_mask].copy()
    else:
        df_sel = pd.DataFrame(columns=df.columns)
        df_oth = df

    fig = go.Figure()

    if len(df_oth):
        fig_oth = px.timeline(
            df_oth.sort_values("start"),
            x_start="start", x_end="finish",
            y="resource", color="order_id",
            hover_data=["order_id", "operation_id", "resource", "start", "finish"],
            category_orders={"order_id": order_list}
        )
        for tr in fig_oth.data:
            tr.opacity = 0.25
            tr.marker = dict(line=dict(width=0))
            # attach customdata if missing
            if tr.customdata is None:
                # Build per-trace customdata matching number of bars in this trace
                # We rebuild using x/y from the trace's base dataframe is complex; instead add once globally:
                pass
            fig.add_trace(tr)

    if len(df_sel):
        fig_sel = px.timeline(
            df_sel.sort_values("start"),
            x_start="start", x_end="finish",
            y="resource", color="order_id",
            hover_data=["order_id", "operation_id", "resource", "start", "finish"],
            category_orders={"order_id": order_list}
        )
        for tr in fig_sel.data:
            tr.opacity = 1.0
            tr.marker = dict(line=dict(width=2))
            fig.add_trace(tr)

    # Attach a single invisible scatter with customdata for reliable click mapping
    fig.add_trace(go.Scatter(
        x=df["start"],
        y=df["resource"],
        mode="markers",
        marker=dict(opacity=0),
        hoverinfo="skip",
        showlegend=False,
        customdata=df["__order__"],
        name="__hitmap__"
    ))

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        height=height_px,
        margin=dict(l=10, r=10, t=10, b=86),  # leave room for bottom bar
        legend_title_text="Order",
        bargap=0.2,
    )
    return fig

# =========================
# Voice transcription (Google via SpeechRecognition)
# =========================
def transcribe_google_wav_bytes(wav_bytes: bytes, language: str = "en-US") -> str:
    """
    Uses SpeechRecognition's Google Web Speech API (free, no key).
    """
    r = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio, language=language)
    except sr.UnknownValueError:
        raise RuntimeError("Could not understand audio.")
    except sr.RequestError as e:
        raise RuntimeError(f"Speech service error: {e}")

# =========================
# UI – visible filter button + bottom mic bar
# =========================
st.markdown("""
<style>
.block-container {padding-top: 0.4rem; padding-bottom: 6rem;}
#bottom-bar {
  position: fixed; left: 0; right: 0; bottom: 0;
  background: white; border-top: 1px solid #eee; padding: 10px 16px; z-index: 1000;
}
#bottom-inner {display: flex; gap: 8px; align-items: center; max-width: 1200px; margin: 0 auto;}
#cmd {flex: 1;}
/* Very visible filter toggle */
#filter-toggle { position: fixed; left: 14px; top: 14px; z-index: 1100; }
</style>
""", unsafe_allow_html=True)

# Very visible filter toggle
ft_col = st.container()
with ft_col:
    col_ft1, col_ft2 = st.columns([0.12, 0.88])
    with col_ft1:
        if st.button(("» Filters" if not st.session_state.show_filters else "« Hide"), key="toggle_filters", help="Show/Hide filters"):
            st.session_state.show_filters = not st.session_state.show_filters

# Sidebar filters (UC1-like): Resource, Order contains, Date range, Undo/Redo
if st.session_state.show_filters:
    with st.sidebar:
        st.markdown("### Filters")
        work_df_all = ensure_work_df()
        res_list = sorted(work_df_all["resource"].unique().tolist())
        res_sel = st.multiselect("Resource", options=res_list, default=st.session_state.filters.get("resources", []) if st.session_state.filters else [])
        order_term = st.text_input("Order contains", value=(st.session_state.filters.get("orders", "") if st.session_state.filters else ""))
        min_d = work_df_all["start"].min().date()
        max_d = work_df_all["finish"].max().date()
        date_range = st.date_input("Date range", value=st.session_state.filters.get("date_range", (min_d, max_d)) if st.session_state.filters else (min_d, max_d))
        if st.button("Apply filters"):
            st.session_state.filters = {"resources": res_sel, "orders": order_term, "date_range": date_range}
            st.success("Filters applied")
        st.divider()
        col_u, col_r = st.columns(2)
        with col_u:
            if st.button("Undo"):
                st.success("Undid.") if do_undo(1) else st.info("Nothing to undo.")
        with col_r:
            if st.button("Redo"):
                st.success("Redid.") if do_redo(1) else st.info("Nothing to redo.")

# Data view
work_df = ensure_work_df()
view_df = apply_filters(work_df, st.session_state.filters or {})

# Gantt (≈85% screen) + robust click mapping using customdata
gantt_height = 720
fig = render_gantt(view_df, st.session_state.selected_order, height_px=gantt_height)
clicks = plotly_events(fig, click_event=True, hover_event=False, select_event=False, override_height=gantt_height, override_width="100%")

# Click any bar to select its whole order (highlight)
if clicks:
    # Prefer customdata if present, else fall back to time/resource window
    cd = clicks[0].get("customdata")
    if cd:
        sel_order = str(cd)
    else:
        res_clicked = clicks[0].get("y")
        x_clicked = pd.to_datetime(clicks[0].get("x"))
        if res_clicked is not None and pd.notna(x_clicked):
            x_clicked = x_clicked if x_clicked.tzinfo else x_clicked.tz_localize(APP_TZ)
            hits = view_df[(view_df["resource"] == str(res_clicked)) &
                           (view_df["start"] <= x_clicked) &
                           (view_df["finish"] >= x_clicked)]
            sel_order = hits.iloc[0]["order_id"] if len(hits) else None
        else:
            sel_order = None
    if sel_order and sel_order != st.session_state.selected_order:
        st.session_state.selected_order = sel_order
        st.toast(f"Selected order: {sel_order}", icon="🎯")
        st.experimental_rerun()

# =========================
# Bottom mic bar
# =========================
st.markdown('<div id="bottom-bar"><div id="bottom-inner">', unsafe_allow_html=True)
colA, colB, colC, colD = st.columns([6, 2, 1, 1])

with colA:
    st.session_state.command_text = st.text_input(
        "Command", value=st.session_state.command_text, key="cmd", label_visibility="collapsed",
        placeholder='e.g., "move order O023 by 2 hours" or "advance this order by 1 day"'
    )

with colB:
    st.caption("Voice (record/stop)")
    audio = mic_recorder(
        start_prompt="🎙️",
        stop_prompt="■",
        just_once=False,
        use_container_width=True,
        format="wav",
        key="mic",
    )
    if audio and isinstance(audio, dict) and audio.get("bytes"):
        wav_bytes = audio["bytes"]
        try:
            transcript = transcribe_google_wav_bytes(wav_bytes, language="en-US")
            st.session_state.command_text = transcript
            st.toast("Transcribed.", icon="🗣️")
        except Exception as e:
            st.error(str(e))

with colC:
    exec_btn = st.button("Interpret")
with colD:
    apply_btn = st.button("Apply")

st.markdown('</div></div>', unsafe_allow_html=True)

# Interpret button: parse only "move" commands (UC1 v1)
if exec_btn:
    try:
        instr = parse_move_command(st.session_state.command_text, st.session_state.selected_order)
        st.session_state.ops_preview = [ {"type": "move_order_by", **instr} ]
        d = instr["delta"]
        st.success(f"Parsed: move {instr['orderId']} by {d.get('days',0)}d {d.get('hours',0)}h {d.get('minutes',0)}m")
    except Exception as e:
        st.session_state.ops_preview = None
        st.error(str(e))

# Apply
if apply_btn:
    ops = st.session_state.get("ops_preview")
    if not ops:
        st.warning("Click Interpret first.")
    else:
        try:
            snapshot()
            for op in ops:
                if op.get("type") == "move_order_by":
                    oid = op["orderId"]
                    d = op["delta"]
                    move_order_by(oid, d.get("days",0), d.get("hours",0), d.get("minutes",0))
            st.success("Applied.")
            st.experimental_rerun()
        except Exception as e:
            st.error(str(e))
