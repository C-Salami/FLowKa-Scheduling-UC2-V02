# nlp_validate.py
from dateutil import parser as dtp
from datetime import datetime
import pytz

DEFAULT_TZ = "Asia/Makassar"
TZ = pytz.timezone(DEFAULT_TZ)

def validate_intent(payload: dict, orders_df, sched_df):
    intent = payload.get("intent")

    def order_exists(oid):
        return oid and (orders_df["order_id"] == oid).any()

    if intent not in ("delay_order", "move_order", "swap_orders"):
        return False, "Unsupported intent"

    # Common: require valid order(s)
    if intent in ("delay_order", "move_order", "swap_orders"):
        oid = payload.get("order_id")
        if not order_exists(oid):
            return False, f"Unknown order_id: {oid}"

    if intent == "swap_orders":
        oid2 = payload.get("order_id_2")
        if not order_exists(oid2):
            return False, f"Unknown order_id_2: {oid2}"
        if oid2 == payload.get("order_id"):
            return False, "Cannot swap the same order."

    if intent == "delay_order":
        if not payload.get("days") and not payload.get("hours"):
            return False, "Delay needs days or hours."
        # normalize to numbers
        try:
            if "days" in payload and payload["days"] is not None:
                payload["days"] = float(payload["days"])
            if "hours" in payload and payload["hours"] is not None:
                payload["hours"] = float(payload["hours"])
        except Exception:
            return False, "Days/Hours must be numeric."
        return True, "ok"

    if intent == "move_order":
        date_str = payload.get("date")
        hhmm = payload.get("time") or "08:00"
        if not date_str:
            return False, "Move needs a date."
        try:
            dt = dtp.parse(f"{date_str} {hhmm}")
            # localize to Asia/Makassar
            if dt.tzinfo is None:
                dt = TZ.localize(dt)
            else:
                dt = dt.astimezone(TZ)
            payload["_target_dt"] = dt
        except Exception:
            return False, f"Unparseable datetime: {date_str} {hhmm}"
        return True, "ok"

    return False, "Invalid payload"
