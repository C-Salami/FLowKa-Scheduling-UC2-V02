import os, json, re
from dateutil import parser as dtp
from nlp_schema import INTENT_SCHEMA

# --- number words -> int (simple MVP up to 20) ---
NUM_WORDS = {
    "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,
    "six":6,"seven":7,"eight":8,"nine":9,"ten":10,
    "eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,
    "sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19,"twenty":20
}
def _num_token_to_int(tok: str):
    t = tok.strip().lower().replace("-", " ")
    if t.isdigit(): return int(t)
    parts = [p for p in t.split() if p]
    if len(parts)==1 and parts[0] in NUM_WORDS: return NUM_WORDS[parts[0]]
    if len(parts)==2 and parts[0] in NUM_WORDS and parts[1] in NUM_WORDS:
        return NUM_WORDS[parts[0]] + NUM_WORDS[parts[1]]
    return None

def _extract_with_openai(user_text: str):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    SYSTEM = ("You normalize factory scheduling edit commands for a Gantt. "
              "Return ONLY JSON matching the given schema. Supported intents: "
              "delay_order, move_order, swap_orders. Order IDs like O021. "
              "Resolve relative dates to Asia/Makassar; default time 08:00; "
              "default delay units = days.")
    USER_GUIDE = (
        'Examples:\n'
        '1) "delay O021 one day" -> {"intent":"delay_order","order_id":"O021","days":1}\n'
        '2) "push order O009 by 24h" -> {"intent":"delay_order","order_id":"O009","hours":24}\n'
        '3) "move o014 to Aug 30 9am" -> {"intent":"move_order","order_id":"O014","date":"2025-08-30","time":"09:00"}\n'
        '4) "swap o027 with o031" -> {"intent":"swap_orders","order_id":"O027","order_id_2":"O031"}\n'
    )
    resp = client.responses.create(
        model="gpt-5.1",
        input=[{"role":"system","content":SYSTEM},
               {"role":"user","content":USER_GUIDE},
               {"role":"user","content":user_text}],
        response_format={"type":"json_schema","json_schema":{"name":"Edit","schema":INTENT_SCHEMA}}
    )
    text = resp.output[0].content[0].text
    data = json.loads(text)
    data["_source"] = "openai"
    return data

def _regex_fallback(user_text: str):
    t = user_text.strip()
    low = t.lower()

    # SWAP: "swap O023 O053" | "swap O023 with O053" | "swap O023 & O053"
    m = re.search(r"(?:^|\b)(swap|switch)\s+(o\d{3})\s*(?:with|and|&)?\s*(o\d{3})\b", low)
    if m:
        return {"intent":"swap_orders","order_id":m.group(2).upper(),"order_id_2":m.group(3).upper(),"_source":"regex"}

    # DELAY with/without 'by', digits or words
    m = re.search(r"(delay|push|postpone)\s+(o\d{3}).*?\bby\b\s+([\w\-]+)\s*(day|days|d|hour|hours|h)\b", low)
    if m:
        n = _num_token_to_int(m.group(3))
        if n is not None:
            unit = m.group(4)
            out = {"intent":"delay_order","order_id":m.group(2).upper(),"_source":"regex"}
            if unit.startswith("d"): out["days"] = n
            else: out["hours"] = n
            return out

    m = re.search(r"(delay|push|postpone)\s+(o\d{3}).*?\b([\w\-]+)\s*(day|days|d|hour|hours|h)\b", low)
    if m:
        n = _num_token_to_int(m.group(3))
        if n is not None:
            unit = m.group(4)
            out = {"intent":"delay_order","order_id":m.group(2).upper(),"_source":"regex"}
            if unit.startswith("d"): out["days"] = n
            else: out["hours"] = n
            return out

    # MOVE: "move Oxxx to/on <datetime>"
    m = re.search(r"(move|set|schedule)\s+(o\d{3})\s+(to|on)\s+(.+)", low)
    if m:
        when = m.group(4)
        try:
            dt = dtp.parse(when, fuzzy=True)
            return {"intent":"move_order","order_id":m.group(2).upper(),
                    "date":dt.date().isoformat(),"time":dt.strftime("%H:%M"),
                    "_source":"regex"}
        except Exception:
            pass

    # fallback "one day"
    m = re.search(r"(delay|push|postpone)\s+(o\d{3}).*\b(one)\s+day\b", low)
    if m:
        return {"intent":"delay_order","order_id":m.group(2).upper(),"days":1,"_source":"regex"}

    return {"intent":"unknown","raw":user_text,"_source":"regex"}

def extract_intent(user_text: str) -> dict:
    try:
        if os.getenv("OPENAI_API_KEY"):
            return _extract_with_openai(user_text)
    except Exception:
        pass
    return _regex_fallback(user_text)
