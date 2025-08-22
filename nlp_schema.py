# nlp_schema.py
INTENT_SCHEMA = {
  "type": "object",
  "properties": {
    "intent": {"type": "string", "enum": ["delay_order", "move_order", "swap_orders"]},
    "order_id": {"type": "string", "pattern": "^O\\d{3}$"},
    "order_id_2": {"type": "string", "pattern": "^O\\d{3}$"},
    "days": {"type": "number"},
    "hours": {"type": "number"},
    "date": {"type": "string"},     # ISO date preferred (YYYY-MM-DD)
    "time": {"type": "string"},     # HH:MM (24h)
    "timezone": {"type": "string", "default": "Asia/Makassar"},
    "note": {"type": "string"}
  },
  "required": ["intent"],
  "additionalProperties": False
}
