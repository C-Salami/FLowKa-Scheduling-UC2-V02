import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

random.seed(7)
np.random.seed(7)

start_date = datetime(2025, 8, 21, 8, 0, 0)
num_orders = 100
wheel_types = ["Urban-200", "Offroad-250", "Racing-180", "HeavyDuty-300", "Eco-160"]
machines = [
    ("M1", "Lathe"), ("M2", "CNC"), ("M3", "Drill"),
    ("M4", "Paint"), ("M5", "Assembly"), ("M6", "QA"),
]

operation_templates = [[
    {"op": "Turning",  "allowed_machines": ["M1"], "base_min": 1.0, "base_max": 2.5},
    {"op": "CNC",      "allowed_machines": ["M2"], "base_min": 1.0, "base_max": 3.0},
    {"op": "Drill",    "allowed_machines": ["M3"], "base_min": 0.5, "base_max": 1.5},
    {"op": "Paint",    "allowed_machines": ["M4"], "base_min": 1.0, "base_max": 2.0},
    {"op": "Assembly", "allowed_machines": ["M5"], "base_min": 1.0, "base_max": 2.0},
    {"op": "QA",       "allowed_machines": ["M6"], "base_min": 0.5, "base_max": 1.0, "optional": True},
]]

type_speed = {"Urban-200":1.0,"Offroad-250":1.2,"Racing-180":0.9,"HeavyDuty-300":1.35,"Eco-160":0.85}

orders = []
for i in range(1, num_orders + 1):
    wtype = random.choice(wheel_types)
    qty = random.randint(20, 120)
    horizon_days = random.randint(3, 14)
    due = start_date + timedelta(days=horizon_days, hours=random.randint(0, 8))
    template = operation_templates[0][:]
    ops = []
    for step in template:
        if step.get("optional", False):
            if random.random() < 0.85:
                ops.append({k:v for k,v in step.items() if k!="optional"})
        else:
            ops.append(step)
    if random.random() < 0.35:
        heat_op = {"op": "HeatTreat", "allowed_machines": ["M2","M3"], "base_min": 0.75, "base_max": 1.75}
        ops.insert(2, heat_op)
    orders.append({"order_id": f"O{i:03d}", "wheel_type": wtype, "quantity": qty, "due_date": due, "ops": ops})

orders_df = pd.DataFrame([{
    "order_id": o["order_id"], "wheel_type": o["wheel_type"], "quantity": o["quantity"], "due_date": o["due_date"]
} for o in orders])

machine_next_free = {m[0]: start_date for m in machines}
schedule_rows = []
orders_sorted = sorted(orders, key=lambda x: (x["due_date"], x["wheel_type"]))

for o in orders_sorted:
    prev_end = start_date
    for seq, step in enumerate(o["ops"], start=1):
        base = np.random.uniform(step["base_min"], step["base_max"])
        setup = np.random.uniform(0.2, 0.6)
        rate = base * type_speed[o["wheel_type"]]
        duration_hours = setup + rate * (0.5 + 0.5*np.log1p(o["quantity"]/20.0))
        earliest_machine = min(step["allowed_machines"], key=lambda m: machine_next_free[m])
        est = max(prev_end, machine_next_free[earliest_machine])
        start_t = est
        end_t = est + timedelta(hours=float(duration_hours))
        machine_next_free[earliest_machine] = end_t
        prev_end = end_t
        schedule_rows.append({
            "order_id": o["order_id"], "wheel_type": o["wheel_type"], "operation": step["op"],
            "machine": earliest_machine, "start": start_t, "end": end_t,
            "duration_hours": duration_hours, "sequence": seq, "due_date": o["due_date"]
        })

schedule_df = pd.DataFrame(schedule_rows)

Path("data").mkdir(exist_ok=True, parents=True)
orders_df.to_csv("data/scooter_orders.csv", index=False)
schedule_df.to_csv("data/scooter_schedule.csv", index=False)
print("Wrote data/scooter_orders.csv and data/scooter_schedule.csv")
