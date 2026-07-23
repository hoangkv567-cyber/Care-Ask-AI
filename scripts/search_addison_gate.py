import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('data/disease_gates.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=== DISEASE GATES FOR ADDISON ===")
if isinstance(data, list):
    for i, g in enumerate(data):
        dis = g.get('disease', '')
        if "Addison" in dis or "thượng thận" in dis:
            print(f"Index {i}: {g}")
elif isinstance(data, dict):
    for k, v in data.items():
        if "Addison" in k or "thượng thận" in k:
            print(f"Key '{k}': {v}")
