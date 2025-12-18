import json
from datetime import datetime

# Load existing storage
with open('/root/ha_config/.storage/ha_washdata.01KCKTVG4Z2JP681851KWN367C', 'r') as f:
    existing = json.load(f)

# Load generated mock data
with open('./cycle_data/washer_cycles.json', 'r') as f:
    generated = json.load(f)

# Handle both formats: with or without 'data' wrapper
generated_data = generated.get('data', generated)
generated_cycles = generated_data['past_cycles']

# Merge cycles (generated goes first, then existing)
merged_cycles = generated_cycles + existing['data']['past_cycles']

# Update storage with merged data
existing['data']['past_cycles'] = merged_cycles
existing['data']['last_active_save'] = datetime.now().isoformat()

# Write back
with open('/root/ha_config/.storage/ha_washdata.01KCKTVG4Z2JP681851KWN367C', 'w') as f:
    json.dump(existing, f, indent=2)

print(f"Merged {len(generated_cycles)} new cycles with {len(existing['data']['past_cycles'])} existing")