#!/usr/bin/env python3
"""
Convert a HA WashData diagnostics dump to import-compatible format.
Usage: python convert_dump_to_import.py <dump.json> <output.json>
"""
import json
import sys

def find_key(obj, key):
    """Recursively find a key in nested dict."""
    if key in obj:
        return obj[key]
    for k, v in obj.items():
        if isinstance(v, dict):
            item = find_key(v, key)
            if item is not None:
                return item
    return None

def convert_dump(input_path, output_path):
    from datetime import datetime
    
    with open(input_path, 'r') as f:
        dump = json.load(f)
    
    # Extract relevant data
    past_cycles = find_key(dump, "past_cycles") or []
    profiles = find_key(dump, "profiles") or {}
    envelopes = find_key(dump, "envelopes") or {}
    
    # Build import-compatible format matching version 2
    export_data = {
        "version": 2,
        "entry_id": "converted",
        "exported_at": datetime.now().isoformat(),
        "data": {
            "profiles": profiles,
            "past_cycles": past_cycles,
            "envelopes": envelopes
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Converted {len(past_cycles)} cycles and {len(profiles)} profiles")
    print(f"Output written to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_dump_to_import.py <dump.json> <output.json>")
        sys.exit(1)
    
    convert_dump(sys.argv[1], sys.argv[2])
