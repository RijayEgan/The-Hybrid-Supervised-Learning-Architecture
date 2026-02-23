#!/usr/bin/env python3
import json
from pathlib import Path

def validate(ann_path):
    data = json.loads(Path(ann_path).read_text())
    errors = []

    for fname, entry in data.items():
        label = entry.get("label")
        if not label or not label.isdigit() or len(label) != 7:
            errors.append((fname, f"Invalid label: {label}"))

    return errors

if __name__ == "__main__":
    errors = validate("Phase_1/annotations.json")
    if not errors:
        print("Validation passed.")
    else:
        print("Validation errors:")
        for e in errors:
            print(" -", e)
