#!/usr/bin/env python3
import base64
import json
import time
import os
from pathlib import Path
from anthropic import Anthropic

PROMPT = """
You are a document processing system. Extract the 7‑digit handwritten student ID.
Return ONLY the digits. If unclear, return "UNCLEAR".
"""

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

DATA_DIR = Path("Phase_3/data")
OUT = Path("Phase_3/ai_labels.json")

def encode_image(path):
    return base64.b64encode(Path(path).read_bytes()).decode()

def label_image(path):
    img_data = encode_image(path)
    resp = client.messages.create(
        model="claude-3-5-sonnet",
        max_tokens=50,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_data
                }},
                {"type": "text", "text": PROMPT}
            ]
        }]
    )
    return resp.content[0].text.strip()

def main():
    results = {}
    for img in sorted(DATA_DIR.glob("*.png")):
        try:
            sid = label_image(img)
            results[img.name] = {"id": sid, "error": None}
        except Exception as e:
            results[img.name] = {"id": None, "error": str(e)}
        time.sleep(0.1)

    OUT.write_text(json.dumps(results, indent=2))
    print("Saved:", OUT)

if __name__ == "__main__":
    main()
