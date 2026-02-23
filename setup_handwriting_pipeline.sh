#!/usr/bin/env zsh
set -euo pipefail

echo "Creating full 5‑phase handwriting recognition pipeline..."

# -----------------------------
# Phase 1 — Annotation UI
# -----------------------------
mkdir -p Phase_1/data

cat > Phase_1/app.py <<'PY'
#!/usr/bin/env python3
import streamlit as st
from pathlib import Path
from PIL import Image
import json
import time

st.set_page_config(page_title="Gold Set Annotation", layout="wide")

DATA_DIR = Path("Phase_1/data")
ANNOTATIONS = Path("Phase_1/annotations.json")
DATA_DIR.mkdir(exist_ok=True, parents=True)

if ANNOTATIONS.exists():
    annotations = json.loads(ANNOTATIONS.read_text())
else:
    annotations = {}

st.title("Phase 1 — Human Annotation Interface")
st.write("Annotate 100 gold‑set images with high‑quality labels.")

image_files = sorted(DATA_DIR.glob("*.png"))

if not image_files:
    st.warning("Place images in Phase_1/data/")
    st.stop()

idx = st.number_input("Image index", 0, len(image_files)-1, 0)
img_path = image_files[idx]
img = Image.open(img_path)

st.image(img, caption=str(img_path.name), width=500)

label = st.text_input("Enter 7‑digit Student ID")
flag = st.checkbox("Flag as ambiguous")

if st.button("Save Annotation"):
    annotations[str(img_path.name)] = {
        "label": label,
        "flagged": flag,
        "timestamp": time.time()
    }
    ANNOTATIONS.write_text(json.dumps(annotations, indent=2))
    st.success("Saved!")
PY

# -----------------------------
# Phase 2 — Validation Tools
# -----------------------------
mkdir -p Phase_2

cat > Phase_2/validate_annotations.py <<'PY'
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
PY

# -----------------------------
# Phase 3 — AI Labeling
# -----------------------------
mkdir -p Phase_3/data

cat > Phase_3/label_with_ai.py <<'PY'
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
PY

# -----------------------------
# Phase 4 — Distillation Training
# -----------------------------
mkdir -p Phase_4

cat > Phase_4/train_student.py <<'PY'
#!/usr/bin/env python3
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

dataset = load_dataset("json", data_files="Phase_4/training_data.json")

def preprocess(batch):
    pixel_values = processor(batch["image"], return_tensors="pt").pixel_values[0]
    labels = processor.tokenizer(batch["label"], return_tensors="pt").input_ids[0]
    return {"pixel_values": pixel_values, "labels": labels}

dataset = dataset.map(preprocess)

args = TrainingArguments(
    output_dir="Phase_4/student_model",
    per_device_train_batch_size=4,
    num_train_epochs=10,
    learning_rate=5e-5,
    logging_steps=50,
)

trainer = Trainer(model=model, args=args, train_dataset=dataset["train"])
trainer.train()

model.save_pretrained("Phase_4/student_model")
processor.save_pretrained("Phase_4/student_model")
PY

# -----------------------------
# Phase 5 — Deployment API
# -----------------------------
mkdir -p Phase_5

cat > Phase_5/api.py <<'PY'
#!/usr/bin/env python3
from flask import Flask, request, jsonify
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

app = Flask(__name__)

processor = TrOCRProcessor.from_pretrained("Phase_4/student_model")
model = VisionEncoderDecoderModel.from_pretrained("Phase_4/student_model")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "no image"}), 400

    img = Image.open(f.stream).convert("RGB")
    pixel_values = processor(img, return_tensors="pt").pixel_values
    ids = model.generate(pixel_values)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    digits = "".join(filter(str.isdigit, text))

    if len(digits) != 7:
        return jsonify({"id": None, "confidence": 0.0})

    return jsonify({"id": digits, "confidence": 0.95})
PY

cat > Phase_5/requirements.txt <<'REQ'
flask
gunicorn
transformers
torch
Pillow
REQ

cat > Phase_5/Dockerfile <<'DOCK'
FROM python:3.10-slim
WORKDIR /app

COPY Phase_5/requirements.txt .
RUN pip install -r requirements.txt

COPY . /app
EXPOSE 8000

CMD ["gunicorn", "-b", "0.0.0.0:8000", "Phase_5.api:app"]
DOCK

mkdir -p .github/workflows

cat > .github/workflows/phase5-ci.yml <<'YML'
name: Phase5 CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - run: pip install -r Phase_5/requirements.txt
      - run: python -m py_compile Phase_5/api.py
YML

echo "All phases created successfully."
