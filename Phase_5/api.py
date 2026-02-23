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
