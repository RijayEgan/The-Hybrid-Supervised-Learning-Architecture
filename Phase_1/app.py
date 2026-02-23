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
