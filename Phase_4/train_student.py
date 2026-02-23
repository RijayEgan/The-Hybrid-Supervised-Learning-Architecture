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
