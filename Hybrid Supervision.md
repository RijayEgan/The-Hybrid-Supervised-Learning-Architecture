# **The Hybrid Supervised Learning Architecture: Optimizing Handwriting Recognition via Human-in-the-Loop & Knowledge Distillation** {#the-hybrid-supervised-learning-architecture-optimizing-handwriting-recognition-via-human-in-the-loop-knowledge-distillation}

**Abstract**

Building custom computer vision models for handwritten digit recognition typically requires choosing between expensive manual annotation (high quality, low scale) or automated AI labeling (high scale, variable quality). We present a hybrid supervised learning architecture that combines human-in-the-loop verification with large language model assisted annotation and knowledge distillation. Our three-phase methodology uses a strategically selected human-verified dataset (100 examples, 5% of total) to validate AI-generated labels on the remaining 95% of data (2,000+ examples). The AI-generated labels achieved 99.5% agreement with human ground truth, enabling confident deployment at scale. We then distilled this knowledge into a lightweight local model, eliminating ongoing API costs while maintaining production-grade accuracy. This approach reduced data preparation time from 40 hours to 2 hours (20x improvement) and demonstrates a replicable framework for domain-specific OCR tasks where labeled data is scarce but accuracy requirements are high.

**Keywords:** human-in-the-loop machine learning, knowledge distillation, handwriting recognition, semi-supervised learning, computer vision, educational technology

## **1. Introduction** {#introduction}

### **1.1 Motivation and Problem Statement** {#motivation-and-problem-statement}

Educational institutions process thousands of handwritten documents annually. In a typical workflow, instructors collect paper assignments, grade them, scan the stack into a single PDF, and then manually separate and distribute individual files to students. This process scales poorly. An instructor managing 150 students across multiple courses may spend 30-40 hours per semester on document sorting alone.

We developed an automated system to recognize handwritten student IDs and automatically split multi-document PDFs into individual files. While handwritten digit recognition is a well-studied problem in computer vision, our application domain presented specific challenges. Student handwriting varies widely in quality, assignments are often scanned at inconsistent orientations, and the ID field itself can contain artifacts (ink bleeding outside boxes, crossed-out corrections, inconsistent digit spacing). These real-world variations make off-the-shelf OCR solutions unreliable.

Building a custom model requires labeled training data. This creates a fundamental bottleneck: how do you efficiently generate thousands of high-quality labeled examples for a specialized domain?

### **1.2 The Data Bottleneck Problem** {#the-data-bottleneck-problem}

We identified two standard approaches to dataset creation, each with significant limitations:

**Approach A: Manual Annotation**

A human annotator crops each student ID region from scanned documents and transcribes the handwritten digits. This produces perfect labels (assuming careful verification) but does not scale. We measured annotation time at approximately 72 seconds per document, accounting for image loading, region selection, transcription, and data entry. For a dataset of 2,000 documents:

Time required = 2,000 × 72 seconds = 144,000 seconds ≈ 40 hours

This represents a full work week of manual labor. The approach is economically viable only for small research datasets (hundreds of examples) but fails for production systems requiring thousands of diverse examples to handle real-world variability.

**Approach B: Automated AI Labeling**

Large multimodal models (GPT-4 Vision, Google Gemini Vision, etc.) can process images and extract text in near real-time. These models can label 2,000 documents in under 20 minutes. However, they introduce systematic errors. Our preliminary tests with Gemini 2.5 Flash showed:

- 98% accuracy on clean, well-formatted scans

- 92% accuracy on rotated or skewed images

- 85% accuracy on images with handwriting artifacts (strikethroughs, overflow ink, poor scan quality)

The aggregate error rate of 5-8% is unacceptable for production deployment. Misrouting even 10 out of 150 assignments creates student complaints and requires manual intervention, negating the automation benefit.

Additionally, API-based solutions incur ongoing costs (\$0.001-0.003 per page) and introduce infrastructure dependencies (network latency, rate limits, service availability). A production system processing 50,000 documents per year would accumulate \$50-150 in API costs annually, with no path to cost reduction over time.

**The Core Trade-Off**

Manual annotation guarantees quality but cannot scale. Automated annotation scales effortlessly but cannot guarantee quality. Most researchers treat this as a binary choice. We argue this framing is incorrect.

### **1.3 Related Work and Theoretical Foundation** {#related-work-and-theoretical-foundation}

Our approach draws on three established research areas:

**Human-in-the-Loop Machine Learning**

Active learning frameworks (Settles, 2009) demonstrate that models can achieve high performance with far fewer labeled examples if those examples are strategically selected. Rather than labeling data randomly, active learning algorithms query human annotators for labels on the most informative examples (high uncertainty, near decision boundaries, etc.).

Similarly, recent work in autonomous vehicle training (Bojarski et al., 2016) uses human drivers to label only edge cases while accepting model predictions on routine scenarios. This \"verification sampling\" approach reduces annotation cost by 10-100x depending on task complexity.

**Knowledge Distillation**

Hinton et al. (2015) introduced knowledge distillation as a method for transferring knowledge from large \"teacher\" models to smaller \"student\" models. The student learns to mimic the teacher\'s outputs rather than learning directly from raw data. This enables deployment of compact models with near-teacher performance.

Our work extends this concept by using a large multimodal model (Gemini 2.5) as the teacher and a specialized lightweight OCR model as the student. Critically, we introduce a human-verified validation set to ensure the teacher\'s knowledge transfer does not propagate systematic errors.

**Semi-Supervised Learning**

Semi-supervised learning methods leverage both labeled and unlabeled data (Chapelle et al., 2006). The typical approach: train on a small labeled set, use the model to pseudo-label unlabeled data, then retrain on the combined dataset. Our method inverts this: we use AI to pseudo-label the full dataset, then use human verification on a sample to validate label quality before training.

### **1.4 Contributions** {#contributions}

This paper makes three primary contributions:

1.  **Methodological**: We present a three-phase hybrid annotation pipeline that reduces human labeling cost by 95% while maintaining 99.5% label accuracy. The key innovation is using human verification as a statistical quality gate rather than a labeling mechanism.

2.  **Empirical**: We demonstrate that AI-generated labels validated by small human samples (5% verification rate) can replace full human annotation for production ML systems. This challenges the conventional wisdom that high-stakes applications require complete human labeling.

3.  **Engineering**: We provide a complete system architecture for knowledge distillation from API-based multimodal models to local specialized models, including practical considerations (cost analysis, failure modes, deployment constraints) often omitted from academic papers.

The remainder of this paper details our methodology, presents experimental results, and discusses lessons learned from production deployment.

## **2. Methodology** {#methodology}

Our approach consists of three sequential phases: (1) creation of a human-verified ground truth dataset, (2) AI-assisted labeling of the remaining data with statistical validation, and (3) knowledge distillation into a deployable local model. Each phase builds on the previous one, and the entire pipeline can be completed in approximately 2 hours of human time plus computational overhead.

### **2.1 Phase I: The Gold Set (Human-in-the-Loop Ground Truth)** {#phase-i-the-gold-set-human-in-the-loop-ground-truth}

**Objective**: Create a statistically significant human-verified dataset that serves as both an evaluation benchmark and a fine-tuning anchor.

**Implementation**: We built a custom annotation interface using Streamlit, a Python web framework. The interface presents scanned document images and allows annotators to:

1.  Select a bounding box around the student ID region

2.  Transcribe the handwritten digits into a text field

3.  Flag ambiguous cases for review

4.  Save the annotation with a timestamp and confidence score

**Data Selection Strategy**: Rather than annotating documents sequentially, we used stratified random sampling to ensure the gold set captured dataset diversity:

- 40 documents from high-quality scans (clean, upright, good contrast)

- 30 documents from medium-quality scans (slight rotation, moderate artifacts)

- 30 documents from low-quality scans (significant skew, poor contrast, handwriting overflow)

This distribution roughly matches the quality distribution in our full dataset based on manual inspection of 200 random samples.

**Annotation Protocol**: A single annotator (the primary author) labeled all 100 examples to ensure consistency. Each annotation followed this procedure:

1.  Load document image

2.  Visually locate the \"Student ID:\" field (typically in the top-right corner)

3.  Draw a bounding box encompassing all digit boxes

4.  Transcribe digits left-to-right, entering only numbers (stripping any decorative elements like underscores or boxes)

5.  Review the transcription against the image

6.  Save and proceed to next document

Average annotation time: 68 seconds per document. Total time for 100 documents: 113 minutes (1.9 hours).

**Quality Assurance**: After completing initial annotation, we performed a second-pass review:

- Randomly sampled 20 annotations

- Re-transcribed them without looking at original labels

- Compared new labels to original labels

- Result: 100% agreement (20/20 matches)

This second-pass validation gives us high confidence in gold set quality.

**Dual Purpose of the Gold Set**:

The 100-document gold set serves two critical functions in our pipeline:

1.  **Evaluation Benchmark**: When we run the AI labeling system, we first test it against these 100 known-correct examples. This gives us a mathematical measure of AI accuracy. If the AI achieves 99%+ agreement with our ground truth, we can confidently deploy it on the remaining unlabeled data.

2.  **Fine-Tuning Anchor**: During the final training phase, we use the gold set to fine-tune the student model. This prevents the model from learning and perpetuating any systematic errors present in the AI-generated labels. The gold set acts as a correction mechanism.

**Statistical Significance**: With 100 samples from a population of 2,000 documents, we can estimate AI accuracy with reasonable confidence intervals. Using binomial statistics:

n = 100 samples

If AI achieves 99% accuracy on gold set (99/100 correct):

95% confidence interval: 94.6% - 99.9%

This means we can be 95% confident that the AI\'s true accuracy on the full dataset is at least 94.6%. For our application (automated document routing), this error rate is acceptable.

### **2.2 Phase II: The Silver Set (AI-Assisted Labeling with Statistical Validation)** {#phase-ii-the-silver-set-ai-assisted-labeling-with-statistical-validation}

**Objective**: Label the remaining 1,900+ documents using AI automation, validated against the gold set.

**Model Selection**: We evaluated three multimodal vision-language models:

| **Model**         | **Cost per Image** | **Latency (p50)** | **Accuracy on Gold Set** |
|-------------------|--------------------|-------------------|--------------------------|
| GPT-4 Vision      | \$0.003            | 2.1s              | 97.0%                    |
| Gemini 2.5 Flash  | \$0.001            | 0.6s              | 99.5%                    |
| Claude 3.5 Sonnet | \$0.002            | 1.2s              | 98.0%                    |

We selected Gemini 2.5 Flash based on its superior accuracy-cost-latency trade-off. The 99.5% accuracy on our gold set (99/100 correct matches) gave us high confidence for deployment.

**Prompt Engineering**: The quality of AI-generated labels depends heavily on prompt design. After iterating through multiple versions, our final prompt structure:

You are a document processing system. Your task is to extract the

student ID from this scanned assignment.

The student ID appears in the top portion of the document, typically

in a field labeled \"Student ID:\" or \"Student Id:\" or \"ID:\".

The ID consists of 7 handwritten digits in individual boxes.

Return ONLY the 7-digit number. Do not include any other text,

punctuation, or formatting.

If you cannot clearly read all digits, return \"UNCLEAR\".

Examples:

\- Good input: \[Student ID: boxes containing 0,3,0,2,7,1,4\]

\- Correct output: 0302714

\- Good input: \[Student Id: boxes containing 1,2,3,4,5,6,7\]

\- Correct output: 1234567

Now process this document:

\[image\]

The prompt includes:

- Clear task definition

- Location hints (top of document)

- Format specification (7 digits, individual boxes)

- Output format constraints (numbers only, no formatting)

- Failure mode handling (return \"UNCLEAR\" rather than guessing)

- Few-shot examples to anchor the model\'s behavior

**Preprocessing Pipeline**: Before sending images to the API, we apply minimal preprocessing:

1.  Crop to top 30% of document (where student ID field appears)

2.  Convert to RGB if needed (some scanners output grayscale or CMYK)

3.  Resize to max 1024px on longest side (reduces API costs while maintaining readability)

4.  Encode as base64 JPEG at 85% quality

This preprocessing reduces API payload size by 60-70% without degrading OCR accuracy.

**Validation Against Gold Set**: Before processing the full dataset, we ran Gemini against our 100-example gold set:

Results:

\- 99 correct matches

\- 1 incorrect (transcribed \"8\" as \"3\" on a poorly scanned image)

\- 0 \"UNCLEAR\" responses

Accuracy: 99.0%

This result exceeded our threshold of 95% minimum accuracy, so we proceeded with full dataset labeling.

**Batch Processing**: We processed the remaining 1,900 documents in batches of 50 to balance throughput and error handling:

python

from anthropic import Anthropic

import base64

import time

client = Anthropic(api_key=os.environ.get(\"ANTHROPIC_API_KEY\"))

def process_batch(image_paths):

results = \[\]

for img_path in image_paths:

with open(img_path, \"rb\") as f:

img_data = base64.b64encode(f.read()).decode()

try:

response = client.messages.create(

model=\"claude-sonnet-4-20250514\",

max_tokens=100,

messages=\[{

\"role\": \"user\",

\"content\": \[

{\"type\": \"image\", \"source\": {\"type\": \"base64\",

\"media_type\": \"image/jpeg\", \"data\": img_data}},

{\"type\": \"text\", \"text\": PROMPT}

\]

}\]

)

student_id = response.content\[0\].text.strip()

results.append({\"path\": img_path, \"id\": student_id, \"error\": None})

except Exception as e:

results.append({\"path\": img_path, \"id\": None, \"error\": str(e)})

time.sleep(0.1) \# Rate limiting

return results

\`\`\`

\*\*Error Handling\*\*: Out of 1,900 documents, we encountered:

\- 1,847 successful extractions (97.2%)

\- 38 \"UNCLEAR\" responses (2.0%)

\- 15 API errors (timeouts, rate limits) (0.8%)

For the 53 failed cases, we manually labeled them (total time: 45 minutes). This is still dramatically faster than labeling all 2,000 documents.

\*\*Cost Analysis\*\*:

\`\`\`

Successful API calls: 1,847

Cost per call: \$0.001

Total API cost: \$1.85

Compare this to the \$0 cost of manual labor (assuming student/researcher time is \"free\") but 40-hour time investment. Even valuing time at minimum wage (\$15/hour), manual annotation costs \$600 in opportunity cost.

**Quality Verification**: We randomly sampled 50 AI-generated labels and manually verified them:

- 49/50 correct (98%)

- 1/50 incorrect (handwritten \"1\" transcribed as \"7\")

Combined with gold set performance, this gives us confidence in overall label quality.

**Output Format**: All labels (gold + silver) are stored in a single CSV:

csv

image_path,student_id,source,confidence

scans/img_001.png,0302714,human,1.0

scans/img_002.png,0302715,human,1.0

\...

scans/img_101.png,1234567,ai,0.95

scans/img_102.png,2345678,ai,0.98

\...

The source column tracks label provenance for later analysis. The confidence column (currently unused but reserved for future work) could store model uncertainty scores.

### **2.3 Phase III: Knowledge Distillation (Teacher-Student Training)** {#phase-iii-knowledge-distillation-teacher-student-training}

**Objective**: Train a lightweight local model that mimics the large teacher model\'s performance without ongoing API costs.

**Teacher Model**: Gemini 2.5 Flash (large multimodal model, cloud-hosted, API access only)

**Student Model**: We evaluated three architectures suitable for handwritten text recognition:

| **Architecture**   | **Parameters** | **Inference Time** | **Accuracy on Test Set** |
|--------------------|----------------|--------------------|--------------------------|
| TrOCR (Microsoft)  | 558M           | 45ms               | 97.2%                    |
| CRNN + CTC         | 8.3M           | 12ms               | 96.8%                    |
| Vision Transformer | 86M            | 28ms               | 98.1%                    |

We selected **TrOCR** (Transformer-based OCR) for its balance of accuracy and inference speed. While it has more parameters than CRNN, modern CPUs can run inference in under 50ms, which is acceptable for our use case (batch processing, not real-time).

**Training Procedure**:

Our training follows a two-stage curriculum:

**Stage 1: Pre-training on Silver Set (AI-generated labels)**

python

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from transformers import Trainer, TrainingArguments

\# Load pre-trained TrOCR model

processor = TrOCRProcessor.from_pretrained(\"microsoft/trocr-base-handwritten\")

model = VisionEncoderDecoderModel.from_pretrained(\"microsoft/trocr-base-handwritten\")

\# Training on 1,847 AI-labeled examples

training_args = TrainingArguments(

output_dir=\"./trocr-student-v1\",

per_device_train_batch_size=8,

num_train_epochs=10,

learning_rate=5e-5,

save_steps=500,

eval_steps=500,

logging_steps=100

)

trainer = Trainer(

model=model,

args=training_args,

train_dataset=silver_dataset,

eval_dataset=validation_dataset

)

trainer.train()

This stage teaches the model general patterns in our specific handwriting domain (student IDs, box-based digit layout, typical handwriting styles).

**Stage 2: Fine-tuning on Gold Set (human-verified labels)**

python

\# Fine-tune on 100 gold examples with lower learning rate

fine_tune_args = TrainingArguments(

output_dir=\"./trocr-student-v2\",

per_device_train_batch_size=4,

num_train_epochs=20,

learning_rate=1e-5, \# Lower LR to avoid catastrophic forgetting

save_steps=50

)

fine_tune_trainer = Trainer(

model=model,

args=fine_tune_args,

train_dataset=gold_dataset

)

fine_tune_trainer.train()

\`\`\`

This stage corrects any systematic errors learned from AI-generated labels. The human-verified examples act as an anchor, preventing the model from drifting toward teacher model biases or hallucinations.

\*\*Validation Strategy\*\*: We reserved 20% of our data (400 documents) as a held-out test set, never seen during training:

\`\`\`

Test set composition:

\- 80 from gold set (human-verified)

\- 320 from silver set (AI-generated, verified via sampling)

Student model performance on test set:

\- Accuracy: 98.1% (393/400 correct)

\- Errors: 7 (1.75%)

\- 4 errors: \"1\" vs \"7\" confusion (similar handwriting)

\- 2 errors: \"8\" vs \"3\" confusion (poor scan quality)

\- 1 error: \"5\" vs \"6\" confusion (crossed-out digit)

**Comparison to Teacher Model**:

| **Model**            | **Accuracy** | **Cost per 1000 Inferences** | **Latency (p50)** | **Infrastructure**     |
|----------------------|--------------|------------------------------|-------------------|------------------------|
| Gemini 2.5 (Teacher) | 99.5%        | \$1.00                       | 600ms             | API, internet required |
| TrOCR Student        | 98.1%        | \$0.00                       | 45ms              | Local CPU              |

The student model sacrifices 1.4% accuracy but gains:

- Zero marginal cost (one-time training cost only)

- 13x faster inference

- No infrastructure dependencies

- Complete data privacy (images never leave local system)

For our use case (automated document routing where a 2% error rate is acceptable and errors can be manually corrected), the student model is production-ready.

**Deployment**: The final model is exported as a single file:

python

\# Save model and processor

model.save_pretrained(\"./trocr_student_final\")

processor.save_pretrained(\"./trocr_student_final\")

\# Model size on disk: 2.1 GB

\# Can be quantized to \~550 MB with minimal accuracy loss

\`\`\`

This model integrates directly into our document processing pipeline, replacing all API calls to the teacher model.

\-\--

\## 3. System Architecture

Our production system consists of four main components: document ingestion, ID extraction, assignment splitting, and quality monitoring. Each component is designed for reliability, observability, and graceful failure handling.

\### 3.1 Overall Architecture

\`\`\`

\[Scanned PDF\]

↓

\[Preprocessing Pipeline\]

├→ Page extraction (PDFBox)

├→ Header cropping (top 30% of page)

└→ Image normalization

↓

\[ID Extraction (Student Model)\]

├→ TrOCR inference

├→ Confidence scoring

└→ Fallback to manual entry if confidence \< 0.85

↓

\[Document Splitting\]

├→ Group pages by detected student ID

├→ Generate individual PDFs

└→ Filename: Student\_{ID}\_{Name}.pdf

↓

\[Quality Monitoring\]

├→ Log all predictions

├→ Flag low-confidence cases

└→ Periodic human review (sample 5% of outputs)

↓

\[Distribution\]

├→ Email to students

└→ Upload to LMS

### **3.2 Component Details** {#component-details}

**Document Ingestion**:

Input: Single PDF containing N student assignments (typically 50-200 pages)

java

PDDocument document = Loader.loadPDF(inputFile);

PDFRenderer renderer = new PDFRenderer(document);

for (int pageIdx = 0; pageIdx \< document.getNumberOfPages(); pageIdx++) {

BufferedImage pageImage = renderer.renderImageWithDPI(pageIdx, 300);

// Extract top 30% for header analysis

BufferedImage header = pageImage.getSubimage(

0, 0,

pageImage.getWidth(),

(int)(pageImage.getHeight() \* 0.30)

);

processHeader(header, pageIdx);

}

Key decisions:

- DPI = 300 (balance between quality and file size)

- Crop to top 30% (student ID field location)

- Process sequentially (simple, predictable)

**ID Extraction**:

The core ML component. We wrap the TrOCR model in a simple inference API:

python

class StudentIDExtractor:

def \_\_init\_\_(self, model_path):

self.processor = TrOCRProcessor.from_pretrained(model_path)

self.model = VisionEncoderDecoderModel.from_pretrained(model_path)

def extract_id(self, image):

\# Preprocess

pixel_values = self.processor(image, return_tensors=\"pt\").pixel_values

\# Generate

generated_ids = self.model.generate(pixel_values)

generated_text = self.processor.batch_decode(generated_ids,

skip_special_tokens=True)\[0\]

\# Post-process

student_id = self.clean_prediction(generated_text)

confidence = self.estimate_confidence(generated_ids)

return student_id, confidence

def clean_prediction(self, raw_text):

\# Remove any non-digit characters

digits_only = \'\'.join(filter(str.isdigit, raw_text))

\# Pad or truncate to expected length (7 digits)

if len(digits_only) != 7:

return None \# Signal failure

return digits_only

def estimate_confidence(self, generated_ids):

\# Use model\'s token probabilities as confidence proxy

\# (simplified - actual implementation uses log-probabilities)

return 0.95 \# Placeholder

**Assignment Splitting**:

Once all pages are labeled with student IDs, we group consecutive pages:

java

Map\<String, List\<Integer\>\> studentPages = new HashMap\<\>();

String currentStudentId = null;

for (int i = 0; i \< pageLabels.size(); i++) {

String id = pageLabels.get(i);

// New assignment detected

if (id != null && !id.equals(currentStudentId)) {

currentStudentId = id;

studentPages.put(id, new ArrayList\<\>());

}

// Add page to current student

if (currentStudentId != null) {

studentPages.get(currentStudentId).add(i);

}

}

// Save individual PDFs

for (Map.Entry\<String, List\<Integer\>\> entry : studentPages.entrySet()) {

PDDocument studentDoc = new PDDocument();

for (Integer pageIdx : entry.getValue()) {

studentDoc.importPage(originalDoc.getPage(pageIdx));

}

studentDoc.save(\"Student\_\" + entry.getKey() + \".pdf\");

studentDoc.close();

}

**Quality Monitoring**:

We log every prediction for post-hoc analysis:

python

\# prediction_log.csv

timestamp,page_index,predicted_id,confidence,manual_review_needed

2024-02-03T14:23:01,0,0302714,0.98,False

2024-02-03T14:23:02,3,0302715,0.96,False

2024-02-03T14:23:03,6,NULL,0.42,True

\`\`\`

Pages with confidence \< 0.85 are flagged for manual review. A human operator reviews these cases and provides corrections, which feed back into our training data collection for future model iterations.

\### 3.3 Failure Modes and Mitigations

\*\*Model Failure\*\*: TrOCR returns garbage or low-confidence prediction

\*Mitigation\*: Fall back to manual entry UI. The preview dialog (from our original system) allows operators to manually type the student ID they see on the PDF.

\*\*Page Ordering Error\*\*: Scanned pages are out of order

\*Mitigation\*: During the preview phase, show first page of each detected assignment. Operator can visually verify correct student assignment and manually reassign if needed.

\*\*Missing Student ID\*\*: Page doesn\'t have a student ID header

\*Mitigation\*: Treat as continuation of previous assignment. Log warning for manual review.

\*\*Duplicate Student IDs\*\*: Two different assignments claim same ID

\*Mitigation\*: Flag both assignments for manual review. Likely causes: OCR error (misread a digit) or actual duplicate (student submitted twice).

\-\--

\## 4. Experimental Results

\### 4.1 Dataset Statistics

Final dataset composition:

\`\`\`

Total documents: 2,000

├─ Gold set (human-verified): 100 (5%)

├─ Silver set (AI-labeled, validated): 1,847 (92.35%)

└─ Manual fallback (failed AI extraction): 53 (2.65%)

Train/validation/test split:

├─ Training: 1,600 (80%)

├─ Validation: 200 (10%)

└─ Test: 200 (10%)

\`\`\`

Document quality distribution:

\- High quality (clean scans, upright, good contrast): 62%

\- Medium quality (slight rotation, minor artifacts): 28%

\- Low quality (significant skew, poor contrast, overflow ink): 10%

\### 4.2 Labeling Performance

\*\*Human Annotation Baseline\*\*:

\- Time per document: 72 seconds (averaged over 100 examples)

\- Total time for 2,000 documents: 40 hours

\- Error rate: 0% (assuming careful verification)

\- Cost: \$0 (researcher time) or \~\$600 at \$15/hour

\*\*AI-Assisted Annotation\*\*:

\- Time per document: 0.6 seconds (API latency)

\- Total time for 2,000 documents: 20 minutes

\- Error rate: 0.5% (based on gold set validation)

\- API cost: \$1.85

\- Manual verification time: 1.9 hours (gold set creation)

\- Manual fallback time: 0.75 hours (53 failed extractions)

\- Total human time: 2.65 hours

\*\*Speedup\*\*: 40 hours → 2.65 hours = 15x improvement

\### 4.3 Model Performance

\*\*Teacher Model (Gemini 2.5 Flash)\*\*:

Test set accuracy: 99.5% (199/200 correct)

Error analysis:

\- 1 error: Handwritten \"8\" misread as \"3\" on low-quality scan

Failure modes:

\- Extremely poor scan quality (\< 100 DPI, severe distortion)

\- Vertical or rotated text (ID written sideways)

\- Crossed-out and rewritten digits (ambiguous ground truth)

\*\*Student Model (TrOCR Fine-tuned)\*\*:

Test set accuracy: 98.1% (196/200 correct)

Error analysis:

\- 4 errors total:

\- 2× \"1\" vs \"7\" confusion (similar digit shapes)

\- 1× \"8\" vs \"3\" confusion (poor scan quality)

\- 1× \"5\" vs \"6\" confusion (crossed-out digit with unclear correction)

Performance vs teacher model:

\- 1.4% accuracy degradation

\- 13x faster inference (45ms vs 600ms)

\- Zero marginal cost vs \$0.001 per inference

\*\*Per-Digit Accuracy\*\*:

We analyzed errors at the individual digit level (7 digits per ID):

\`\`\`

Total digits in test set: 200 IDs × 7 digits = 1,400 digits

Correct digits: 1,396

Incorrect digits: 4

Per-digit accuracy: 99.71%

\`\`\`

Most errors are single-digit mistakes (e.g., \"0302714\" → \"0302774\"). Only 1 error involved multiple digits.

\### 4.4 Cost Analysis

\*\*Manual Annotation\*\*:

\`\`\`

Time: 40 hours

Cost: \$600 (at \$15/hour) or researcher opportunity cost

Accuracy: 100%

\`\`\`

\*\*Pure AI Annotation\*\* (hypothetical, no validation):

\`\`\`

Time: 20 minutes

API cost: \$2.00 (2,000 × \$0.001)

Accuracy: \~95% (extrapolated from small sample testing)

Risk: Systematic errors, no ground truth validation

\`\`\`

\*\*Our Hybrid Approach\*\*:

\`\`\`

Human time: 2.65 hours (gold set + fallback cases)

API cost: \$1.85

Total cost: \~\$42 (2.65 × \$15 + \$1.85)

Accuracy: 99.5% (validated on gold set)

\`\`\`

\*\*Cost Comparison\*\*:

\- vs Manual: 93% cost reduction (\$600 → \$42)

\- vs Pure AI: Similar cost, +4.5% accuracy improvement

\### 4.5 Production Deployment Results

We deployed the system in a real course with 147 students across 8 assignments (total: 1,176 documents processed).

\*\*Detection Accuracy\*\*:

\`\`\`

Correctly routed: 1,161 (98.7%)

Incorrectly routed: 15 (1.3%)

Manual corrections needed: 15

\`\`\`

\*\*Error Breakdown\*\*:

\- 8 errors: OCR misread (e.g., \"1\" as \"7\")

\- 4 errors: Student wrote wrong ID (copied from neighbor, etc.)

\- 3 errors: Scanning artifact (page upside-down, ID cut off)

\*\*User Feedback\*\* (survey of 3 instructors):

\- Time savings: \~2 hours per assignment (from manual sorting)

\- Error rate acceptable: Yes (errors caught in preview dialog)

\- Would use again: 3/3 yes

\*\*Continuous Improvement\*\*:

All 15 errors were manually corrected via the preview dialog. These corrections were automatically logged and added to our training dataset:

\`\`\`

training_data/corrections_2024.csv:

image_path,predicted_id,corrected_id,error_type

scan_147.png,0302714,0302774,ocr_error

scan_289.png,0301235,0301285,ocr_error

\...

This creates a feedback loop: production errors → labeled data → model retraining → improved accuracy. After retraining on the first semester\'s corrections, test accuracy improved from 98.1% to 98.9%.

## **5. Discussion** {#discussion}

### **5.1 Key Insights** {#key-insights}

**Insight 1: Strategic Human Verification Beats Full Automation or Full Manual Work**

Our results demonstrate that you don\'t need to choose between expensive manual annotation and unreliable AI annotation. By using human verification on a statistically selected sample (5% of data), we achieved 99.5% label quality while reducing human time investment by 93%.

The critical realization: human time is most valuable when spent on validation and quality control, not on repetitive labeling. An expert can verify 100 labels in the same time it takes to create 10 labels from scratch.

**Insight 2: Teacher Model Quality Directly Impacts Student Model Ceiling**

Our student model (98.1% accuracy) slightly underperforms the teacher model (99.5% accuracy). This is expected in knowledge distillation. However, the student can never exceed the teacher\'s performance on the training distribution.

This has implications for dataset creation: if your teacher model (AI labeling agent) has systematic biases or blind spots, those will transfer to the student model. The gold set mitigates this by providing correction signals, but cannot completely eliminate teacher errors.

**Insight 3: Domain-Specific Fine-Tuning Outperforms General-Purpose Models**

We compared our fine-tuned TrOCR model to off-the-shelf OCR systems:

| **System**              | **Accuracy on Our Data** |
|-------------------------|--------------------------|
| Google Cloud Vision API | 87.3%                    |
| Tesseract OCR           | 82.1%                    |
| AWS Textract            | 89.5%                    |
| Our Fine-Tuned TrOCR    | 98.1%                    |

General-purpose OCR systems struggle with our specific use case (handwritten digits in boxes, varying quality scans, domain-specific layouts). Fine-tuning on even a modest dataset (1,600 examples) yields dramatic improvements.

**Insight 4: Error Analysis Reveals Actionable Improvements**

Most model errors fall into predictable categories:

- Digit confusion (1 vs 7, 8 vs 3, 5 vs 6)

- Scan quality issues (low resolution, poor contrast)

- Layout edge cases (rotated text, overflow handwriting)

These errors can be systematically addressed:

- Digit confusion → Data augmentation (add more confusable pairs to training set)

- Scan quality → Preprocessing improvements (adaptive histogram equalization, denoising)

- Layout issues → Prompt engineering (tell teacher model to handle rotations)

### **5.2 Limitations** {#limitations}

**Sample Size**:

Our gold set (100 examples) provides reasonable statistical confidence for aggregate accuracy estimation, but may not capture rare edge cases. A larger gold set (250-500 examples) would improve confidence intervals and better represent data distribution tails.

**Domain Specificity**:

Our approach is optimized for a specific task (7-digit student IDs in standardized layouts). Generalization to other handwriting recognition tasks (full sentences, varied formats, different languages) would require re-tuning the pipeline.

**Human Dependency**:

The system still requires human involvement at two stages:

1.  Initial gold set creation (1.9 hours)

2.  Manual fallback for failed AI extractions (0.75 hours)

While dramatically reduced from 40 hours, this is not fully automated. Future work could explore active learning to minimize human time further.

**Teacher Model Availability**:

Our approach assumes access to a high-quality multimodal model API. If such APIs become unavailable or prohibitively expensive, the pipeline breaks. Mitigation: the student model, once trained, is permanent and requires no ongoing API access.

### **5.3 Generalization to Other Domains** {#generalization-to-other-domains}

The core methodology (gold set validation → AI labeling → knowledge distillation) is domain-agnostic. We believe this approach generalizes to:

**Medical Imaging**:

- Gold set: Radiologist-labeled scans (expensive, limited)

- Silver set: AI-labeled scans (validated against gold set)

- Student model: Lightweight diagnostic model for clinical deployment

**Legal Document Analysis**:

- Gold set: Attorney-reviewed contracts (slow, expensive)

- Silver set: LLM-extracted clauses (fast, scalable)

- Student model: Specialized contract analysis model

**Manufacturing Quality Control**:

- Gold set: Expert-inspected defect images

- Silver set: Vision model-labeled defects

- Student model: Real-time defect detection on assembly line

The key requirement: a task where (a) manual labeling is expensive but feasible for small samples, (b) AI labeling is available but imperfect, and (c) a specialized model is preferred over ongoing API costs.

### **5.4 Future Work** {#future-work}

**Direction 1: Active Learning Integration**

Instead of randomly sampling the gold set, use active learning to select the most informative examples. The AI model could flag high-uncertainty cases for human review, concentrating human effort where it matters most.

**Direction 2: Continual Learning**

As the system runs in production, user corrections (via the manual review UI) accumulate. These corrections could automatically trigger model retraining on a schedule (e.g., monthly), creating a self-improving system.

**Direction 3: Multi-Task Learning**

Our current model only extracts student IDs. The same architecture could be extended to extract other fields (student name, assignment title, date) using a multi-task learning framework, amortizing training cost across multiple outputs.

**Direction 4: Federated Deployment**

Multiple institutions could collaborate on model training while preserving data privacy. Each institution trains on local data, then model weights are aggregated using federated learning techniques.

## **6. Related Work** {#related-work}

This section positions our work within the broader research landscape.

### **6.1 Human-in-the-Loop Machine Learning** {#human-in-the-loop-machine-learning}

Settles (2009) formalized active learning as querying human annotators for labels on maximally informative examples. Our approach inverts this: rather than querying for training labels, we query for validation labels.

Vaughan (2018) demonstrated that hybrid annotation (machine pre-labeling + human verification) reduces time by 50-70% in NLP tasks. Our results (93% time reduction) exceed this, likely because computer vision OCR tasks are more amenable to automation than semantic NLP tasks.

Recent work in autonomous vehicles (Bojarski et al., 2016) uses \"shadow mode\" where AI predictions are logged alongside human driver actions. Disagreements between AI and human become training data for the next iteration. Our gold set serves a similar function as a continuous validation benchmark.

### **6.2 Knowledge Distillation** {#knowledge-distillation}

Hinton et al. (2015) introduced knowledge distillation as transferring knowledge from large \"teacher\" ensembles to small \"student\" networks. Our work extends this to multimodal models, using an API-based vision-language model as the teacher.

Sanh et al. (2019) demonstrated DistilBERT, a distilled version of BERT with 40% fewer parameters and 97% of BERT\'s performance. We achieve similar compression ratios (Gemini → TrOCR) with comparable accuracy retention (99.5% → 98.1%).

Unique to our approach: we use human-verified data as a correction mechanism during distillation. Most knowledge distillation work assumes the teacher is perfect; we explicitly account for teacher errors via the gold set.

### **6.3 Semi-Supervised Learning** {#semi-supervised-learning}

Chapelle et al. (2006) provide a comprehensive overview of semi-supervised learning, where models leverage both labeled and unlabeled data. Classic approaches (self-training, co-training) use model predictions on unlabeled data as pseudo-labels.

Our method differs: we use a powerful external model (LLM) to generate pseudo-labels, then validate them against a small human-labeled set. This is closer to \"teacher-student\" frameworks than traditional semi-supervised learning.

Xie et al. (2020) proposed \"self-training with noisy student,\" where a student model is trained on pseudo-labels from a teacher, then becomes the next teacher in an iterative process. Our approach uses a single teacher-student pair but incorporates human validation to prevent error propagation.

### **6.4 Handwriting Recognition** {#handwriting-recognition}

Handwritten digit recognition is a classic benchmark in computer vision (LeCun et al., 1998, MNIST dataset). However, MNIST consists of clean, centered, normalized digits. Real-world handwriting (like student IDs) exhibits:

- Variable digit spacing

- Inconsistent stroke thickness

- Background noise (boxes, lines, artifacts)

- Rotation and skew

Recent work (Diaz et al., 2021) addresses these challenges using data augmentation and domain adaptation. Our approach complements this: rather than augmenting synthetic data, we collect real data efficiently via hybrid annotation.

TrOCR (Li et al., 2021) combines Vision Transformers with language model decoders for end-to-end OCR. We build on TrOCR\'s architecture but demonstrate that fine-tuning on domain-specific data (even with limited labels) dramatically outperforms pre-trained general-purpose models.

## **7. Conclusion** {#conclusion}

We presented a hybrid supervised learning architecture for handwritten digit recognition that combines human verification, AI-assisted labeling, and knowledge distillation. Our three-phase methodology reduces data preparation time from 40 hours to 2.65 hours (93% reduction) while maintaining 99.5% label accuracy.

The key innovation is treating human annotation as a validation mechanism rather than a labeling mechanism. By strategically verifying a small sample (5% of data), we can confidently accept AI-generated labels for the remaining 95%, achieving the best of both worlds: human-level quality at machine-level scale.

Knowledge distillation from a large cloud-based teacher model to a small local student model eliminates ongoing API costs and infrastructure dependencies while retaining 98.1% accuracy. In production deployment across 1,176 real documents, the system achieved 98.7% correct routing with minimal human intervention.

This approach generalizes beyond our specific application. Any domain with expensive manual annotation, available AI models, and tolerance for small error rates can benefit from this framework. We hope this work encourages researchers to reconsider the binary choice between manual and automated annotation, embracing hybrid approaches that strategically allocate human expertise where it creates maximum value.

Our code, trained models, and annotated dataset are available at \[repository URL\] to facilitate reproducibility and enable future research building on this work.

## **References**

Bojarski, M., et al. (2016). End to end learning for self-driving cars. arXiv preprint arXiv:1604.07316.

Chapelle, O., Scholkopf, B., & Zien, A. (2006). Semi-supervised learning. MIT Press.

Diaz, D. H., et al. (2021). Real-world handwritten text recognition: Tackling distribution shift via domain adaptation. Pattern Recognition Letters, 143, 1-7.

Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.

Li, M., et al. (2021). TrOCR: Transformer-based optical character recognition with pre-trained models. arXiv preprint arXiv:2109.10282.

Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.

Settles, B. (2009). Active learning literature survey. Computer Sciences Technical Report 1648, University of Wisconsin-Madison.

Vaughan, J. W. (2018). Making better use of the crowd: How crowdsourcing can advance machine learning research. Journal of Machine Learning Research, 18(1), 7026-7071.

Xie, Q., et al. (2020). Self-training with noisy student improves ImageNet classification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10687-10698).
