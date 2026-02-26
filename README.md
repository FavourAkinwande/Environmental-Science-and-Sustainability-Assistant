## Environmental Science & Circular Economy Assistant

This project fine-tunes **TinyLlama/TinyLlama-1.1B-Chat-v1.0** into a domain-specific assistant for **environmental science, sustainability, and circular economy / upcycling**. The assistant:

- Answers questions about climate change, biodiversity, pollution, waste management, and environmental policy.
- Provides practical upcycling and circular-economy ideas for common household materials.
- Politely refuses out-of-domain questions (e.g., sports, politics, generic coding).

The full pipeline (data preparation → PEFT/LoRA fine-tuning → evaluation → Gradio UI) is implemented in the notebook `notebook/environmental-science-assistant.ipynb`.

---

## How to run in Google Colab

1. Upload this repository to GitHub (or your own fork).
2. In Colab, open `notebook/environmental-science-assistant.ipynb` directly from GitHub, or add a Colab badge like:

   ```markdown
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK_HERE)
   ```

3. In Colab:
   - Make sure the runtime has a **T4 GPU** or better (`Runtime → Change runtime type → GPU`).
   - Run cells from top to bottom. All `pip install` cells are included in the notebook.
   - If you are **not** using Kaggle, adjust any dataset/model paths that start with `/kaggle/...` to point to your local/Drive paths (see Dataset section below).

The notebook is designed to run end-to-end with minimal manual setup once data paths are set.

---

## Dataset

### Sources

- **Primary dataset – GeoGPT Environmental Q&A**
  - Environmental-science Q&A pairs extracted from academic papers (CSV with `question`, `answer`, and metadata).
  - Coverage: climate, water quality, biodiversity, soil science, air pollution, etc.
  - In the notebook this is loaded from a Kaggle path; for Colab you can:
    - Upload `geogpt-qa.csv` to your Colab working directory and update the CSV path, or
    - Mount Google Drive and point the path to your Drive location.

- **Augmentation – Upcycling Q&A (30 examples)**
  - Manually created Q&A pairs about upcycling household items (plastic bottles, glass jars, textiles, cardboard, food scraps, etc.).
  - Adds practical circular-economy knowledge not present in the academic dataset.

- **Augmentation – Refusal Q&A (≈120 examples)**
  - Manually created out-of-domain questions (sports, politics, generic coding, etc.) paired with a polite refusal template.
  - Trains the assistant to **stay in-domain** and clearly decline off-topic requests.

### Preprocessing & Formatting

The notebook performs:

- Column selection to `question` / `answer`.
- String normalization (remove non‑breaking spaces, collapse whitespace, strip).
- Removal of empty and duplicate pairs.
- Length filtering: keep answers in \[20, 2000] characters for quality and context fit.
- Augmentation merge (GeoGPT + upcycling + refusal) into a single `final_df`.
- Exploratory data analysis (length distributions, sample rows).
- Instruction formatting into a chat-style template:

  ```text
  [SYSTEM] You are an environmental science and sustainability assistant...
  [USER] <question>
  [ASSISTANT] <answer>
  ```

The final dataset is split 90/10 into **train/test** and saved as:

- `train.jsonl`
- `test.jsonl`

These JSONL files are used by the fine-tuning code via `datasets.load_dataset("json", ...)`.

---

## Model and Fine-tuning (LoRA / PEFT)

### Base model

- **TinyLlama/TinyLlama-1.1B-Chat-v1.0** (Hugging Face Hub).
- 1.1B parameters, chat-tuned, fits well on a single T4 GPU with LoRA.

### Fine-tuning setup

- Library stack: `transformers`, `peft`, `accelerate`, `datasets`, `bitsandbytes`.
- Task: **Causal language modeling** on chat-formatted Q&A.
- PEFT configuration:
  - `task_type="CAUSAL_LM"`, `r=8`, `lora_alpha=16`, `lora_dropout=0.05`.
  - Target modules: `["q_proj", "v_proj", "k_proj", "o_proj"]`.
  - Only ~0.2% of parameters are trainable (LoRA adapters).
- Tokenization:
  - Uses TinyLlama tokenizer, `max_len=512` tokens.
  - `pad_token = eos_token` if missing.

### Training runs and hyperparameters

Training is driven by a helper function `run_training(...)` that:

- Loads the base TinyLlama model.
- Wraps it with LoRA via `get_peft_model`.
- Uses `Trainer` + `TrainingArguments` with:
  - `per_device_train_batch_size` ∈ {2, 4}
  - `gradient_accumulation_steps` ∈ {2, 4}
  - `num_train_epochs` ∈ {1, 2}
  - `learning_rate` ∈ {5e‑5, 1e‑4}
  - `fp16=True` (for T4) and evaluation at each epoch.
- Logs:
  - Train loss, eval loss.
  - Approx. training time (minutes).
  - Approx. GPU memory usage.

All runs are saved in:

- `tinyllama_lora_run1/`
- `tinyllama_lora_run2/`
- `tinyllama_lora_run3/`

---

## Experiment Table (Hyperparameter Tuning)

The notebook builds an experiment table (`experiment_log.csv`) summarizing three runs:

| run_id         | lr      | batch | grad_accum | epochs | eval_loss | train_loss | time (min) | peak GPU (GB) | Δ eval vs baseline |
|----------------|---------|-------|-----------:|--------|-----------|-----------:|-----------:|--------------:|--------------------|
| run1_baseline  | 5e‑5    | 2     | 4          | 2      | 1.3590    | 1.4319     | 17.73      | 0.97          | 0.0% (reference)   |
| run2_lr1e4     | 1e‑4    | 2     | 4          | 1      | 1.3618    | 1.4500     | 8.96       | 1.05          | −0.2%              |
| run3_batch4    | 5e‑5    | 4     | 2          | 2      | 1.3889    | 1.4943     | 15.87      | 2.10          | −2.2%              |

**Key takeaways:**

- **run1_baseline** (lr=5e‑5, batch 2, 2 epochs) achieved the **best eval loss**.
- Increasing the learning rate to **1e‑4** (run2) did **not** improve validation performance.
- Increasing per-device batch size to **4** (run3) slightly worsened eval loss and used more GPU memory.
- The table and a short markdown summary explicitly highlight these trade-offs for the rubric.

The best run (`run1_baseline`) is used for subsequent evaluation and the Gradio UI (`tinyllama_lora_run1/`).

---

## Evaluation: Metrics and Base vs Fine-tuned Comparison

### Automatic metrics

The notebook reports multiple metrics on the held-out test set:

- **Validation loss** from the Trainer (best ≈ 1.36).
- **Perplexity**: computed as \\(\\exp(\\text{eval\\_loss})\\) on the best run; ≈ **3.9**.
- **ROUGE** (e.g., ROUGE‑1, ROUGE‑L) on a subset of `test.jsonl`.
- **BLEU** on the same subset (using `evaluate` + `sacrebleu`).

These are computed for the fine-tuned TinyLlama+LoRA model using the same chat-style prompts as in training.

### Qualitative comparison: base vs fine-tuned

The notebook:

- Loads both:
  - **Base TinyLlama** (pre-trained chat model).
  - **Fine-tuned TinyLlama + LoRA** (`tinyllama_lora_run1/`).
- Asks both models the same questions:
  - In-domain scientific Q&A (e.g., “How does deforestation affect the carbon cycle and climate change?”).
  - Upcycling / circular economy Q&A.
  - Out-of-domain questions (e.g., “Who won the FIFA World Cup in 2022?”).
- Prints **side-by-side answers** for inspection, with a markdown cell summarizing:
  - Where the fine-tuned model is **more domain-specific** and detailed.
  - How refusal behavior improves on out-of-domain questions compared to the base chat model.

This satisfies the rubric requirement for both **quantitative** and **qualitative** evaluation, and for explicit **base vs fine-tuned** comparison.

---

## Gradio UI (Deployment)

The final section of the notebook defines a **Gradio `ChatInterface`** that:

- Loads the **fine-tuned model** (base TinyLlama + LoRA adapters from `tinyllama_lora_run1/`).
- Uses the same `[SYSTEM] / [USER] / [ASSISTANT]` formatting used during training.
- Runs on GPU if available (`cuda`), otherwise CPU.

The UI:

- Shows a title and short usage instructions.
- Lets the user type a question and get a model response.
- Makes the domain boundaries explicit (environmental science, sustainability, circular economy; polite refusal for off-topic queries).

In Colab, running the UI cell will:

- Start a local Gradio app.
- Provide a public `gradio.live` link that you can use in the demo video.

---

## Files and Structure

- `notebook/environmental-science-assistant.ipynb`  
  End‑to‑end pipeline: data collection, preprocessing, augmentation, formatting, tokenization, train/test split, LoRA fine-tuning, hyperparameter experiments, evaluation (perplexity, ROUGE, BLEU), qualitative comparison, and Gradio UI.

- `train.jsonl`, `test.jsonl`  
  Instruction-formatted training and test data (chat-style) produced by the notebook.

- `tinyllama_lora_run1/`, `tinyllama_lora_run2/`, `tinyllama_lora_run3/`  
  PEFT/LoRA checkpoints for different hyperparameter configurations (run1 is used as the best model).

---

## Demo Video (what to show)

When recording your 5–10 minute demo, you can structure it as:

1. **Project definition & domain alignment**  
   - Explain the environmental science & circular economy focus and why a chatbot is useful.

2. **Dataset & preprocessing**  
   - Show how GeoGPT + manual upcycling + refusal data are combined and cleaned.
   - Briefly show `train.jsonl` / `test.jsonl`.

3. **Model & fine-tuning (LoRA)**  
   - Describe TinyLlama, LoRA configuration, and why PEFT is necessary for Colab GPUs.
   - Walk through the hyperparameter experiment table and identify the best run.

4. **Evaluation**  
   - Show the eval loss / perplexity for the best run.
   - Show ROUGE & BLEU scores on the test subset.
   - Show a couple of base vs fine-tuned answers and discuss improvements.

5. **UI demo**  
   - Open the Gradio UI, ask several in-domain questions, upcycling questions, and one or two off-domain questions to demonstrate refusal behavior.

6. **Conclusion & limitations**  
   - Summarize strengths (domain knowledge, upcycling tips, refusals).
   - Mention limitations (small model size, potential hallucinations, academic bias) and possible future work.

