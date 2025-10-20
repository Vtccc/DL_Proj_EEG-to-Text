<h1 align="center"> Prompting the Mind: EEG-to-Text Translation
with Multimodal LLMs and Semantic Control </h1>

[**Mohammed Salah Al-Radhi**](https://malradhi.github.io/),  [**Sadi Mahmud Shurid**](#), [**Géza Németh**](https://scholar.google.ro/citations?user=Qf5PHwoAAAAJ&hl=en/)

Department of Telecommunications and Artificial Intelligence, Budapest University of Technology and Economics, Budapest, Hungary

### 📑 Table of Contents
- ⚡ [How to Reproduce](#how-to-reproduce)
- 🙏 [Acknowledgments](#acknowledgments)

---

## How to Reproduce

Follow these steps to set up and run the project:

---

1️⃣ Clone the Repository

Clone the original repository into your working directory:

```bash
git clone https://github.com/Sadi-Mahmud-Shurid/PTM.git
cd PTM
```

2️⃣ Download Preprocessed Data

Download the preprocessed EEG dataset from the link below and place it in the project directory:

📦 [Google Drive – Preprocessed EEG Data](https://drive.google.com/drive/folders/1XqV6MMl28iYXkQBMEFHfEXllGmCbqpOu)

3️⃣ Install Dependencies

Install all required Python dependencies:  

```bash
pip install -r requirements.txt
```
⚠️ Note: The program may prompt you to install different package versions than those listed in requirements.txt. Follow these prompts to ensure compatibility and prevent errors in later stages.

4️⃣ Train EEG Encoder

Run the following command to align the EEG encoder with CLIP embeddings:  

```bash
python train_eeg_classifier.py \
    --eeg_dataset data/block/eeg_55_95_std.pth \
    --splits_path data/block/block_splits_by_image_all.pth \
    --output ./eeg_encoder_55-95_40_classes \
    --image_dir data/images/
```
5️⃣ Fine-Tune LLM

Run the fine-tuning script to align EEG representations with the chosen LLM backbone (DeepSeek-LLM 7B in this example):

```bash
python finetune_llm.py \
    --eeg_dataset data/block/eeg_55_95_std.pth \
    --splits_path data/block/block_splits_by_image_all.pth \
    --eeg_encoder_path ./eeg_encoder_55-95_40_classes \
    --image_dir data/images/ \
    --output "deepseek_eeg_model_7B_base" \
    --llm_backbone_name_or_path "deepseek-ai/deepseek-llm-7b-base" \
    --load_in_8bit \
    --bf16
```
6️⃣ Run Inference
Use the trained model to generate text from EEG signals:

```bash
python inference.py \
    --model_path "deepseek_eeg_model_7B_base/" \
    --eeg_dataset data/block/eeg_55_95_std.pth \
    --image_dir data/images/ \
    --dest "deepseek_eeg_model_7B_base_results.csv" \
    --splits_path data/block/block_splits_by_image_all.pth
```
7️⃣ Evaluation

To run the evaluation, execute the `metrics_based_evaluation_notebook`.


## Acknowledgments
This repository builds upon and extends the excellent work by Abhijit Mishra and collaborators in the Thought2Text project.

💻 Foundational Codebase: [github.com/abhijitmishra/Thought2Text](https://github.com/abhijitmishra/Thought2Text.git)
