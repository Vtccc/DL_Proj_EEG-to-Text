# DL_Proj_EEG-to-Text

## Team Information
### Team Name
#### EEG-to-Text Translation Using the Thought2Text Framework
### Team Mumber
#### 1. CHEN, Jiayu; Neptun Code: PUOOOZ
#### 2. Wang, Hongen; Neptun Code: FT7J14

## Project Description:
#### This project explores how brain signals (EEG) can be translated into natural text using state-of-the-art large language models. The baseline is the Thought2Text framework, which combines EEG encoders, multimodal alignment, and LLM fine-tuning. In this project, we’ll experiment with the DeepSeek-7B model instead of the original Mistral-7B to see how model choice impacts accuracy and descriptiveness.As a small project for Deeplearning course, we attached the following link as reference for the trained model. Downloading these trained models can helping you reproduce our work precisly as any change in the configuration may change the results significantly and LLM implementations running on GPUs exhibit inherent non-determinism, leading to slight deviations in results across separate runs.
https://huggingface.co/datasets/Jasmine1122/data-and-classes/tree/main


## Description of the Files
datautils.py: Implements dataset classes and data splitting/filtering classes. These handle loading and processing EEG signals, image data, and text annotations to support model training and inference.

model.py: Defines the EEGModelForCausalLM class, which integrates an EEG encoder (eeg_encoder), a language model (llm), and a projection layer (mm_proj). It supports model loading, saving, and text generation, serving as the core structure for EEG-to-text conversion.

train_eeg_classifier.py: Trains the EEG encoder using a combination of contrastive loss (MSELoss) and classification loss (CrossEntropyLoss). This aligns EEG signal encodings with image embeddings and enables category prediction, featuring a custom EEGEncoderTrainer.

test_eeg_classifier.py: Evaluates the performance of the trained EEG encoder by calculating classification accuracy on the test set, assessing its ability to classify EEG signals.

inference.py / inference_only_eeg.py: Perform model inference. They load trained models, process EEG signals from the test set, generate corresponding image description texts, and save results as CSV files. The two may differ in input processing details.

inference_chance2.py: A variant of the inference script. It uses random tensors instead of EEG embeddings during generation, likely for baseline comparison to evaluate the model's reliance on valid EEG inputs.

finetune_llm.py: Fine-tunes the language model in stages. Stage 2 trains on image data using CLIP embeddings, while Stage 3 further adapts using EEG embeddings. It supports 8-bit quantization and LoRA (Low-Rank Adaptation) for efficient training.

loss.py: Defines loss functions including ContrastiveLoss and MSELoss, which are used to optimize model parameters during training.

config.py: Contains the EEGEncoderConfig class, storing configuration parameters for the EEG encoder (e.g., model dimensions, number of attention heads, layers) to support model construction.

pretrain_data_processor.py: Processes raw data from BDD and Mind BIG Data sources (including ImageNet and MNIST datasets) into .pt files. It normalizes the data and splits it into training, validation, and test sets for pretraining.

requirements.txt: Lists the Python libraries and their versions required for the project, facilitating environment setup.

## How to Reproduce

1.Download Preprocessed Data
Download the preprocessed EEG dataset from the link below and place it in the project directory:
https://drive.google.com/drive/folders/1XqV6MMl28iYXkQBMEFHfEXllGmCbqpOu

2.Clone the Repository

```bash
git clone https://github.com/Vtccc/DL_Proj_EEG-to-Text.git
```

3.Install Dependencies

```bash
pip install -r requirements.txt
```

4.EEG Encoder alignmment with CLIP embeddings

Run the following command to align the EEG encoder with CLIP embeddings: 

```python
python train_eeg_classifier.py \
    --eeg_dataset /content/drive/MyDrive/data/capstone/block/eeg_55_95_std.pth \
    --splits_path /content/drive/MyDrive/data/capstone/block/block_splits_by_image_all.pth \
    --output ./eeg_encoder_55-95_40_classes \
    --image_dir /content/drive/MyDrive/data/capstone/images/
```

The checkpoints for the encoder will be stored in './eeg_encoder_55-95_40_classes'.

Report Can be Found in
https://wandb.ai/1072356040-budapesti-m-szaki-s-gazdas-gtudom-nyi-egyetem/huggingface/reports/Copy-of-1072356040-s-EEG-Encoder-Train-Report--VmlldzoxNTA4MTQxNw

5.Fine-Tune LLM
```python
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
Upon completion, the traine model will be available under 'deepseek_eeg_model_7B_base' directory.

6.Run Inference
Use the trained model to generate text from EEG signals:

Prompt type could choose from "basic" "detailed" "creative" to get different inference results

```python
python inference.py \
  --model_path "deepseek_eeg_model_7B_base/" \
  --splits_path "../data/block/block_splits_by_image_all.pth" \
   --eeg_dataset "../data/block/eeg_55_95_std.pth" \
  --image_dir "../data/images/" \
  --prompt_type detailed \
  --batch_size 1 \
  --device cuda:0 \
  --results_dir ./results
```

7.Evaluation

We evaluate the model's generations through popular NLG metrics such as BLEU, METEOR and ROUGE.
To run the evaluation, execute the `eval` notebook.

## Acknowledgments
This repository builds upon and extends the excellent work by Abhijit Mishra and collaborators in the Thought2Text project. 
And also thanks to Sadi Mahmud Shurid and collaborators in PTM project.

Foundational Codebase: 
https://github.com/abhijitmishra/Thought2Text.git
https://github.com/Sadi-Mahmud-Shurid/PTM.git
