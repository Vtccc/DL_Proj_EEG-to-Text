# DL_Proj_EEG-to-Text

## Team Information
### Team Name
#### EEG-to-Text Translation Using the Thought2Text Framework
#### Team Number
#### 1. CHEN, Jiayu; Neptun Code: PUOOOZ
#### 2. Wang, Hongen; Neptun Code: FT7J14

## Project Description:
#### This project explores how brain signals (EEG) can be translated into natural text using state-of-the-art large language models. The baseline is the Thought2Text framework, which combines EEG encoders, multimodal alignment, and LLM fine-tuning. In this project, we’ll experiment with the DeepSeek-7B model instead of the original Mistral-7B to see how model choice impacts accuracy and descriptiveness.


## Description of the Files
#### datautils.py: Implements dataset classes (e.g., EEGDataset, EEGFineTuningDataset, EEGInferenceDataset) and data splitting/filtering classes (Splitter, SplitterFineTuning, Filter). These handle loading and processing EEG signals, image data, and text annotations to support model training and inference.

#### model.py: Defines the EEGModelForCausalLM class, which integrates an EEG encoder (eeg_encoder), a language model (llm), and a projection layer (mm_proj). It supports model loading, saving, and text generation, serving as the core structure for EEG-to-text conversion.

#### train_eeg_classifier.py: Trains the EEG encoder using a combination of contrastive loss (MSELoss) and classification loss (CrossEntropyLoss). This aligns EEG signal encodings with image embeddings and enables category prediction, featuring a custom EEGEncoderTrainer.

#### test_eeg_classifier.py: Evaluates the performance of the trained EEG encoder by calculating classification accuracy on the test set, assessing its ability to classify EEG signals.

#### inference.py / inference_only_eeg.py: Perform model inference. They load trained models, process EEG signals from the test set, generate corresponding image description texts, and save results as CSV files. The two may differ in input processing details.

#### inference_chance2.py: A variant of the inference script. It uses random tensors instead of EEG embeddings during generation, likely for baseline comparison to evaluate the model's reliance on valid EEG inputs.

#### finetune_llm.py: Fine-tunes the language model in stages. Stage 2 trains on image data using CLIP embeddings, while Stage 3 further adapts using EEG embeddings. It supports 8-bit quantization and LoRA (Low-Rank Adaptation) for efficient training.

#### loss.py: Defines loss functions including ContrastiveLoss and MSELoss, which are used to optimize model parameters during training.

#### config.py: Contains the EEGEncoderConfig class, storing configuration parameters for the EEG encoder (e.g., model dimensions, number of attention heads, layers) to support model construction.

#### pretrain_data_processor.py: Processes raw data from BDD and Mind BIG Data sources (including ImageNet and MNIST datasets) into .pt files. It normalizes the data and splits it into training, validation, and test sets for pretraining.

#### requirements.txt: Lists the Python libraries and their versions required for the project, facilitating environment setup.

#### commands.txt / run_inference_only_eeg.sh: Provide example commands or shell scripts for running training and inference scripts, simplifying task execution for users.


## How to run
#### 1. Open the DL_Proj_EGG2TXT.ipynb
#### 2. Download Preprocessed Data
#### Download the preprocessed EEG dataset from the link below and place it in the project directory:
#### https://drive.google.com/drive/folders/1XqV6MMl28iYXkQBMEFHfEXllGmCbqpOu
#### 3. Run python notebook

