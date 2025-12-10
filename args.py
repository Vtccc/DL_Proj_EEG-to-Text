import argparse
import os


def get_args_for_encoder_training():
    # Define options
    parser = argparse.ArgumentParser(description="Template")

    # Dataset options

    ### BLOCK DESIGN ###
    # Data
    parser.add_argument(
        "--eeg_dataset", default=None, help="EEG dataset path"
    )  # 5-95Hz
    parser.add_argument("--image_dir", default=None, help="ImageNet dataset path")
    # Splits
    parser.add_argument(
        "--splits_path", default=None, help="splits path"
    )  # All subjects
    ### BLOCK DESIGN ###
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save the model checkpoints and logs.",
    )

    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32")

    parser.add_argument(
        "-sn", "--split_num", default=0, type=int, help="split number"
    )  # leave this always to zero.

    # Subject selecting
    parser.add_argument(
        "-sub",
        "--subject",
        default=0,
        type=int,
        help="choose a subject from 1 to 6, default is 0 (all subjects)",
    )

    # Time options: select from 20 to 460 samples from EEG data
    parser.add_argument(
        "-tl", "--time_low", default=20, type=float, help="lowest time value"
    )
    parser.add_argument(
        "-th", "--time_high", default=460, type=float, help="highest time value"
    )
    # Training options
    parser.add_argument("--save_every", type=int, default=5)

    parser.add_argument("--device", type=str, default="cuda")

    # train args

    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of epochs for training."
    )
    parser.add_argument(
        "--save_steps",
        default=5000,
        type=int,
        help="Number of steps between saving checkpoints.",
    )
    parser.add_argument(
        "--logging_steps", default=30, type=int, help="Number of steps between logging."
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--optim",
        default="adamw_hf",
        type=str,
        help="Optimizer to use for training.",
    )
    parser.add_argument(
        "--weight_decay", default=0.001, type=float, help="Weight decay to apply."
    )
    parser.add_argument(
        "--max_grad_norm",
        default=0.3,
        type=float,
        help="Max gradient norm to clip gradients.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform.",
    )
    parser.add_argument(
        "--warmup_ratio",
        default=0.3,
        type=float,
        help="Ratio of total steps to perform linear learning rate warmup.",
    )
    parser.add_argument(
        "--group_by_length",
        action="store_true",
        help="Whether to group samples of roughly the same length together.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default="constant",
        type=str,
        help="Type of learning rate scheduler.",
    )
    # Parse arguments
    args = parser.parse_args()
    return args


def get_args_for_llm_finetuning():
    # Define options
    parser = argparse.ArgumentParser(description="Template")

    # Dataset options

    ### BLOCK DESIGN ###
    # Data
    parser.add_argument(
        "--eeg_dataset", default=None, help="EEG dataset path"
    )  # 5-95Hz
    parser.add_argument("--image_dir", default=None, help="ImageNet dataset path")
    # Splits
    parser.add_argument(
        "--splits_path", default=None, help="splits path"
    )  # All subjects
    ### BLOCK DESIGN ###
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save the model checkpoints and logs.",
    )
    parser.add_argument(
        "--llm_backbone_name_or_path",
        type=str,
        default="",
        help="Name or path of the image tower model.",
    )
    parser.add_argument(
        "--load_in_8bit", default=False, help="load LLM in 8 bit", action="store_true"
    )
    parser.add_argument(
        "--use_lora", default=False, help="load LLM in 8 bit", action="store_true"
    )
    parser.add_argument(
        "--no_stage2", default=False, help="Directly begin stage3", action="store_true"
    )
    parser.add_argument(
        "--eeg_encoder_path",
        type=str,
        required=True,
        help="Path to the fine-tuned EEG encoder",
    )
    parser.add_argument(
        "--saved_pretrained_model_path",
        type=str,
        default="/tmp",
        help="Directory to load the model checkpoints",
    )
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32")

    parser.add_argument(
        "-sn", "--split_num", default=0, type=int, help="split number"
    )  # leave this always to zero.

    # Subject selecting
    parser.add_argument(
        "-sub",
        "--subject",
        default=0,
        type=int,
        help="choose a subject from 1 to 6, default is 0 (all subjects)",
    )

    # Time options: select from 20 to 460 samples from EEG data
    parser.add_argument(
        "-tl", "--time_low", default=20, type=float, help="lowest time value"
    )
    parser.add_argument(
        "-th", "--time_high", default=460, type=float, help="highest time value"
    )
    # Training options
    parser.add_argument("--save_every", type=int, default=5)

    parser.add_argument("--device", type=str, default="cuda")

    # train args

    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for training."
    )
    parser.add_argument(
        "--num_epochs_image", type=int, default=6, help="Number of epochs for training."
    )
    parser.add_argument(
        "--num_epochs_eeg", type=int, default=6, help="Number of epochs for training."
    )
    parser.add_argument(
        "--save_steps",
        default=5000,
        type=int,
        help="Number of steps between saving checkpoints.",
    )
    parser.add_argument(
        "--logging_steps", default=30, type=int, help="Number of steps between logging."
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=16,
        type=int,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--optim",
        default="adamw_hf",
        type=str,
        help="Optimizer to use for training.",
    )
    parser.add_argument(
        "--weight_decay", default=0.001, type=float, help="Weight decay to apply."
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed precision) training.",
    )
    parser.add_argument(
        "--bf16", action="store_true", help="Whether to use bfloat16 training."
    )
    parser.add_argument(
        "--max_grad_norm",
        default=0.3,
        type=float,
        help="Max gradient norm to clip gradients.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform.",
    )
    parser.add_argument(
        "--warmup_ratio",
        default=0.2,
        type=float,
        help="Ratio of total steps to perform linear learning rate warmup.",
    )
    parser.add_argument(
        "--group_by_length",
        action="store_true",
        help="Whether to group samples of roughly the same length together.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default="constant",
        type=str,
        help="Type of learning rate scheduler.",
    )
    parser.add_argument(
        "--report_to",
        default="tensorboard",
        type=str,
        help="Where to report training metrics.",
    )
    # Parse arguments
    args = parser.parse_args()
    return args


def get_args_for_llm_inference():
    # Define options
    parser = argparse.ArgumentParser(description="EEG-LLM Inference with Prompt Experimentation")
    
    # Model and data paths
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pretrained model"
    )
    parser.add_argument(
        "--eeg_dataset", 
        type=str,
        required=True,
        help="EEG dataset path (5-95Hz)"
    )
    parser.add_argument(
        "--image_dir", 
        type=str,
        required=True,
        help="ImageNet dataset path"
    )
    parser.add_argument(
        "--splits_path", 
        type=str,
        required=True,
        help="Path to data splits"
    )
    
    # Prompt experimentation options
    prompt_choices = ["basic", "detailed", "concise", "creative", 
                      "contextual", "question", "chain_of_thought", 
                      "instruction_focused", "all"]
    
    parser.add_argument(
        "--prompt_type", 
        type=str, 
        default="default",
        choices=prompt_choices,
        help="Type of prompt to use (or 'all' to run all)"
    )
    parser.add_argument(
        "--prompt_types", 
        type=str, 
        nargs="+", 
        default=None,
        help="Specific prompt types to run (overrides --prompt_type)"
    )
    
    # Experiment configuration
    parser.add_argument(
        "--results_dir", 
        type=str, 
        default="./experiment_results",
        help="Directory to save experiment results"
    )
    parser.add_argument(
        "--dest", 
        type=str,
        default=None,
        help="Destination file path for results (CSV format). If not provided, will be generated automatically."
    )
    parser.add_argument(
        "-sn", "--split_num", 
        default=0, 
        type=int, 
        help="Split number to use"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,
        help="Batch size for inference"
    )
    
    # Subject selecting
    parser.add_argument(
        "-sub", "--subject",
        default=0,
        type=int,
        help="choose a subject from 1 to 6, default is 0 (all subjects)"
    )

    # Time options: select from 20 to 460 samples from EEG data
    parser.add_argument(
        "-tl", "--time_low", 
        default=20, 
        type=float, 
        help="lowest time value"
    )
    parser.add_argument(
        "-th", "--time_high", 
        default=460, 
        type=float, 
        help="highest time value"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=100,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--max_prefix_length", 
        type=int, 
        default=512,
        help="Maximum length for prefix tokens"
    )
    parser.add_argument(
        "--max_suffix_length", 
        type=int, 
        default=128,
        help="Maximum length for suffix tokens"
    )
    parser.add_argument(
        "--do_sample", 
        action="store_true",
        help="Use sampling instead of greedy decoding"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=1.0,
        help="Temperature for sampling"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=1.0,
        help="Top-p for nucleus sampling"
    )
    parser.add_argument(
        "--repetition_penalty", 
        type=float, 
        default=1.1,
        help="Repetition penalty"
    )
    
    # Hardware and performance
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda:0",
        help="Device to use (cuda:0, cuda:1, or cpu)"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=4,
        help="Number of workers for data loading"
    )
    
    # Debugging and logging
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    parser.add_argument(
        "--save_every", 
        type=int, 
        default=5,
        help="Save intermediate results every N batches"
    )
    
    # Optional: Model specific settings
    parser.add_argument(
        "--clip_model", 
        default="openai/clip-vit-base-patch32",
        help="CLIP model to use"
    )
    
    args = parser.parse_args()
    
    # Post-process arguments
    if args.prompt_types is not None:
        args.prompt_type = args.prompt_types
    
    # Set default destination if not provided
    if args.dest is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.dest = os.path.join(args.results_dir, f"inference_results_{timestamp}.csv")
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Set log level
    import logging
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    logging.basicConfig(level=numeric_level)
    
    # Validate required paths exist
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model path does not exist: {args.model_path}")
    
    if not os.path.exists(args.eeg_dataset):
        print(f"Warning: EEG dataset path does not exist: {args.eeg_dataset}")
    
    if not os.path.exists(args.image_dir):
        print(f"Warning: Image directory path does not exist: {args.image_dir}")
    
    if not os.path.exists(args.splits_path):
        raise ValueError(f"Splits path does not exist: {args.splits_path}")
    
    return args


def get_args_for_prompt_experiments():
    """Arguments specifically for running batch prompt experiments"""
    parser = argparse.ArgumentParser(description="Batch Prompt Experiments")
    
    # Experiment configuration
    parser.add_argument(
        "--config_file",
        type=str,
        default="experiment_config.json",
        help="Path to experiment configuration file"
    )
    parser.add_argument(
        "--prompt_types",
        type=str,
        nargs="+",
        default=["all"],
        help="Prompt types to run (default: all)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./prompt_experiment_results",
        help="Directory to save experiment results"
    )
    parser.add_argument(
        "--max_parallel",
        type=int,
        default=1,
        help="Maximum number of parallel experiments to run"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip experiments that already have results"
    )
    
    # Common inference parameters (for batch experiments)
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pretrained model"
    )
    parser.add_argument(
        "--eeg_dataset",
        type=str,
        required=True,
        help="EEG dataset path"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Image dataset path"
    )
    parser.add_argument(
        "--splits_path",
        type=str,
        required=True,
        help="Path to data splits"
    )
    parser.add_argument(
        "--split_num",
        type=int,
        default=0,
        help="Split number to use"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    # Test each argument parser
    print("Testing argument parsers...")
    
    try:
        print("\n1. Encoder Training Args:")
        encoder_args = get_args_for_encoder_training()
        print(f"   Output dir: {encoder_args.output}")
        
        print("\n2. LLM Finetuning Args:")
        finetune_args = get_args_for_llm_finetuning()
        print(f"   Output dir: {finetune_args.output}")
        print(f"   EEG encoder path: {finetune_args.eeg_encoder_path}")
        
        print("\n3. LLM Inference Args:")
        # Create dummy args for testing
        import sys
        sys.argv = [sys.argv[0], "--model_path", "test", "--eeg_dataset", "test", 
                   "--image_dir", "test", "--splits_path", "test"]
        inference_args = get_args_for_llm_inference()
        print(f"   Model path: {inference_args.model_path}")
        print(f"   Prompt type: {inference_args.prompt_type}")
        print(f"   Results dir: {inference_args.results_dir}")
        
    except SystemExit:
        print("Argument test completed (some arguments were missing, as expected)")