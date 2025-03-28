#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tuning script for LLM human preference prediction.
Uses Modal and Axolotl to fine-tune the base model on the training set.
"""

import os
import json
import pandas as pd
import modal
from datetime import datetime

# Define Modal app
app = modal.App("llm-preference-finetuning")

# Define a proper function for Modal image setup
def setup_modal_image():
    print("Building Modal image...")

# Create Modal image with necessary dependencies
image = (modal.Image.debian_slim()
    .pip_install(
        "transformers>=4.34.0",
        "torch>=2.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "accelerate>=0.23.0",
        "bitsandbytes>=0.41.0",
        "peft>=0.5.0",
        "trl>=0.7.2",
        "datasets>=2.14.0",
        "tqdm>=4.66.0",
        "wandb>=0.15.0",
        "huggingface-hub>=0.17.0",
        "axolotl>=0.3.0",
        "sentencepiece>=0.1.99",  # Required for tokenization
        "protobuf>=4.25.1"  # Required for model loading
    )
)

# Define the model to use
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # Using Qwen2.5 0.5B Instruct model

# Create Modal volumes for storing model weights and logs
model_volume = modal.Volume.from_name("finetuning-models-vol", create_if_missing=True)
log_volume = modal.Volume.from_name("finetuning-logs-vol", create_if_missing=True)

# Define a function to check and prepare model files and logs using the volume
@app.function(
    image=image,
    volumes={"/models": model_volume, "/logs": log_volume}
)
def copy_model_and_logs():
    import os
    import glob
    import tarfile
    import shutil
    import time
    
    result = {
        "model_files": [],
        "log_files": [],
        "model_archive_path": None,
        "log_archive_path": None
    }
    
    # Check if model files exist
    if os.path.exists("/models"):
        model_files = glob.glob("/models/**/*", recursive=True)
        result["model_files"] = model_files
        print(f"Found {len(model_files)} model files in /models directory")
        
        # Create a tarball of the model for easier download
        if not os.path.exists("/models/qwen-preference-ft.tar.gz"):
            model_dir = "/models/qwen-preference-ft"
            if os.path.exists(model_dir):
                print(f"Creating model archive from {model_dir}")
                with tarfile.open("/models/qwen-preference-ft.tar.gz", "w:gz") as tar:
                    tar.add(model_dir, arcname=os.path.basename(model_dir))
                result["model_archive_path"] = "/models/qwen-preference-ft.tar.gz"
                print(f"Model archive created at {result['model_archive_path']}")
            else:
                print(f"Warning: Model directory {model_dir} not found")
        else:
            result["model_archive_path"] = "/models/qwen-preference-ft.tar.gz"
            print(f"Using existing model archive at {result['model_archive_path']}")
    else:
        print("Warning: /models directory not found")
    
    # Check if log files exist and create summary file
    if os.path.exists("/logs"):
        # Create a summary of all files in the container for debugging
        with open("/logs/container_files_summary.txt", "w") as f:
            f.write("=== Files in /models ===\n")
            if os.path.exists("/models"):
                for item in os.listdir("/models"):
                    item_path = os.path.join("/models", item)
                    f.write(f"{item_path} {'[DIR]' if os.path.isdir(item_path) else '[FILE]'}\n")
                    if os.path.isdir(item_path):
                        for subitem in os.listdir(item_path):
                            f.write(f"  {os.path.join(item_path, subitem)}\n")
            
            f.write("\n=== Files in /logs ===\n")
            for item in os.listdir("/logs"):
                item_path = os.path.join("/logs", item)
                f.write(f"{item_path} {'[DIR]' if os.path.isdir(item_path) else '[FILE]'}\n")
        
        # Copy any missing logs from training output
        if os.path.exists("/models/trainer_state.json") and not os.path.exists("/logs/trainer_state.json"):
            shutil.copy("/models/trainer_state.json", "/logs/trainer_state.json")
            print("Copied trainer_state.json to logs directory")
        
        # Create an extraction of the training metrics from any log files that exist
        metrics_extracted = False
        with open("/logs/training_metrics.txt", "w") as metrics_file:
            # Look for metrics in various log files
            for log_path in ['/logs/training_log.txt', '/models/trainer_log.txt']:
                if os.path.exists(log_path):
                    print(f"Extracting metrics from {log_path}")
                    with open(log_path, 'r') as source:
                        for line in source:
                            if 'loss' in line or 'metric' in line or 'eval' in line:
                                metrics_file.write(line)
                                metrics_extracted = True
        
        if not metrics_extracted:
            print("Warning: Could not extract training metrics from any log file")
        
        # Find all log files
        log_files = glob.glob("/logs/**/*", recursive=True)
        result["log_files"] = log_files
        print(f"Found {len(log_files)} log files in /logs directory")
        for log_file in log_files:
            if os.path.isfile(log_file):
                print(f"  {log_file} ({os.path.getsize(log_file)} bytes)")
        
        # Create a more comprehensive log archive with proper directory structure
        if not os.path.exists("/logs/training_logs.tar.gz"):
            # Add a timestamp file to the logs
            timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
            with open("/logs/archive_creation_time.txt", "w") as f:
                f.write(f"Archive created at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total log files: {len(log_files)}\n")
            
            # Create a manifest of all log files
            with open("/logs/log_manifest.txt", "w") as f:
                f.write(f"Log files manifest created at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                for log_file in log_files:
                    if os.path.isfile(log_file):
                        file_size = os.path.getsize(log_file)
                        f.write(f"{log_file} ({file_size} bytes)\n")
            
            # Create a consolidated metrics file from individual metrics files
            metrics_files = glob.glob("/logs/metrics/*.json")
            if metrics_files:
                print(f"Consolidating {len(metrics_files)} metrics files")
                all_metrics = []
                for metrics_file in sorted(metrics_files):
                    try:
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                            # Extract step number from filename
                            step = int(os.path.basename(metrics_file).replace('step_', '').replace('.json', ''))
                            all_metrics.append({"step": step, **metrics})
                    except Exception as e:
                        print(f"Error processing metrics file {metrics_file}: {str(e)}")
                
                # Save consolidated metrics in a format suitable for training_analysis.py
                with open("/logs/consolidated_metrics.json", "w") as f:
                    json.dump(all_metrics, f, indent=2)
                
                # Also save in the format expected by training_analysis.py
                with open("/logs/training_metrics.txt", "w") as f:
                    for metric in all_metrics:
                        if 'loss' in metric:
                            f.write('{"loss": ' + str(metric.get('loss', 0)) + ', "grad_norm": ' + str(metric.get('grad_norm', 'nan')) + ', "learning_rate": ' + str(metric.get('learning_rate', 0)) + ', "mean_token_accuracy": ' + str(metric.get('mean_token_accuracy', 0)) + ', "epoch": ' + str(metric.get('epoch', 0)) + '}\n')
            
            print(f"Creating log archive from {len(log_files)} files")
            with tarfile.open("/logs/training_logs.tar.gz", "w:gz") as tar:
                # Add files with proper directory structure
                for log_file in log_files:
                    if os.path.isfile(log_file):
                        # Preserve relative path within the logs directory
                        rel_path = os.path.relpath(log_file, "/logs")
                        print(f"Adding {log_file} as {rel_path} to archive")
                        tar.add(log_file, arcname=rel_path)
            
            result["log_archive_path"] = "/logs/training_logs.tar.gz"
            print(f"Log archive created at {result['log_archive_path']}")
            
            # Verify archive contents
            with tarfile.open("/logs/training_logs.tar.gz", "r:gz") as tar:
                archive_contents = tar.getnames()
                print(f"Archive contains {len(archive_contents)} files")
                for content in archive_contents[:10]:  # Show first 10 files
                    print(f"  - {content}")
                if len(archive_contents) > 10:
                    print(f"  ... and {len(archive_contents) - 10} more files")
        else:
            result["log_archive_path"] = "/logs/training_logs.tar.gz"
            print(f"Using existing log archive at {result['log_archive_path']}")
    else:
        print("Warning: /logs directory not found")
    
    return result

# Define function to download model archive from volume
@app.function(
    image=image,
    volumes={"/models": model_volume}
)
def download_model_archive():
    import os
    if os.path.exists("/models/qwen-preference-ft.tar.gz"):
        with open("/models/qwen-preference-ft.tar.gz", "rb") as f:
            return f.read()
    return None

# Define function to download logs archive from volume
@app.function(
    image=image,
    volumes={"/logs": log_volume}
)
def download_logs_archive():
    import os
    import tarfile
    import io
    import glob
    import json
    import time
    
    try:
        # Check if we need to create or update the archive
        log_files = glob.glob("/logs/**/*", recursive=True)
        log_files = [f for f in log_files if os.path.isfile(f) and not f.endswith('training_logs.tar.gz')]
        
        # If we have log files but no archive, or if the archive is older than the newest log file
        archive_path = "/logs/training_logs.tar.gz"
        need_new_archive = False
        
        if not os.path.exists(archive_path):
            need_new_archive = True
        else:
            archive_mtime = os.path.getmtime(archive_path)
            for log_file in log_files:
                if os.path.getmtime(log_file) > archive_mtime:
                    need_new_archive = True
                    break
        
        if need_new_archive and log_files:
            print(f"Creating/updating log archive with {len(log_files)} files")
            
            # Create a manifest of all log files
            with open("/logs/log_manifest.txt", "w") as f:
                f.write(f"Log files manifest created at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                for log_file in log_files:
                    file_size = os.path.getsize(log_file)
                    f.write(f"{log_file} ({file_size} bytes)\n")
            
            # Process metrics files if they exist
            metrics_files = glob.glob("/logs/metrics/*.json")
            if metrics_files:
                print(f"Consolidating {len(metrics_files)} metrics files")
                all_metrics = []
                for metrics_file in sorted(metrics_files):
                    try:
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                            # Extract step number from filename
                            step = int(os.path.basename(metrics_file).replace('step_', '').replace('.json', ''))
                            all_metrics.append({"step": step, **metrics})
                    except Exception as e:
                        print(f"Error processing metrics file {metrics_file}: {str(e)}")
                
                # Save consolidated metrics in a format suitable for training_analysis.py
                with open("/logs/consolidated_metrics.json", "w") as f:
                    json.dump(all_metrics, f, indent=2)
                
                # Also save in the format expected by training_analysis.py
                with open("/logs/training_metrics.txt", "w") as f:
                    for metric in all_metrics:
                        if 'loss' in metric:
                            f.write('{"loss": ' + str(metric.get('loss', 0)) + ', "grad_norm": ' + str(metric.get('grad_norm', 'nan')) + ', "learning_rate": ' + str(metric.get('learning_rate', 0)) + ', "mean_token_accuracy": ' + str(metric.get('mean_token_accuracy', 0)) + ', "epoch": ' + str(metric.get('epoch', 0)) + '}\n')
            
            # Create the archive
            with tarfile.open(archive_path, "w:gz") as tar:
                for log_file in log_files:
                    # Preserve relative path within the logs directory
                    rel_path = os.path.relpath(log_file, "/logs")
                    tar.add(log_file, arcname=rel_path)
            
            print(f"Log archive created at {archive_path}")
        
        # Return the archive contents
        if os.path.exists(archive_path):
            print(f"Reading log archive from {archive_path}")
            with open(archive_path, "rb") as f:
                return f.read()
        elif log_files:
            # If we have log files but couldn't create an archive, create one in memory
            print("Creating in-memory log archive")
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
                for log_file in log_files:
                    rel_path = os.path.relpath(log_file, "/logs")
                    tar.add(log_file, arcname=rel_path)
            buffer.seek(0)
            return buffer.read()
        elif os.path.exists("/logs/training_log.txt"):
            with open("/logs/training_log.txt", "rb") as f:
                return f.read()
        
        print("No log files found to download")
        return None
    except Exception as e:
        print(f"Error preparing logs for download: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

@app.function(
    image=image,
    gpu="H100",  # Using H100 for faster training
    timeout=7200,  # 2 hours
    volumes={
        "/data": modal.Volume.from_name("finetuning-data-vol"),
        "/models": model_volume,
        "/logs": log_volume
    }
)
def run_finetuning(train_data_json, config_yaml):
    """Run fine-tuning using a direct approach with transformers and PEFT."""
    import os
    import sys
    import json
    import torch
    import time
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import Dataset
    
    # Configure longer timeouts for Hugging Face downloads
    import huggingface_hub
    huggingface_hub.constants.HF_TIMEOUT = 120.0  # Increase timeout to 2 minutes
    
    print("\n=== Starting simplified fine-tuning process ===")
    
    # Check volumes
    print("\nChecking volumes:")
    if os.path.exists("/data"):
        print("Volume /data exists")
        os.makedirs("/data", exist_ok=True)
    else:
        print("Volume /data NOT found, creating directory")
        os.makedirs("/data", exist_ok=True)
    
    if os.path.exists("/models"):
        print("Volume /models exists")
        os.makedirs("/models", exist_ok=True)
    else:
        print("Volume /models NOT found, creating directory")
        os.makedirs("/models", exist_ok=True)
    
    # Parse the training data directly from the JSON string passed to the function
    print("\nLoading training data from passed JSON string")
    train_data = [json.loads(line) for line in train_data_json.strip().split('\n')]
    
    print(f"Loaded {len(train_data)} training examples")
    
    # Ensure all data is properly typed to avoid PyArrow errors
    cleaned_data = []
    for item in train_data:
        # Convert all values to strings to avoid type issues
        cleaned_item = {}
        for key, value in item.items():
            if isinstance(value, (int, float)):
                cleaned_item[key] = str(value)
            else:
                cleaned_item[key] = value
        cleaned_data.append(cleaned_item)
    
    # Create a simple dataset
    train_dataset = Dataset.from_list(cleaned_data)
    print("Dataset created successfully")
    
    # Load model and tokenizer with retry logic
    print("\nLoading model and tokenizer")
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    max_retries = 3
    retry_delay = 10  # seconds
    
    # Function to load with retries
    def load_with_retries(load_func, description, max_attempts=max_retries, delay=retry_delay):
        for attempt in range(1, max_attempts + 1):
            try:
                print(f"Attempt {attempt}/{max_attempts} to load {description}...")
                return load_func()
            except Exception as e:
                print(f"Attempt {attempt} failed: {str(e)}")
                if attempt < max_attempts:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 1.5  # Exponential backoff
                else:
                    print(f"All {max_attempts} attempts failed.")
                    raise
    
    try:
        # Load tokenizer with retries
        tokenizer = load_with_retries(
            lambda: AutoTokenizer.from_pretrained(model_id, trust_remote_code=True),
            "tokenizer"
        )
        print("Tokenizer loaded successfully")
        
        # Load model with retries
        model = load_with_retries(
            lambda: AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto"
            ),
            "model"
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        # Create a minimal log file even if model loading fails
        os.makedirs("/logs", exist_ok=True)
        with open("/logs/error_log.txt", "w") as f:
            f.write(f"Error loading model: {str(e)}\n")
        with open("/logs/training_log.txt", "w") as f:
            f.write(f"Training failed: Error loading model at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Create test files to verify volume access
        try:
            with open("/logs/test.txt", "w") as f:
                f.write("This is a test file to verify volume access\n")
            print("Created test file in /logs")
        except Exception as log_err:
            print(f"Error creating test file in logs: {str(log_err)}")
            
        raise
    
    # Configure LoRA
    print("\nConfiguring LoRA")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    # Prepare model for training
    model = get_peft_model(model, lora_config)
    print("PEFT model prepared successfully")
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="/models",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        warmup_steps=10,
        max_steps=1000,
        learning_rate=5e-5,
        fp16=True,
        logging_steps=1,
        save_steps=50,
        save_total_limit=3,
        report_to="none"
    )
    
    # Create trainer
    print("\nSetting up trainer")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args
        # Removed all unsupported parameters
    )
    
    # Set up logging directory
    os.makedirs("/logs", exist_ok=True)
    log_file_path = "/logs/training_log.txt"
    
    # Create a log file to capture training output
    with open(log_file_path, "w") as log_file:
        log_file.write(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        log_file.write(f"Model: {model_id}\n")
        log_file.write(f"LoRA config: {lora_config}\n")
        log_file.write(f"Training args: {training_args}\n\n")
        
    # Also create separate log files for different types of logs
    os.makedirs("/logs/metrics", exist_ok=True)
    with open("/logs/config_summary.txt", "w") as f:
        f.write(f"=== Training Configuration ===\n")
        f.write(f"Model: {model_id}\n")
        f.write(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset size: {len(train_dataset)} examples\n")
        f.write(f"Training arguments:\n{vars(training_args)}\n")
        f.write(f"LoRA configuration:\n{vars(lora_config)}\n")
    
    # Custom callback to log metrics with enhanced formatting for analysis
    class LoggingCallback(TrainerCallback):
        def __init__(self):
            self.metrics_history = []
            self.start_time = time.time()
            
            # Create a header for the metrics file
            with open("/logs/training_metrics.txt", "w") as f:
                f.write("# Training metrics log - Created at: {}\n".format(
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                f.write("# Format: {'loss': value, 'grad_norm': value, 'learning_rate': value, 'mean_token_accuracy': value, 'epoch': value}\n")
            
            # Create a CSV version for easier analysis
            with open("/logs/training_metrics.csv", "w") as f:
                f.write("step,loss,grad_norm,learning_rate,mean_token_accuracy,epoch,elapsed_time\n")
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                # Write to main log file
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"{logs}\n")
                
                # Get step number
                step = state.global_step if state and hasattr(state, 'global_step') else 0
                
                # Calculate elapsed time
                elapsed_time = time.time() - self.start_time
                
                # Add elapsed time to logs
                logs_with_time = {**logs, "elapsed_time": elapsed_time}
                
                # Write to separate metrics file for easier extraction
                with open(f"/logs/metrics/step_{step:05d}.json", "w") as f:
                    json.dump(logs_with_time, f, indent=2)
                
                # Also create a consolidated metrics file in JSONL format
                with open("/logs/all_metrics.jsonl", "a") as f:
                    json.dump({"step": step, **logs_with_time}, f)
                    f.write("\n")
                
                # Save to history
                self.metrics_history.append({"step": step, **logs_with_time})
                
                # Write in the format expected by training_analysis.py
                with open("/logs/training_metrics.txt", "a") as f:
                    # Format the metrics in the expected format
                    if 'loss' in logs:
                        metrics_str = '{"loss": ' + str(logs.get('loss', 0))
                        metrics_str += ', "grad_norm": ' + str(logs.get('grad_norm', 'nan'))
                        metrics_str += ', "learning_rate": ' + str(logs.get('learning_rate', 0))
                        metrics_str += ', "mean_token_accuracy": ' + str(logs.get('train/accuracy', logs.get('eval/accuracy', 0)))
                        metrics_str += ', "epoch": ' + str(logs.get('epoch', 0))
                        metrics_str += '}\n'
                        f.write(metrics_str)
                
                # Write CSV version
                with open("/logs/training_metrics.csv", "a") as f:
                    loss = logs.get('loss', 0)
                    grad_norm = logs.get('grad_norm', 'nan')
                    lr = logs.get('learning_rate', 0)
                    accuracy = logs.get('train/accuracy', logs.get('eval/accuracy', 0))
                    epoch = logs.get('epoch', 0)
                    f.write(f"{step},{loss},{grad_norm},{lr},{accuracy},{epoch},{elapsed_time:.2f}\n")
        
        def on_train_end(self, args, state, control, **kwargs):
            # Save final consolidated metrics
            with open("/logs/consolidated_metrics.json", "w") as f:
                json.dump(self.metrics_history, f, indent=2)
            
            # Create a summary file
            with open("/logs/training_summary.txt", "w") as f:
                f.write("=== Training Summary ===\n")
                f.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total steps: {state.global_step if state else 0}\n")
                f.write(f"Total time: {time.time() - self.start_time:.2f} seconds\n")
                
                # Calculate improvements
                if self.metrics_history:
                    first_loss = next((m.get('loss', None) for m in self.metrics_history if 'loss' in m), None)
                    last_loss = next((m.get('loss', None) for m in reversed(self.metrics_history) if 'loss' in m), None)
                    
                    if first_loss is not None and last_loss is not None:
                        loss_improvement = ((first_loss - last_loss) / first_loss) * 100
                        f.write(f"Loss improvement: {loss_improvement:.2f}%\n")
                    
                    # Add other metrics as needed
                    f.write("\nFinal metrics:\n")
                    if self.metrics_history and 'loss' in self.metrics_history[-1]:
                        for k, v in self.metrics_history[-1].items():
                            f.write(f"  {k}: {v}\n")
    
    # Add callback to trainer
    trainer.add_callback(LoggingCallback())
    
    # Train model
    print("\nStarting training")
    try:
        trainer.train()
        print("Training completed successfully")
        
        # Save model
        print("\nSaving model")
        output_dir = "/models/qwen-preference-ft"
        trainer.save_model(output_dir)
        print(f"Model saved to {output_dir}")
        
        # Create a tarball of the model for easier download
        print("Creating model archive for download...")
        import tarfile
        import glob
        
        # Debug: Check if directories exist and what they contain
        print(f"Models directory exists: {os.path.exists('/models')}")
        if os.path.exists('/models'):
            print(f"Models directory contents: {os.listdir('/models')}")
        
        print(f"Logs directory exists: {os.path.exists('/logs')}")
        if os.path.exists('/logs'):
            print(f"Logs directory contents: {os.listdir('/logs')}")
        
        # List all mounted volumes
        print(f"Available mounted directories: {glob.glob('/*/'):}")
        
        archive_path = "/models/qwen-preference-ft.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(output_dir, arcname=os.path.basename(output_dir))
        
        print(f"Model archive created at {archive_path}")
        print(f"Archive exists: {os.path.exists(archive_path)}")
        
        # Create a tarball of the logs
        logs_archive_path = "/logs/training_logs.tar.gz"
        with tarfile.open(logs_archive_path, "w:gz") as tar:
            tar.add("/logs", arcname="logs")
        
        print(f"Logs archive created at {logs_archive_path}")
        print(f"Logs archive exists: {os.path.exists(logs_archive_path)}")
        
        # Also save a simple text file in both directories as a test
        with open("/models/test.txt", "w") as f:
            f.write("This is a test file")
        
        with open("/logs/test.txt", "w") as f:
            f.write("This is a test file")
        
        return {
            "status": "success", 
            "model_path": output_dir,
            "archive_path": archive_path,
            "logs_path": "/logs",
            "logs_archive_path": logs_archive_path
        }
    except Exception as e:
        print(f"Training error: {str(e)}")
        raise
    
    # List the output files
    print("\nFine-tuned model files:")
    for root, dirs, files in os.walk("/models"):
        for file in files:
            print(os.path.join(root, file))
    
    return {"status": "success", "model_path": "/models"}

def prepare_axolotl_config(output_dir="config"):
    """
    Prepare the Axolotl configuration file for fine-tuning.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(base_dir, output_dir)
    os.makedirs(config_dir, exist_ok=True)
    
    # Create Axolotl config
    config = f"""
base_model: Qwen/Qwen2.5-0.5B-Instruct
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer

# Model configuration
model_config:
  num_labels: 3  # Three-way classification
  trust_remote_code: true  # Required for Qwen model

load_in_8bit: false
load_in_4bit: false  # Not using quantization for Qwen model
strict: false
trust_remote_code: true  # Required for Qwen model

# Training hyperparameters optimized for Qwen2.5
sequence_len: 4096  # Qwen supports up to 32k but using 4k for efficiency
max_steps: 1000
eval_steps: 50
save_steps: 100
warmup_steps: 100
lora_dropout: 0.1
lr_scheduler: cosine
learning_rate: 5e-5
weight_decay: 0.01
optimizer: adamw_torch
gradient_checkpointing: true
flash_attention: true

datasets:
  - path: /data/train.jsonl
    type: json
    field_human: prompt
    field_assistant: response_a_text
    field_target: target
    field_response_b: response_b_text

dataset_prepared_path: null
val_set_size: 0.05
output_dir: /models

adapter: lora
lora_model_dir:
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

sequence_len: 2048
sample_packing: true
pad_to_sequence_len: true

# Training configuration for three-way classification
training:
  loss_fn: CrossEntropyLoss
  label_smoothing: 0.1
  class_weights: [1.0, 1.0, 1.0]  # Can be adjusted based on class distribution

wandb_project: llm-preference-prediction
wandb_run_name: qwen2.5-0.5b-preference-ft
wandb_watch: false
wandb_log_model: false

gradient_accumulation_steps: 2  # Reduced since H100 can handle larger batches
micro_batch_size: 16  # Increased for H100 GPU with 80GB VRAM
num_epochs: 3
optimizer: adamw_torch
lr_scheduler: cosine
learning_rate: 5.0e-5
train_on_inputs: false
group_by_length: false
bf16: true
fp16: false
tf32: false

gradient_checkpointing: true
early_stopping_patience: 3
resume_from_checkpoint: false
local_rank: 0
logging_steps: 10
xformers_attention: false
flash_attention: true

warmup_steps: 100
evals_per_epoch: 4
saves_per_epoch: 1
debug: false
deepspeed: null
weight_decay: 0.01
fsdp: null
special_tokens:
  bos_token: "<s>"
  eos_token: "</s>"
  unk_token: "<unk>"
"""
    
    config_path = os.path.join(config_dir, "axolotl_config.yml")
    with open(config_path, "w") as f:
        f.write(config)
    
    print(f"Axolotl config saved to {config_path}")
    return config_path

def prepare_training_data():
    """
    Prepare the training data for fine-tuning.
    Formats the data according to the prompt template.
    """
    # Load training data
    train_df = pd.read_csv("data/train.csv")
    print(f"Loaded {len(train_df)} training examples")
    
    # No need for target mapping since we're using numerical outputs directly
    
    # Prepare prompt template
    prompt_template = """
Human query: {prompt}

Assistant A's response:
{response_a}

Assistant B's response:
{response_b}

Which response would a human prefer? Output EXACTLY one number:
0 = Assistant A is better
1 = Assistant B is better
2 = They are equally good

Answer:
"""
    
    # Format the training data
    formatted_data = []
    for _, row in train_df.iterrows():
        # Format the prompt
        prompt = prompt_template.format(
            prompt=row["prompt"],
            response_a=row["response_a_text"],
            response_b=row["response_b_text"]
        )
        
        # Create the example (convert all numeric values to strings)
        example = {
            "prompt_id": str(row["prompt_id"]),
            "prompt": prompt,
            "completion": str(row["target"]),  # Convert to string for model training
            "response_a_text": row["response_a_text"],
            "response_b_text": row["response_b_text"],
            "target": str(row["target"])  # Convert to string to avoid type issues
        }
        
        formatted_data.append(example)
    
    # Save the formatted training data with absolute path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(base_dir, "data/formatted"), exist_ok=True)
    train_path = os.path.join(base_dir, "data/formatted/train.jsonl")
    
    with open(train_path, "w") as f:
        for example in formatted_data:
            f.write(json.dumps(example) + "\n")
    
    print(f"Formatted training data saved to {train_path}")
    return train_path

def run_finetuning_process():
    """Run the fine-tuning process."""
    import os
    import requests
    import tarfile
    import time
    
    print("Starting fine-tuning process...")
    
    # Prepare training data and config
    train_data_path = prepare_training_data()  # Creates data/formatted/train.jsonl
    config_path = prepare_axolotl_config(output_dir="config")  # Creates config/axolotl_config.yml
    
    # Create local directories for model and logs
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_model_dir = os.path.join(base_dir, "models")
    local_logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(local_model_dir, exist_ok=True)
    os.makedirs(local_logs_dir, exist_ok=True)
    
    # Read the contents of the files to pass directly to the Modal function
    with open(train_data_path, "r") as f:
        train_data_content = f.read()
    
    with open(config_path, "r") as f:
        config_content = f.read()
    
    # Run fine-tuning using Modal app context
    print("Running fine-tuning on Modal...")
    result = {"status": "failed", "error": "Unknown error"}
    
    try:
        with app.run() as modal_app:
            try:
                # Pass the file contents directly instead of paths
                result = run_finetuning.remote(train_data_content, config_content)
            except KeyboardInterrupt:
                print("Fine-tuning interrupted by user")
                result = {"status": "failed", "error": "Interrupted by user"}
                return False
            
            # Download the model if training was successful
            if result['status'] == 'success':
                print(f"Fine-tuning completed successfully. Attempting to download model...")
                try:
                    print("Downloading model and logs via volumes...")
                    
                    # Create local directories for model and logs
                    import shutil
                    import glob
                    import tarfile
                    
                    # 1. Copy model files from the Modal volume to local directory
                    print("Copying model files from volume...")
                    
                    # Ensure the local model directory exists
                    os.makedirs(local_model_dir, exist_ok=True)
                    
                    # Use the globally defined functions
                    # Execute the copy function to prepare archives
                    copy_result = copy_model_and_logs.remote()
                    print(f"Found {len(copy_result['model_files'])} model files and {len(copy_result['log_files'])} log files")
                    
                    # Download model archive if it exists
                    if copy_result["model_archive_path"]:
                        print("Downloading model archive...")
                        model_data = download_model_archive.remote()
                        if model_data:
                            # Save the archive locally
                            local_model_path = os.path.join(local_model_dir, "qwen-preference-ft.tar.gz")
                            with open(local_model_path, "wb") as f:
                                f.write(model_data)
                            print(f"Model archive downloaded to {local_model_path}")
                            
                            # Extract the archive
                            with tarfile.open(local_model_path, "r:gz") as tar:
                                tar.extractall(path=local_model_dir)
                            print(f"Model extracted to {local_model_dir}")
                        else:
                            print("Failed to download model archive")
                    
                    # Download logs archive if it exists
                    if copy_result["log_archive_path"]:
                        print("\nDownloading log archive...")
                        log_data = download_logs_archive.remote()
                        if log_data:
                            # Ensure the logs directory exists
                            os.makedirs(local_logs_dir, exist_ok=True)
                            
                            # Save the archive locally
                            local_log_path = os.path.join(local_logs_dir, "training_logs.tar.gz")
                            with open(local_log_path, "wb") as f:
                                f.write(log_data)
                            print(f"Log archive downloaded to {local_log_path}")
                            
                            # Extract the archive
                            with tarfile.open(local_log_path, "r:gz") as tar:
                                members = tar.getnames()
                                print(f"Log archive contains {len(members)} files")
                                if members:
                                    print(f"Sample files: {members[:5]}")
                                    if len(members) > 5:
                                        print(f"... and {len(members) - 5} more files")
                                tar.extractall(path=local_logs_dir)
                            print(f"Logs extracted to {local_logs_dir}")
                            
                            # List extracted files and create a directory structure for analysis
                            extracted_logs = []
                            for root, dirs, files in os.walk(local_logs_dir):
                                for file in files:
                                    if file != "training_logs.tar.gz":
                                        log_path = os.path.join(root, file)
                                        extracted_logs.append(os.path.relpath(log_path, local_logs_dir))
                            
                            print(f"Extracted {len(extracted_logs)} log files")
                            
                            # Check for training metrics file for analysis
                            training_metrics_path = os.path.join(local_logs_dir, "training_metrics.txt")
                            if os.path.exists(training_metrics_path):
                                print(f"Found training metrics file at {training_metrics_path}")
                                print("You can now run training analysis with:")
                                print(f"python -m src.training_analysis --log_file {training_metrics_path} --output_dir results/training_analysis")
                            else:
                                # Look for alternative metrics files
                                metrics_files = [f for f in extracted_logs if "metric" in f.lower() or "training" in f.lower()]
                                if metrics_files:
                                    print(f"Found potential metrics files: {metrics_files}")
                                    print("You can analyze these with the training_analysis.py script")
                                else:
                                    print("No training metrics files found for analysis")
                        else:
                            print("Failed to download log archive")
                    
                except Exception as e:
                    print(f"Error downloading model or logs: {str(e)}")
    except KeyboardInterrupt:
        print("Fine-tuning interrupted by user")
        result = {"status": "failed", "error": "Interrupted by user"}
        return False
    except Exception as e:
        print(f"Fine-tuning failed: {str(e)}")
        result = {"status": "failed", "error": str(e)}
        return False
            
    print(f"Fine-tuning completed with status: {result['status']}")
    
    # Save fine-tuning metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata = {
        "base_model": BASE_MODEL_ID,
        "timestamp": timestamp,
        "train_data_path": train_data_path,
        "config_path": config_path,
        "result": result
    }
    
    os.makedirs("results", exist_ok=True)
    with open(f"results/finetuning_metadata_{timestamp}.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Fine-tuning metadata saved to results/finetuning_metadata_{timestamp}.json")
    
    # Check if we have training logs and prepare them for analysis
    log_path = os.path.join(local_logs_dir, "training_log.txt")
    if os.path.exists(log_path):
        print(f"Training logs found at {log_path}")
        print("You can now run the training analysis script:")
        print(f"python -m src.training_analysis --log_file={log_path} --output_dir=results/training_analysis")
    else:
        print(f"Training logs not found at expected path: {log_path}")
        
        # Try to find logs elsewhere
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file == "training_log.txt":
                    alt_log_path = os.path.join(root, file)
                    print(f"Found training logs at alternative location: {alt_log_path}")
                    print("You can run the training analysis script with:")
                    print(f"python -m src.training_analysis --log_file={alt_log_path} --output_dir=results/training_analysis")
                    break
    return True

if __name__ == "__main__":
    # This will be executed when running the script directly
    run_finetuning_process()
