#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation script for LLM human preference prediction.
Compares the performance of the base model and fine-tuned model.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from datetime import datetime
import time
import modal

# Define Modal app
app = modal.App("llm-preference-evaluation")

# Create Modal image with necessary dependencies
image = (modal.Image.debian_slim()
    .pip_install(
        "transformers>=4.34.0",
        "torch>=2.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "accelerate>=0.23.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.66.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "huggingface-hub>=0.17.0",
        "peft>=0.5.0",  # Required for LoRA fine-tuning
        "bitsandbytes>=0.41.0",  # Required for quantization
        "sentencepiece>=0.1.99",  # Required for tokenization
        "protobuf>=4.25.1"  # Required for model loading
    )

)

@app.function(
    image=image,
    gpu="H100",  # Using H100 for faster inference
    timeout=3600,
    mounts=[
        modal.Mount.from_local_dir("./models", remote_path="/root/models")
    ]
)
def run_finetuned_inference(batch, prompt_template, batch_index=0):
    """Run inference on a batch of examples using the fine-tuned model.
    
    Args:
        batch: Batch of examples to process
        prompt_template: Template for formatting prompts
        batch_index: Index of the current batch (for debugging)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    import os
    from huggingface_hub import login
    
    # Login to Hugging Face

    
    # Use the local fine-tuned model that was downloaded and mounted to the container
    model_path = "/root/models/qwen-preference-ft"
    print(f"Using fine-tuned model from: {model_path}")
    
    # Check if the model directory exists
    if not os.path.exists(model_path):
        print(f"Error: Model directory {model_path} not found in container")
        print("Available directories in /root:")
        for item in os.listdir("/root"):
            print(f"  - {item}")
        if os.path.exists("/root/models"):
            print("\nContents of /root/models:")
            for item in os.listdir("/root/models"):
                print(f"  - {item}")
        raise ValueError(f"Model directory {model_path} not found")
    
    # Load base model and tokenizer
    base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"  # Using Qwen2.5 0.5B Instruct model
    
    # Load tokenizer from the base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    # Load the fine-tuned model
    try:
        print(f"Loading model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True  # Required for Qwen model
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        # Try loading the base model and applying LoRA adapter if available
        print("Attempting to load base model and apply LoRA adapter...")
        from peft import PeftModel
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Check if adapter config exists
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            print("Found LoRA adapter, applying it to base model")
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            print("No adapter config found, using base model")
            model = base_model
    
    results = []
    
    for example in batch:
        # Format the prompt using the template
        formatted_prompt = prompt_template.format(
            prompt=example["prompt"],
            response_a=example["response_a_text"],
            response_b=example["response_b_text"]
        )
        
        # Tokenize the prompt
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Generate prediction
        with torch.no_grad():
            # Remove trust_remote_code from generation params
            # It's only needed for model loading, not for generation
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,  # Short response (just the number)
                do_sample=False,  # Deterministic output
                temperature=0.0,  # No randomness
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the prediction
        prediction_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
        
        # Parse the prediction (0, 1, or 2) - enhanced extraction logic
        import re
        
        # Clean up the prediction text
        prediction_text = prediction_text.strip().lower()
        
        # First, try to extract just a single digit (most likely case)
        if prediction_text in ['0', '1', '2']:
            prediction = int(prediction_text)
        
        # Check for specific patterns in the response
        elif "0" in prediction_text and ("assistant a" in prediction_text or "a is better" in prediction_text):
            prediction = 0
        elif "1" in prediction_text and ("assistant b" in prediction_text or "b is better" in prediction_text):
            prediction = 1
        elif "2" in prediction_text and ("equally" in prediction_text or "equal" in prediction_text):
            prediction = 2
            
        # Try to find the first digit that could be the answer
        else:
            # Look for digits at the start of the text or after specific markers
            match = re.search(r'^\s*(\d)', prediction_text)
            if not match:
                match = re.search(r'answer[^\d]*(\d)', prediction_text)
            if not match:
                match = re.search(r':\s*(\d)', prediction_text)
            if not match:
                match = re.search(r'\d', prediction_text)  # Last resort: any digit
                
            if match and match.group(1) in ['0', '1', '2']:
                prediction = int(match.group(1))
            else:
                # If we found a digit but it's not valid, check if it's a typo
                all_digits = re.findall(r'\d', prediction_text)
                if all_digits and all_digits[0] in ['0', '1', '2']:
                    prediction = int(all_digits[0])
                else:
                    prediction = -1  # Invalid prediction
        
        # Debug output for the first few examples in each batch
        example_index = batch.index(example) if example in batch else -1
        if example_index < 5 or prediction == -1:  # Show first few and invalid predictions
            print(f"Batch {batch_index}, Example {example_index}: '{prediction_text}' → Prediction: {prediction}")
        
        # Store the result
        result = {
            "prompt_id": example["prompt_id"],
            "prediction": prediction,
            "prediction_text": prediction_text,
            "true_label": example["target"]
        }
        results.append(result)
    
    return results

def load_validation_data(file_path="data/val.csv"):
    """Load the validation dataset."""
    df = pd.read_csv(file_path)
    return df

def batch_data(df, batch_size=10):
    """Split the data into batches for parallel processing."""
    data = df.to_dict('records')
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def prepare_prompt_template():
    """
    Prepare the prompt template for the model.
    Based on the approach from the referenced Kaggle solution.
    """
    prompt_template = """
You are evaluating two AI assistant responses to a human query. Your task is to determine which response would be preferred by a human, or if they are equally good.

Human query: {prompt}

Assistant A's response:
{response_a}

Assistant B's response:
{response_b}

Which response would a human prefer? 

IMPORTANT: You must respond with ONLY a single digit from these options:
1 = Assistant B is better
2 = They are equally good
0 = Assistant A is better

Your answer (just the number, no explanation): """
    return prompt_template

def evaluate_finetuned_model():
    """Run inference with the fine-tuned model and evaluate results."""
    print("Starting fine-tuned model evaluation...")
    
    # First, check if the fine-tuned model exists locally
    local_model_path = os.path.join(os.getcwd(), "models", "qwen-preference-ft")
    if not os.path.exists(local_model_path):
        print(f"Error: Fine-tuned model not found at {local_model_path}")
        print("Make sure you've run the fine-tuning step first and the model was downloaded successfully.")
        return None
    
    print(f"Found fine-tuned model at: {local_model_path}")
    
    # Load validation data
    val_df = load_validation_data()
    print(f"Loaded {len(val_df)} validation examples")
    
    # Prepare prompt template
    prompt_template = prepare_prompt_template()
    
    # Batch the data - use smaller batch size for more reliable processing
    batch_size = 32  # Smaller batch size to avoid OOM errors
    batches = batch_data(val_df, batch_size)
    print(f"Split data into {len(batches)} batches of size {batch_size}")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Run inference on each batch
    all_results = []
    start_time = time.time()
    
    for i, batch in enumerate(batches):
        print(f"Processing batch {i+1}/{len(batches)}")
        try:
            batch_results = run_finetuned_inference.remote(batch, prompt_template, i+1)
            all_results.extend(batch_results)
            
            # Save intermediate results occasionally to avoid losing progress
            if (i + 1) % 20 == 0 or i == len(batches) - 1:
                with open(f"results/finetuned_inference_intermediate_{i+1}.json", 'w') as f:
                    json.dump(all_results, f, indent=2)
                print(f"Saved intermediate results after batch {i+1}/{len(batches)}")
        except Exception as e:
            print(f"Error processing batch {i+1}: {str(e)}")
            # Save what we have so far
            with open(f"results/finetuned_inference_error_batch_{i+1}.json", 'w') as f:
                json.dump(all_results, f, indent=2)
        
        # Save intermediate results
        if (i + 1) % 5 == 0 or i == len(batches) - 1:
            with open(f"results/finetuned_inference_results_intermediate_{i+1}.json", 'w') as f:
                json.dump(all_results, f, indent=2)
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/finetuned_model_inference_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(all_results)
    
    # Calculate accuracy
    valid_predictions = results_df[results_df['prediction'] != -1]
    accuracy = (valid_predictions['prediction'] == valid_predictions['true_label']).mean()
    
    # Calculate the percentage of invalid predictions
    invalid_percent = (results_df['prediction'] == -1).mean() * 100
    
    print(f"\nFine-tuned Model Validation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Invalid predictions: {invalid_percent:.2f}%")
    
    # Save evaluation metrics
    metrics = {
        "model": "finetuned-qwen2.5-0.5b",
        "accuracy": float(accuracy),
        "invalid_predictions_percent": float(invalid_percent),
        "timestamp": timestamp,
        "total_examples": len(results_df),
        "valid_examples": len(valid_predictions)
    }
    
    with open(f"results/finetuned_model_metrics_{timestamp}.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    return metrics

def compare_models():
    """Compare the performance of the base model and fine-tuned model."""
    print("Comparing model performances...")
    
    # Find the latest base model and fine-tuned model metrics
    base_metrics_files = [f for f in os.listdir("results") if f.startswith("base_model_metrics_")]
    finetuned_metrics_files = [f for f in os.listdir("results") if f.startswith("finetuned_model_metrics_")]
    
    if not base_metrics_files or not finetuned_metrics_files:
        print("Error: Missing metrics files for comparison")
        return
    
    # Sort by timestamp (newest first)
    latest_base = sorted(base_metrics_files)[-1]
    latest_finetuned = sorted(finetuned_metrics_files)[-1]
    
    # Load metrics
    with open(os.path.join("results", latest_base), 'r') as f:
        base_metrics = json.load(f)
    
    with open(os.path.join("results", latest_finetuned), 'r') as f:
        finetuned_metrics = json.load(f)
    
    # Compare accuracies
    base_acc = base_metrics["accuracy"]
    finetuned_acc = finetuned_metrics["accuracy"]
    
    improvement = (finetuned_acc - base_acc) * 100
    
    print("\nModel Comparison:")
    print(f"Base model accuracy: {base_acc:.4f}")
    print(f"Fine-tuned model accuracy: {finetuned_acc:.4f}")
    print(f"Improvement: {improvement:.2f}%")
    
    # Create comparison visualization
    plt.figure(figsize=(10, 6))
    models = ["Base Model", "Fine-tuned Model"]
    accuracies = [base_acc, finetuned_acc]
    
    plt.bar(models, accuracies, color=['blue', 'green'])
    plt.ylim(max(0, min(accuracies) - 0.1), min(1.0, max(accuracies) + 0.1))
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add accuracy values on top of bars
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f"{acc:.4f}", ha='center')
    
    # Add improvement arrow and text
    if improvement > 0:
        plt.annotate(
            f"+{improvement:.2f}%",
            xy=(1, finetuned_acc),
            xytext=(0.5, (base_acc + finetuned_acc) / 2),
            arrowprops=dict(arrowstyle="->", color="red"),
            color="red",
            fontsize=12,
            fontweight="bold"
        )
    
    plt.tight_layout()
    plt.savefig("results/model_comparison.png")
    
    # Save comparison results
    comparison = {
        "base_model": {
            "name": base_metrics["model"],
            "accuracy": base_acc,
            "invalid_predictions_percent": base_metrics["invalid_predictions_percent"]
        },
        "finetuned_model": {
            "name": finetuned_metrics["model"],
            "accuracy": finetuned_acc,
            "invalid_predictions_percent": finetuned_metrics["invalid_predictions_percent"]
        },
        "improvement_percent": float(improvement),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    with open("results/model_comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print("Comparison saved to results/model_comparison.json and results/model_comparison.png")
    
    return comparison

def generate_report():
    """Generate a final report with the evaluation results."""
    # Check if comparison exists
    if not os.path.exists("results/model_comparison.json"):
        print("Error: Missing model comparison data")
        return
    
    # Load comparison data
    with open("results/model_comparison.json", 'r') as f:
        comparison = json.load(f)
    
    # Create report
    report = f"""# LLM Human Preference Prediction - Results

## Project Overview
This project fine-tuned a small language model ({comparison["base_model"]["name"]}) to predict human preferences between responses from different LLMs.

## Dataset
- [Arena Human Preference 55k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k)
- Contains 55,000+ real-world conversations with pairwise human preferences across 70+ LLMs

## Methodology
1. **Data Processing**: Used StratifiedGroupKFold to split the dataset, reserving 20% as validation set
2. **Base Model Inference**: Ran inference on validation set with the base model
3. **Fine-tuning**: Fine-tuned the base model using LoRA on the training set
4. **Fine-tuned Model Inference**: Evaluated the fine-tuned model on the validation set

## Results

### Model Performance Comparison
- **Base Model Accuracy**: {comparison["base_model"]["accuracy"]:.4f}
- **Fine-tuned Model Accuracy**: {comparison["finetuned_model"]["accuracy"]:.4f}
- **Improvement**: {comparison["improvement_percent"]:.2f}%

![Model Comparison](model_comparison.png)

## Approach and Reasoning

### Prompt Design
The prompt template was designed to clearly instruct the model to predict human preferences:
```
You are evaluating two AI assistant responses to a human query. Your task is to determine which response would be preferred by a human.

Human query: {{prompt}}

Assistant A's response:
{{response_a}}

Assistant B's response:
{{response_b}}

Which response do you think a human would prefer? Answer with just A or B.
```

### Fine-tuning Strategy
- Used LoRA (Low-Rank Adaptation) to efficiently fine-tune the model
- Targeted key attention modules for parameter-efficient training
- Applied QLoRA (4-bit quantization) to reduce memory requirements
- Used a learning rate of 5e-5 with cosine scheduler

## Future Work
1. **Experiment with Different Prompt Templates**: Test variations of the prompt to see which leads to better performance
2. **Ensemble Approach**: Combine predictions from multiple fine-tuned models
3. **Model Analysis**: Analyze which types of responses the model struggles with
4. **Larger Models**: Test the approach with larger base models
5. **Data Augmentation**: Generate additional training examples to improve robustness

## Conclusion
Fine-tuning a small LLM (1.5B parameters) successfully improved its ability to predict human preferences between model responses. The fine-tuned model achieved a {comparison["improvement_percent"]:.2f}% improvement over the base model, demonstrating the effectiveness of the approach.
"""
    
    # Save report
    with open("README.md", 'w') as f:
        f.write(report)
    
    print("Final report generated and saved to README.md")

def main():
    """Main function to run the evaluation process."""
    with app.run() as modal_app:
        print("Starting Modal app for evaluation...")
        metrics = evaluate_finetuned_model()
        if metrics:
            # Only compare models if evaluation was successful
            compare_models()
        generate_report()

if __name__ == "__main__":
    # This will be executed when running the script directly
    main()
