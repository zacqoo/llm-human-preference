#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inference script for LLM human preference prediction.
Prepares prompts and runs inference on the validation set using Modal.
"""

import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import modal
import time
from datetime import datetime

# Define Modal app
app = modal.App("llm-preference-inference")

# Create Modal image with necessary dependencies
image = (modal.Image.debian_slim()
    .pip_install(
        "vllm>=0.3.0",
        "transformers>=4.34.0",
        "torch>=2.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "accelerate>=0.23.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.66.0",
        "huggingface-hub>=0.17.0"
    )

)

# Define the model to use
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # Using Qwen2.5 0.5B Instruct model

@app.function(
    image=image,
    gpu="H100",  # Using H100 for faster inference
    timeout=7200,  # Increased timeout to 2 hours
    retries=2  # Retry on failure
)
def run_inference(batch, model_id, prompt_template):
    """Run inference on a batch of examples using vLLM for faster performance."""
    from vllm import LLM, SamplingParams
    import os
    from huggingface_hub import login
    
    # Initialize vLLM model with optimized settings for better throughput
    # Optimize vLLM initialization for H100 GPU
    llm = LLM(
        model=model_id,
        tensor_parallel_size=1,  # Will be automatically set by Modal
        gpu_memory_utilization=0.98,  # Maximize GPU memory usage
        max_num_batched_tokens=32768,  # Double the token batch size for H100
        max_num_seqs=512,  # Double the sequence capacity
        trust_remote_code=True,  # Required for Qwen model
        dtype="float16",  # Use half precision for speed
        max_model_len=4096,  # Reduced context for faster inference
        enforce_eager=True,  # Force eager execution for small models
        enable_prefix_caching=True,  # Enable prefix caching
        swap_space=1,  # Enable GPU swap space for better memory management
        disable_log_stats=True,  # Disable logging for better performance
        #quantization="awq"  # Use AWQ quantization for faster inference
    )
    
    # Function to truncate text if needed (simple character-based truncation for speed)
    def truncate_if_needed(text, max_length=1000):
        if len(text) > max_length:
            return text[:max_length] + "...[truncated]"
        return text
    
    # Prepare prompts for the batch with simple length checks
    prompts = []
    skipped_examples = []
    
    # Estimate token length using character count (faster than tokenization)
    # A rough estimate is 4 characters per token for English text
    char_to_token_ratio = 4
    max_char_length = 8000 * char_to_token_ratio  # ~8000 tokens
    
    for i, example in enumerate(batch):
        # Apply aggressive truncation for very long inputs
        truncated_prompt = truncate_if_needed(example["prompt"], 800)
        truncated_response_a = truncate_if_needed(example["response_a_text"], 2500)
        truncated_response_b = truncate_if_needed(example["response_b_text"], 2500)
        
        # Format the prompt
        formatted_prompt = prompt_template.format(
            prompt=truncated_prompt,
            response_a=truncated_response_a,
            response_b=truncated_response_b
        )
        
        # Simple character-based length check (much faster than tokenization)
        if len(formatted_prompt) > max_char_length:
            print(f"Warning: Prompt {i} is too long ({len(formatted_prompt)} chars). Truncating further.")
            # Try more aggressive truncation
            truncated_prompt = truncate_if_needed(example["prompt"], 500)
            truncated_response_a = truncate_if_needed(example["response_a_text"], 1500)
            truncated_response_b = truncate_if_needed(example["response_b_text"], 1500)
            
            formatted_prompt = prompt_template.format(
                prompt=truncated_prompt,
                response_a=truncated_response_a,
                response_b=truncated_response_b
            )
            
            # If still too long, skip it
            if len(formatted_prompt) > max_char_length:
                print(f"Warning: Prompt {i} is still too long. Skipping.")
                skipped_examples.append(i)
                continue
        
        prompts.append(formatted_prompt)
    
    # Set sampling parameters for direct answer
    sampling_params = SamplingParams(
        max_tokens=5,  # Reduced since we expect short answers
        temperature=0.0,  # Deterministic output
        stop=["\n", "."],  # Stop at newline or period
        frequency_penalty=0.0,
        presence_penalty=0.0,
        top_p=1.0  # No nucleus sampling
    )
    
    # Generate completions for all prompts in parallel
    outputs = llm.generate(prompts, sampling_params)
    
    # Process results
    results = []
    batch_idx = 0
    
    for i, example in enumerate(batch):
        # Skip examples that were too long
        if i in skipped_examples:
            # Add a default prediction for skipped examples
            results.append({
                "prompt_id": example["prompt_id"],
                "prediction": 2,  # Default to "equally good" for skipped examples
                "prediction_text": "2",
                "confidence": 0.0
            })
            continue
        
        # Process the output for this example
        output = outputs[batch_idx]
        batch_idx += 1
        prediction_text = output.outputs[0].text.strip().lower()
        
        # Parse the prediction (0, 1, or 2)
        try:
            prediction = int(prediction_text.strip())
            if prediction not in [0, 1, 2]:
                prediction = -1  # Invalid prediction
        except ValueError:
            prediction = -1  # Invalid prediction
        
        # Store the result
        result = {
            "prompt_id": example["prompt_id"],
            "prediction": prediction,
            "prediction_text": prediction_text,
            "true_label": example["target"]
        }
        results.append(result)
    
    return results

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

Which response would a human prefer? Output EXACTLY one number:
0 = Assistant A is better
1 = Assistant B is better
2 = They are equally good

Answer: """
    return prompt_template

def load_validation_data(file_path="data/val.csv"):
    """Load the validation dataset."""
    df = pd.read_csv(file_path)
    return df

def batch_data(df, batch_size=32):
    """Split the data into batches for parallel processing."""
    data = df.to_dict('records')
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def run_validation_inference():
    """Run inference on the validation set and evaluate the results."""
    print("Starting validation inference...")
    
    # Load validation data
    val_df = load_validation_data()
    print(f"Loaded {len(val_df)} validation examples")
    
    # Prepare prompt template
    prompt_template = prepare_prompt_template()
    
    # Use a larger batch size for better throughput, but still reasonable
    batch_size = 32  # Increased from 8 for better performance
    batches = batch_data(val_df, batch_size)
    print(f"Split data into {len(batches)} batches of size {batch_size}")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Run inference on each batch with robust error handling
    all_results = []
    start_time = time.time()
    failed_batches = []
    
    for i, batch in enumerate(tqdm(batches)):
        print(f"Processing batch {i+1}/{len(batches)}")
        try:
            # Process batch
            batch_results = run_inference.remote(batch, MODEL_ID, prompt_template)
            all_results.extend(batch_results)
            
                    # Save intermediate results less frequently to reduce I/O overhead
            if (i + 1) % 50 == 0 or i == len(batches) - 1:
                with open(f"results/inference_results_intermediate_{i+1}.json", 'w') as f:
                    json.dump(all_results, f, indent=2)
                print(f"Saved intermediate results after batch {i+1}/{len(batches)}")
        except Exception as e:
            print(f"Error processing batch {i}: {str(e)}")
            failed_batches.append((i, batch))
            
            # Save intermediate results immediately after any failure
            with open(f"results/inference_results_before_failure_batch_{i}.json", 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"Saved results after error in batch {i}")
            
            # Add default predictions for failed batch
            for example in batch:
                all_results.append({
                    "prompt_id": example["prompt_id"],
                    "prediction": 2,  # Default to "equally good"
                    "prediction_text": "2",
                    "true_label": example["target"]
                })
    
    # Try to process failed batches one by one with smaller context
    if failed_batches:
        print(f"Failed to process {len(failed_batches)} batches: {[i for i, _ in failed_batches]}")
        print("Attempting to process failed examples individually...")
        
        for batch_idx, batch in failed_batches:
            for example_idx, example in enumerate(batch):
                try:
                    # Process single example with a single-item batch
                    print(f"Retrying example {example_idx} from batch {batch_idx}")
                    single_result = run_inference.remote([example], MODEL_ID, prompt_template)
                    
                    # Find the position in all_results where we added the default prediction
                    result_idx = None
                    for idx, result in enumerate(all_results):
                        if result["prompt_id"] == example["prompt_id"] and result["prediction"] == 2 and result["prediction_text"] == "2":
                            result_idx = idx
                            break
                    
                    # Replace the default prediction with the actual result if found
                    if result_idx is not None:
                        all_results[result_idx] = single_result[0]
                        print(f"Successfully replaced result for example {example_idx} in batch {batch_idx}")
                except Exception as e:
                    print(f"Still failed on example {example_idx} in batch {batch_idx}: {str(e)}")
    
    end_time = time.time()
    print(f"Inference completed in {end_time - start_time:.2f} seconds")
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/base_model_inference_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(all_results)
    
    # Calculate accuracy
    valid_predictions = results_df[results_df['prediction'] != -1]
    accuracy = (valid_predictions['prediction'] == valid_predictions['true_label']).mean()
    
    # Calculate the percentage of invalid predictions
    invalid_percent = (results_df['prediction'] == -1).mean() * 100
    
    print(f"\nValidation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Invalid predictions: {invalid_percent:.2f}%")
    
    # Save evaluation metrics
    metrics = {
        "model": MODEL_ID,
        "accuracy": float(accuracy),
        "invalid_predictions_percent": float(invalid_percent),
        "timestamp": timestamp,
        "inference_time_seconds": end_time - start_time,
        "total_examples": len(results_df),
        "valid_examples": len(valid_predictions)
    }
    
    with open(f"results/base_model_metrics_{timestamp}.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    return metrics

# Entry point for Modal deployment
def main():
    run_validation_inference()

if __name__ == "__main__":
    # Deploy the app and run inference
    with app.run() as modal_app:
        main()
