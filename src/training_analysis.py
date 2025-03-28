#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training analysis script for LLM fine-tuning.
Parses training logs and generates plots for loss, accuracy, and other metrics.
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def parse_training_logs(log_file):
    """Parse training logs to extract metrics."""
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    # Skip header lines that start with #
    log_lines = [line for line in log_content.split('\n') if line.strip() and not line.startswith('#')]
    
    # Try different parsing approaches
    metrics_data = []
    
    # First try to parse as JSON objects
    for line in log_lines:
        try:
            # Clean up the line if needed
            clean_line = line.strip()
            # Parse the JSON object
            metrics = json.loads(clean_line)
            if 'loss' in metrics:
                metrics_data.append({
                    'loss': float(metrics.get('loss', 0)),
                    'learning_rate': float(metrics.get('learning_rate', 0)),
                    'mean_token_accuracy': float(metrics.get('mean_token_accuracy', 0)),
                    'epoch': float(metrics.get('epoch', 0)),
                    'grad_norm': metrics.get('grad_norm', 'nan')
                })
        except json.JSONDecodeError:
            # If JSON parsing fails, try regex as fallback
            pass
    
    # If JSON parsing didn't work, try regex
    if not metrics_data:
        print("JSON parsing failed, trying regex pattern matching...")
        pattern = r"{'loss': ([\d\.]+), 'grad_norm': (?:nan|[\d\.]+), 'learning_rate': ([\d\.\-e]+), 'mean_token_accuracy': ([\d\.]+), 'epoch': ([\d\.]+)}"
        matches = re.findall(pattern, log_content)
        
        if matches:
            for match in matches:
                metrics_data.append({
                    'loss': float(match[0]),
                    'learning_rate': float(match[1]),
                    'mean_token_accuracy': float(match[2]),
                    'epoch': float(match[3])
                })
    
    # Convert to DataFrame
    if not metrics_data:
        print("Warning: No metrics data found in the log file!")
        print("First few lines of the log file:")
        for line in log_lines[:5]:
            print(f"  {line}")
        # Return an empty DataFrame with the expected columns
        return pd.DataFrame(columns=['loss', 'learning_rate', 'mean_token_accuracy', 'epoch', 'step'])
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Add step numbers
    metrics_df['step'] = range(1, len(metrics_df) + 1)
    
    print(f"Successfully parsed {len(metrics_df)} metrics entries")
    return metrics_df

def plot_metrics(metrics_df, output_dir):
    """Generate plots for training metrics."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Check if DataFrame is empty
    if metrics_df.empty:
        print("Error: No metrics data to plot. Please check the log file format.")
        # Create a simple error plot
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No metrics data found", ha='center', va='center', fontsize=20)
        plt.axis('off')
        error_plot_path = os.path.join(output_dir, f'error_no_data_{timestamp}.png')
        plt.savefig(error_plot_path)
        print(f"Error plot saved to {error_plot_path}")
        return {'error': 'No metrics data found'}
    
    # Set style for plots
    plt.style.use('ggplot')
    
    # Plot loss
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_df['step'], metrics_df['loss'], 'b-', linewidth=2)
    plt.title('Training Loss over Steps', fontsize=16)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True)
    
    # Add smoothed loss line
    window_size = min(50, len(metrics_df) // 10)
    if window_size > 0:
        smoothed_loss = metrics_df['loss'].rolling(window=window_size).mean()
        plt.plot(metrics_df['step'], smoothed_loss, 'r-', linewidth=2, alpha=0.7, 
                label=f'Moving Average (window={window_size})')
        plt.legend()
    
    plt.tight_layout()
    loss_plot_path = os.path.join(output_dir, f'loss_plot_{timestamp}.png')
    plt.savefig(loss_plot_path)
    print(f"Loss plot saved to {loss_plot_path}")
    
    # Plot accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_df['step'], metrics_df['mean_token_accuracy'], 'g-', linewidth=2)
    plt.title('Mean Token Accuracy over Steps', fontsize=16)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True)
    
    # Add smoothed accuracy line
    if window_size > 0:
        smoothed_acc = metrics_df['mean_token_accuracy'].rolling(window=window_size).mean()
        plt.plot(metrics_df['step'], smoothed_acc, 'r-', linewidth=2, alpha=0.7, 
                label=f'Moving Average (window={window_size})')
        plt.legend()
    
    plt.tight_layout()
    acc_plot_path = os.path.join(output_dir, f'accuracy_plot_{timestamp}.png')
    plt.savefig(acc_plot_path)
    print(f"Accuracy plot saved to {acc_plot_path}")
    
    # Plot learning rate
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_df['step'], metrics_df['learning_rate'], 'c-', linewidth=2)
    plt.title('Learning Rate over Steps', fontsize=16)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    lr_plot_path = os.path.join(output_dir, f'learning_rate_plot_{timestamp}.png')
    plt.savefig(lr_plot_path)
    print(f"Learning rate plot saved to {lr_plot_path}")
    
    # Create combined plot
    plt.figure(figsize=(14, 10))
    
    # Loss subplot
    plt.subplot(3, 1, 1)
    plt.plot(metrics_df['step'], metrics_df['loss'], 'b-', linewidth=2)
    if window_size > 0:
        plt.plot(metrics_df['step'], smoothed_loss, 'r-', linewidth=2, alpha=0.7, 
                label=f'Moving Average (window={window_size})')
        plt.legend()
    plt.title('Training Loss', fontsize=14)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True)
    
    # Accuracy subplot
    plt.subplot(3, 1, 2)
    plt.plot(metrics_df['step'], metrics_df['mean_token_accuracy'], 'g-', linewidth=2)
    if window_size > 0:
        plt.plot(metrics_df['step'], smoothed_acc, 'r-', linewidth=2, alpha=0.7, 
                label=f'Moving Average (window={window_size})')
        plt.legend()
    plt.title('Mean Token Accuracy', fontsize=14)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True)
    
    # Learning rate subplot
    plt.subplot(3, 1, 3)
    plt.plot(metrics_df['step'], metrics_df['learning_rate'], 'c-', linewidth=2)
    plt.title('Learning Rate', fontsize=14)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.grid(True)
    
    plt.tight_layout()
    combined_plot_path = os.path.join(output_dir, f'combined_metrics_plot_{timestamp}.png')
    plt.savefig(combined_plot_path)
    print(f"Combined metrics plot saved to {combined_plot_path}")
    
    # Save metrics to CSV
    metrics_csv_path = os.path.join(output_dir, f'training_metrics_{timestamp}.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Training metrics saved to {metrics_csv_path}")
    
    # Generate summary statistics with safety checks
    summary = {'total_steps': len(metrics_df)}
    
    # Only calculate statistics if we have data
    if len(metrics_df) > 0:
        # Loss statistics
        summary['initial_loss'] = metrics_df['loss'].iloc[0]
        summary['final_loss'] = metrics_df['loss'].iloc[-1]
        summary['loss_reduction'] = metrics_df['loss'].iloc[0] - metrics_df['loss'].iloc[-1]
        
        # Avoid division by zero
        if metrics_df['loss'].iloc[0] != 0:
            summary['loss_reduction_percent'] = (1 - metrics_df['loss'].iloc[-1] / metrics_df['loss'].iloc[0]) * 100
        else:
            summary['loss_reduction_percent'] = 0
        
        # Accuracy statistics
        summary['initial_accuracy'] = metrics_df['mean_token_accuracy'].iloc[0]
        summary['final_accuracy'] = metrics_df['mean_token_accuracy'].iloc[-1]
        summary['accuracy_improvement'] = metrics_df['mean_token_accuracy'].iloc[-1] - metrics_df['mean_token_accuracy'].iloc[0]
        
        # Avoid division by zero
        if metrics_df['mean_token_accuracy'].iloc[0] != 0:
            summary['accuracy_improvement_percent'] = (metrics_df['mean_token_accuracy'].iloc[-1] / metrics_df['mean_token_accuracy'].iloc[0] - 1) * 100
        else:
            summary['accuracy_improvement_percent'] = 0
            
        summary['min_loss'] = metrics_df['loss'].min()
        summary['max_accuracy'] = metrics_df['mean_token_accuracy'].max()
        summary['timestamp'] = timestamp
    
    # Save summary to JSON
    summary_path = os.path.join(output_dir, f'training_summary_{timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Training summary saved to {summary_path}")
    
    return {
        'metrics_df': metrics_df,
        'loss_plot_path': loss_plot_path,
        'acc_plot_path': acc_plot_path,
        'lr_plot_path': lr_plot_path,
        'combined_plot_path': combined_plot_path,
        'metrics_csv_path': metrics_csv_path,
        'summary_path': summary_path,
        'summary': summary
    }

def main():
    parser = argparse.ArgumentParser(description='Analyze training logs and generate plots')
    parser.add_argument('--log_file', type=str, required=True, help='Path to the training log file')
    parser.add_argument('--output_dir', type=str, default='results/training_analysis', help='Directory to save plots and metrics')
    
    args = parser.parse_args()
    
    # Parse logs and generate plots
    metrics_df = parse_training_logs(args.log_file)
    results = plot_metrics(metrics_df, args.output_dir)
    
    # Print summary
    print("\nTraining Summary:")
    print(f"Total Steps: {results['summary']['total_steps']}")
    print(f"Initial Loss: {results['summary']['initial_loss']:.4f}")
    print(f"Final Loss: {results['summary']['final_loss']:.4f}")
    print(f"Loss Reduction: {results['summary']['loss_reduction']:.4f} ({results['summary']['loss_reduction_percent']:.2f}%)")
    print(f"Initial Accuracy: {results['summary']['initial_accuracy']:.4f}")
    print(f"Final Accuracy: {results['summary']['final_accuracy']:.4f}")
    print(f"Accuracy Improvement: {results['summary']['accuracy_improvement']:.4f} ({results['summary']['accuracy_improvement_percent']:.2f}%)")
    print(f"Minimum Loss: {results['summary']['min_loss']:.4f}")
    print(f"Maximum Accuracy: {results['summary']['max_accuracy']:.4f}")

if __name__ == "__main__":
    main()
