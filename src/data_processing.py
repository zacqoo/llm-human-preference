#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data processing script for LLM human preference prediction.
Downloads and processes the Arena Human Preference dataset,
then splits it using StratifiedGroupKFold.
"""

import os
import json
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def download_dataset():
    """Download the dataset from HuggingFace."""
    print("Downloading dataset...")
    dataset = load_dataset("lmarena-ai/arena-human-preference-55k")
    return dataset

def process_dataset(dataset):
    """Process the dataset into a format suitable for training."""
    print("Processing dataset...")
    
    # Convert to pandas DataFrame for easier manipulation
    df = pd.DataFrame(dataset["train"])
    
    # First, let's inspect the structure of the data
    print("\nDataset structure sample:")
    sample_row = df.iloc[0].to_dict()
    for key, value in sample_row.items():
        print(f"{key}: {type(value).__name__}")
        if isinstance(value, (dict, list)):
            print(f"  Sample content: {value}")
        elif isinstance(value, str) and len(value) > 100:
            print(f"  Sample content: {value[:100]}...")
        else:
            print(f"  Sample content: {value}")
    
    # Based on inspection, process the data accordingly
    # Extract model names from the dataset
    df['model_a'] = df['model_a']
    df['model_b'] = df['model_b']
    
    # Clean response text (remove list brackets if present)
    df['response_a_text'] = df['response_a'].apply(lambda x: json.loads(x)[0] if x.startswith('[') else x)
    df['response_b_text'] = df['response_b'].apply(lambda x: json.loads(x)[0] if x.startswith('[') else x)
    
    # Clean prompt (take first item if it's a list)
    df['prompt'] = df['prompt'].apply(lambda x: json.loads(x)[0] if x.startswith('[') else x)
    
    # Create target variable with three categories:
    # 0: model A wins
    # 1: model B wins
    # 2: tie
    df['target'] = df.apply(lambda row: 
        0 if row['winner_model_a'] == 1 else 
        1 if row['winner_model_b'] == 1 else 
        2 if row['winner_tie'] == 1 else 
        -1, axis=1)
    
    # Verify no invalid targets
    invalid_targets = df[df['target'] == -1]
    if len(invalid_targets) > 0:
        print(f"Warning: Found {len(invalid_targets)} rows with invalid targets")
        print(invalid_targets.head())
    
    # Create a unique identifier for each conversation/prompt
    df['prompt_id'] = df['id'].astype(str)
    
    # Drop unnecessary columns
    df = df[['prompt_id', 'prompt', 'model_a', 'model_b', 
             'response_a_text', 'response_b_text', 'target']]
    
    return df

def analyze_dataset(df):
    """Analyze the dataset and print statistics."""
    print("\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    
    # Count unique prompts
    unique_prompts = df['prompt_id'].nunique()
    print(f"Unique prompts: {unique_prompts}")
    
    # Model distribution
    print("\nModel A distribution:")
    print(df['model_a'].value_counts().head(10))
    print("\nModel B distribution:")
    print(df['model_b'].value_counts().head(10))
    
    # Target distribution
    print("\nTarget distribution:")
    target_dist = df['target'].value_counts(normalize=True).sort_index()
    target_mapping = {
        0: "Model A wins",
        1: "Model B wins",
        2: "Tie"
    }
    for target, percentage in target_dist.items():
        print(f"{target_mapping.get(target, 'Unknown')}: {percentage:.2%}")
    
    # Create visualizations directory if it doesn't exist
    os.makedirs("results/visualizations", exist_ok=True)
    
    # Plot target distribution
    plt.figure(figsize=(10, 6))
    target_counts = df['target'].value_counts().sort_index()
    plt.bar([target_mapping[i] for i in target_counts.index], target_counts.values)
    plt.title('Distribution of Model Preferences')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/visualizations/target_distribution.png')
    plt.close()
    
    # Create visualizations directory if it doesn't exist
    os.makedirs("results/visualizations", exist_ok=True)
    
    # Plot model distributions
    plt.figure(figsize=(12, 8))
    top_models = pd.concat([df['model_a'], df['model_b']]).value_counts().head(15)
    sns.barplot(x=top_models.values, y=top_models.index)
    plt.title('Top 15 Models in the Dataset')
    plt.xlabel('Count')
    plt.tight_layout()
    plt.savefig('results/visualizations/model_distribution.png')
    
    # Plot target distribution
    plt.figure(figsize=(8, 6))
    df['target'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Distribution of Preferred Responses (A vs B)')
    plt.savefig('results/visualizations/target_distribution.png')
    
    return

def split_dataset(df, test_size=0.2, n_splits=5, random_state=42):
    """
    Split the dataset using StratifiedGroupKFold.
    
    This ensures:
    1. Prompts in the training set don't appear in the validation set
    2. The distribution of targets is similar in both sets
    """
    print("\nSplitting dataset...")
    
    # Create a stratified group k-fold splitter
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Get the split indices
    groups = df['prompt_id']
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Get the first fold as validation set
    for train_idx, val_idx in sgkf.split(X, y, groups):
        break
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    
    # Verify the split quality
    print("\nSplit Quality Check:")
    print(f"Training target distribution: {train_df['target'].value_counts(normalize=True)}")
    print(f"Validation target distribution: {val_df['target'].value_counts(normalize=True)}")
    
    # Check for prompt overlap
    train_prompts = set(train_df['prompt_id'])
    val_prompts = set(val_df['prompt_id'])
    overlap = train_prompts.intersection(val_prompts)
    print(f"Prompt overlap between train and validation: {len(overlap)}")
    
    # Check model distribution in both sets
    train_models = pd.concat([train_df['model_a'], train_df['model_b']]).value_counts(normalize=True)
    val_models = pd.concat([val_df['model_a'], val_df['model_b']]).value_counts(normalize=True)
    
    # Plot model distribution comparison
    plt.figure(figsize=(12, 8))
    top_models = pd.concat([df['model_a'], df['model_b']]).value_counts().head(10).index
    
    train_dist = train_models.loc[top_models].values
    val_dist = val_models.loc[top_models].values
    
    width = 0.35
    x = np.arange(len(top_models))
    
    plt.bar(x - width/2, train_dist, width, label='Train')
    plt.bar(x + width/2, val_dist, width, label='Validation')
    plt.xticks(x, top_models, rotation=45, ha='right')
    plt.title('Model Distribution in Train vs Validation Sets')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/visualizations/split_model_distribution.png')
    
    return train_df, val_df

def save_datasets(train_df, val_df, output_dir="data"):
    """Save the processed datasets to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    
    print(f"\nDatasets saved to {output_dir}")
    
    # Also save in jsonl format for easier loading with transformers
    with open(os.path.join(output_dir, "train.jsonl"), 'w') as f:
        for _, row in train_df.iterrows():
            f.write(json.dumps(row.to_dict()) + '\n')
            
    with open(os.path.join(output_dir, "val.jsonl"), 'w') as f:
        for _, row in val_df.iterrows():
            f.write(json.dumps(row.to_dict()) + '\n')
    
    print(f"Datasets also saved in jsonl format")

def main():
    """Main function to process and split the dataset."""
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Download and process dataset
    dataset = download_dataset()
    df = process_dataset(dataset)
    
    # Analyze dataset
    analyze_dataset(df)
    
    # Split dataset
    train_df, val_df = split_dataset(df)
    
    # Save datasets
    save_datasets(train_df, val_df)
    
    print("Data processing complete!")

if __name__ == "__main__":
    main()
