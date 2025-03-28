#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extract training logs from a text file and save them in a format suitable for analysis.
"""

import re
import argparse

def extract_training_metrics(input_file, output_file):
    """Extract training metrics from log file and save to a clean format."""
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Extract metrics lines
    pattern = r"{'loss': [\d\.]+, 'grad_norm': (?:nan|[\d\.]+), 'learning_rate': [\d\.\-e]+, 'mean_token_accuracy': [\d\.]+, 'epoch': [\d\.]+}"
    metrics_lines = re.findall(pattern, content)
    
    # Write to output file
    with open(output_file, 'w') as f:
        for line in metrics_lines:
            f.write(line + '\n')
    
    print(f"Extracted {len(metrics_lines)} metrics entries to {output_file}")
    return len(metrics_lines)

def main():
    parser = argparse.ArgumentParser(description='Extract training metrics from logs')
    parser.add_argument('--input_file', type=str, required=True, help='Input log file')
    parser.add_argument('--output_file', type=str, required=True, help='Output file for extracted metrics')
    
    args = parser.parse_args()
    extract_training_metrics(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
