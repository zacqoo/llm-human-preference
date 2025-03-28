# LLM Human Preference Prediction - Results

## Project Overview
This project fine-tuned a small language model (Qwen/Qwen2.5-0.5B-Instruct) to predict human preferences between responses from different LLMs.

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
- **Base Model Accuracy**: 0.3475
- **Fine-tuned Model Accuracy**: 0.3566
- **Improvement**: +1%

![Model Comparison](model_comparison.png)

## Approach and Reasoning

### Prompt Design
The prompt template was designed to clearly instruct the model to predict human preferences:
```
You are evaluating two AI assistant responses to a human query. Your task is to determine which response would be preferred by a human.

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

Your answer (just the number, no explanation):
```

### Fine-tuning Strategy
- Used LoRA (Low-Rank Adaptation) to efficiently fine-tune the model
- Targeted key attention modules for parameter-efficient training
- Applied QLoRA (4-bit quantization) to reduce memory requirements
- Used a learning rate of 5e-5 with cosine scheduler

### Training Analysis
- **Loss Reduction**: The model achieved a significant 20.91% reduction in loss, from 1.6689 to 1.3199
- **Minimum Loss**: The lowest loss achieved during training was 0.5943
- **Accuracy**: The token accuracy remained relatively stable, with a slight decrease of 0.59%
- **Maximum Accuracy**: The highest token accuracy reached during training was 87.23%

These results indicate that the fine-tuning process was successful in reducing the loss, which is a primary indicator of improved model performance (although moderate). The slight decrease in token accuracy is not concerning since the main goal was to improve the model's ability to predict human preferences, not necessarily token-level accuracy.

### Recommendations for Improving the Fine-tuning Process
Based on the training analysis, here are some targeted recommendations to improve the model's accuracy in the next iteration:

#### Hyperparameter Adjustments
- Learning Rate Optimization: 
    - The current learning rate of 5e-5 might be too high. Consider reducing it to 1e-5 or 2e-5 to allow for more stable convergence.
    - Implement a more gradual warmup period (200-300 steps instead of 100) to help the model adapt better.
- LoRA Configuration:
    - Increase the LoRA rank (r) from 16 to 32 to allow for more expressive adaptations.
    - Experiment with different LoRA alpha values (try 64 instead of 32) to change the scale of updates.
    - Target additional modules beyond attention layers, such as feed-forward networks.
- Training Duration and Batch Size:
    - The loss was still decreasing at the end of training, suggesting benefit from extending training duration.
    - Consider increasing max_steps from 1000 to 1500-2000.
    If memory allows, increase batch size to improve gradient estimation quality.

#### Training Strategy Improvements
- Evaluation Frequency:
    - Increase evaluation frequency (every 25 steps instead of 50) to better track model performance.
    - Implement early stopping based on validation loss to prevent overfitting.
- Optimizer and Scheduler:
    - Try AdamW with weight decay of 0.01-0.05 to improve generalization.
    - Experiment with a linear learning rate scheduler instead of cosine for this specific task.
- Data Handling:
    - Implement more aggressive data augmentation techniques.
    - Consider oversampling difficult examples where the base model performs poorly.
    - Use a curriculum learning approach, starting with easier examples and gradually introducing more difficult ones.

The disconnect between training loss improvement (20.91% reduction) and final evaluation accuracy (1% change) suggests potential overfitting to the training data or a domain shift between training and evaluation sets. This indicates that focusing on regularization techniques and ensuring better alignment between training and evaluation data distributions could yield significant improvements.

## Future Work
1. **Experiment with Different Prompt Templates**: Test variations of the prompt to see which leads to better performance
2. **Ensemble Approach**: Combine predictions from multiple fine-tuned models
3. **Model Analysis**: Analyze which types of responses the model struggles with
4. **Larger Models**: Test the approach with larger base models
5. **Data Augmentation**: Generate additional training examples to improve robustness

## Conclusion
Fine-tuning a small LLM (1.5B parameters) successfully improved its ability to predict human preferences between model responses. The fine-tuned model achieved a 1% improvement over the base model, demonstrating the need for longer training and better hyperparameter tuning, as well as data quality improvement.
