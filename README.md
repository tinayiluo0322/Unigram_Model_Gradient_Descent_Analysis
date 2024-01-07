# Unigram Model Analysis with PyTorch and NLTK

## Overview
This repository contains a Python script that demonstrates the implementation and analysis of a Unigram Probability Model using PyTorch and NLTK. The script focuses on understanding the distribution of character frequencies in a given text corpus using a statistical approach.

## Purpose
The primary purpose of this code is to:
1. Build a Unigram model to estimate the probabilities of individual characters in a text.
2. Utilize gradient descent optimization to refine these probability estimates.
3. Analyze and visualize the results to compare the estimated probabilities with the true character frequencies in the text.

## Code Breakdown
### Imports and Definitions
- **Libraries**: The script uses `nltk`, `numpy`, `torch`, and `matplotlib`.
- **Helper Functions**:
  - `onehot`: Creates a one-hot encoding for a token given a vocabulary.
  - `logit`: Computes the inverse sigmoid of a given value.
  - `normalize`: Normalizes a tensor to sum up to 1.
  - `loss_fn`: Calculates the loss to maximize probability.

### Unigram Model Class
- **Class Definition**: `Unigram` is a PyTorch `nn.Module` class.
- **Parameters**: Initializes with vocabulary size `V`.
- **Forward Method**: Calculates the estimated probabilities of each character in the vocabulary.

### Gradient Descent Example
- **Function**: `gradient_descent_example` demonstrates the entire process.
- **Steps**:
  1. **Data Preparation**: Tokenizes text data and converts it into one-hot encoded format.
  2. **Model Initialization**: Sets up the Unigram model and parameters for gradient descent.
  3. **Training Loop**: Applies gradient descent to optimize the model's parameter.
  4. **Loss Calculation**: Computes and records the loss during training.
  5. **Result Visualization**: Plots the loss history and compares the estimated probabilities with the actual character frequencies.

### Main Execution
- The script executes `gradient_descent_example` if run as the main program.

## Dataset
The script uses text from Jane Austen's "Sense and Sensibility" as provided by the `nltk` library.

## Visualization
- **Loss Graph**: Shows the change in loss across iterations.
- **Probability Distributions**: Compares the estimated character probabilities with the actual frequencies from the text.

## How to Use
1. **Setup**: Ensure Python environment has PyTorch, NLTK, NumPy, and Matplotlib installed.
2. **Run Script**: Execute the script to see the Unigram model in action and visualize the results.
