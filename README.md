# depression-subtype-classification
Multi-class depression subtype classification from social media text using MentalBERT, Twitter-RoBERTa, TF-IDF + SVM, Optuna, and LIME.


# Multi-Class Depression Type Classification from Social Media Text

A deep learning NLP project for classifying five depression subtypes from social media text, with comparison against a traditional machine learning baseline.

## Overview

This project explores whether transformer-based language models can better capture the contextual and linguistic differences between depression subtypes than traditional text classification methods.

The dataset contains 12,998 social media texts across five classes:

- Atypical depression
- Bipolar depression
- Major depressive disorder
- Postpartum depression
- Psychotic depression

Three modelling approaches were compared:

- TF-IDF + Support Vector Machine
- MentalBERT
- Twitter-RoBERTa

## Key Results

| Model | Accuracy | F1-Macro |
|---|---:|---:|
| TF-IDF + SVM | 76.10% | 75.86% |
| Twitter-RoBERTa | 82.97% | 83.57% |
| MentalBERT | 83.33% | 83.18% |

Transformer models substantially improved performance over the traditional baseline, particularly for depression subtypes requiring more contextual linguistic understanding.

## Methodology

The project includes:

- exploratory analysis of social media text
- TF-IDF + SVM baseline modelling
- fine-tuning MentalBERT
- fine-tuning Twitter-RoBERTa
- class-weighted cross-entropy for class imbalance
- Bayesian hyperparameter optimisation with Optuna
- F1-macro based evaluation
- confusion matrix and error analysis
- local model explanations using LIME

## Model Design

### TF-IDF + SVM

A traditional machine learning baseline using a 5,000-dimensional TF-IDF representation with Support Vector Machines.

### MentalBERT

A BERT-based language model pre-trained on mental health related text and fine-tuned for five-class depression classification.

### Twitter-RoBERTa

A RoBERTa model pre-trained on large-scale Twitter data, providing representations adapted to social-media language.

## Class Imbalance

The dataset contains moderate class imbalance.

Class-weighted cross-entropy was used during transformer training so minority classes contributed more strongly to the loss function.

## Hyperparameter Optimisation

Optuna was used to perform Bayesian hyperparameter search across:

- learning rate
- batch size
- weight decay
- warmup ratio

Twenty trials were evaluated for each transformer model, with pruning used to terminate poorly performing runs early.

## Interpretability

LIME was used to inspect individual predictions and identify words contributing to model decisions.

Examples of subtype-specific signals included:

- atypical depression: hypersomnia, overeating, sensitivity
- postpartum depression: baby, pregnancy, temporal references
- bipolar depression: racing, energy, thoughts

## Evaluation

F1-macro was selected as the primary evaluation metric because it gives equal importance to each depression subtype and is more informative than overall accuracy under class imbalance.

The largest transformer improvement was observed for major depressive disorder, where contextual modelling helped distinguish it from linguistically similar classes.

## Limitations

This project is a research prototype and is not intended for clinical diagnosis.

Key limitations include:

- social-media-specific data
- English-only text
- demographic and platform bias
- no clinical validation
- text-only input
- residual confusion between clinically overlapping subtypes

Any real-world deployment would require clinical validation, human oversight, privacy safeguards, fairness evaluation, and regulatory review.

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- MentalBERT
- Twitter-RoBERTa
- scikit-learn
- Optuna
- LIME
- Pandas
- NumPy
- Matplotlib

## Repository Contents

- `depression_classification.ipynb`  
  Complete preprocessing, baseline modelling, transformer fine-tuning, hyperparameter optimisation, evaluation and interpretability workflow.

## Running the Notebook

1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/depression-subtype-classification.git
cd depression-subtype-classification
