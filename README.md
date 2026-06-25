Scam Guard BERT Detector

This project is a text classification model that detects scam and phishing messages using a fine-tuned BERT model.

It is trained on a mixture of Discord phishing datasets and custom scam message datasets.

Overview

The goal of this project is to classify a message as either scam or not scam using a pretrained multilingual BERT model.

The model takes a text message as input and outputs a probability score indicating whether the message is malicious.

Dataset

The dataset is built from multiple sources including:
- Hugging Face Discord phishing datasets
- Custom CSV datasets containing scam and non-scam messages

All datasets are merged into a single training set.

Preprocessing

- Text is tokenized using bert-base-multilingual-cased tokenizer
- Sequences are padded and truncated to a maximum length of 128 tokens
- Attention masks are used to ignore padding tokens
- Labels are converted to float tensors for binary classification

Model Architecture

The model uses a pretrained BERT encoder:

- bert-base-multilingual-cased as the backbone
- CLS token representation extracted from final hidden state
- Linear classification layer on top
- Output is a single logit for binary classification

Training

The model is trained using:
- Binary Cross Entropy with Logits Loss
- Adam optimizer

Data is shuffled and split into training and test sets.

Training is done using mini-batches with a DataLoader.

Evaluation

The model is evaluated using:
- Loss on test set
- Accuracy based on thresholded sigmoid output

Limitations

This model is not production-ready and may suffer from:

- Dataset imbalance
- Overfitting due to limited data diversity
- Weak generalization across different scam styles
- Lack of advanced evaluation metrics like precision and recall

Usage

After training, the model can be used to classify new messages as scam or not scam by passing tokenized input through the network and applying a sigmoid threshold.
