# 🤖 LLM Classification Finetuning

This project is my solution submission to the Kaggle competition [LLM Classification Finetuning](https://www.kaggle.com/competitions/llm-classification-finetuning), where the task is to build a classifier that can predict human preferences between two LLM-generated responses to a given prompt.

## 📌 Problem Statement

In each instance, a prompt and two generated responses (A and B) are provided. The goal is to predict which response is preferred by a human annotator, or if the responses are equally good (tie).

The task is formulated as a **3-class classification problem**:

- `winner_model_a`
- `winner_model_b`
- `winner_tie`

The evaluation metric is **Log Loss**, meaning models are rewarded for confidence in correct predictions.

## 🧠 Model Architecture

- **Backbone**: `DeBERTaV3 Base` from `keras-nlp`
- Each prompt+response pair (A and B) is encoded separately.
- The embeddings are concatenated and pooled.
- A dense layer with softmax activation outputs a 3-class probability.

## 🧪 Training Details

- Sequence length: 384
- Batch size: 8 (using mixed precision)
- Epochs: 10 with EarlyStopping
- Scheduler: Cosine Learning Rate Decay
- Optimizer: Adam with learning rate `5e-6`

## 📁 Repository Structure

```
llm-classification-finetuning/
├── notebook/
│   └── llm_classification_final.ipynb
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   └── train.py
├── submission/
│   └── submission.csv
├── requirements.txt
└── README.md
```

## 🏆 Competition Result

- Public Score: `1.04737` (Log Loss)
- Submission File: [submission/submission.csv](submission/submission.csv)

## 🚀 How to Run

```bash
pip install -r requirements.txt
python src/train.py
```

## 📚 Future Work

- Try pairwise margin loss instead of classification.
- Add response quality augmentation.
- Use ensemble of multiple backbones.

## 📌 Acknowledgments

- Kaggle, for hosting the competition
- keras-nlp team for pretrained DeBERTaV3
