# 📄 Resume Screening using BERT + Tabular Features

An intelligent system to automatically assess job candidates based on their resumes and structured profile data, predicting whether they should be shortlisted for an interview.

This project simulates a real-world human resource analytics pipeline, integrating **Natural Language Processing (NLP)** and **tabular data modeling** in a **multimodal deep learning** setup.

---

##  Objective

To build a hybrid model that learns from both unstructured **resume text** and structured **profile features** to predict candidate shortlisting likelihood.

---

##  Data Preprocessing

The Kaggle `resume` dataset is preprocessed using standard text and data cleaning steps:
- Removed URLs and web links
- Removed special characters and emojis
- Extracted categorical and numerical features using Python’s `re` library

---

##  Feature Engineering

The following features are extracted or derived from resumes:
-  Degree Field  
-  Years of Experience (YOE)  
-  Number of Previous Jobs  
-  Booleans indicating presence of GitHub, LinkedIn, internship experience, etc.

---

##  Dataset Labelling

The original dataset is unlabelled. To create labels for training:
- A **heuristic approach** is used (based on features like job experience, project count, internship presence, etc.)
- CGPA is **not used in labelling** due to its frequent absence

---

##  Dataset Preparation

### Tabular Features:
- Categorical columns → One-hot encoding  
- Numerical columns → Standardized using `StandardScaler`

### Textual Features:
- BERT tokenizer from `bert-base-uncased` is used  
- Tokenized resume text into 128-dim vectors with padding and truncation

### Dataset Class:
- A PyTorch `ResumeBERTDataset` class returns:
  - BERT input tokens
  - Tabular feature tensors
  - Corresponding binary labels

---

##  Model Architecture: `BERTTabularModel`

A **multimodal deep learning model** with two branches:

-  **Text Branch**  
  Uses frozen `bert-base-uncased` model to extract dense embeddings from resume texts

-  **Tabular Branch**  
  A small MLP to handle structured profile features

-  **Fusion Layer**  
  The outputs of both branches are concatenated and passed to a classification head to produce the final prediction

---

##  Training Setup

- **Loss Function**: Binary Cross-Entropy (`BCELoss`)  
- **Optimizer**: Adam (learning rate = `1e-3`)  
- **Epochs**: 10  
- **Batch Size**: 16

Each epoch logs:
- Train and validation **loss** & **accuracy**
- **Precision**, **Recall**, **F1-score**
- **ROC-AUC** and **ROC Curve**

All logs are saved to `../models/training_logs.txt`. ROC and metric plots are also saved.

---

##  Evaluation

The final model evaluation includes:
- Loss & accuracy curves  
- ROC Curve  
- Precision, Recall, F1, and AUC metrics  
- Model weights saved as `bert_tabular_model.pth`(Although added to gitignore)

---

##  TODO

-  Build a FastAPI-based interface to simulate deployment
-  Compare with baseline MLP and a pure BERT-only model
-  Improve resume upload & evaluation mechanism in pipeline
-  Replace current dataset with a higher-quality, deduplicated resume dataset
-  Explore better labeling heuristics or semi-supervised techniques

