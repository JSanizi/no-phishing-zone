# 🛡️ No Phishing Zone  
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-Academic-green.svg)
![Status](https://img.shields.io/badge/status-Completed-success.svg)
![Repo size](https://img.shields.io/github/repo-size/JSanizi/no-phishing-zone)
![Last commit](https://img.shields.io/github/last-commit/JSanizi/no-phishing-zone)

*A Machine Learning–Based Spam and Phishing Email Detection System*  

**Author:** Janice Sakina Akiza Nizigama  
**University:** Aalborg University Copenhagen  
**Supervisor:** Yan Kyaw Tun  
**Project Period:** Spring 2025  
**Repository:** [JSanizi/no-phishing-zone](https://github.com/JSanizi/no-phishing-zone/tree/main)  

---

## 📖 Overview  

**No Phishing Zone** is a research-based engineering project exploring how various **machine learning algorithms** can be trained and compared to detect spam and phishing emails effectively.  
The project implements a complete spam-filtering pipeline, from **dataset preprocessing and vectorization** to **model training, tuning, and evaluation**, culminating in a **functional spam filter application** that simulates email classification.  

The study compares seven models:  
- Autoencoder (Anomaly Detection)  
- Naive Bayes  
- Logistic Regression  
- Random Forest  
- AdaBoost  
- Gradient Boost  
- K-Nearest Neighbors  

After comprehensive testing, **Random Forest** and **Logistic Regression** achieved the highest performance, with weighted F1-scores of **0.86** and **0.80**, respectively.  

---

## 🧠 Project Architecture  

### System Components  
1. **Data Preprocessing & Cleaning**  
   - Loads datasets (`SpamAssassin.csv`, `CEAS_08.csv`)  
   - Cleans text and removes noise  
   - Converts text into TF-IDF vectors  

2. **Model Training & Hyperparameter Tuning**  
   - Trains seven ML models using scikit-learn  
   - Performs parameter optimization with `GridSearchCV`  
   - Saves the best-performing models for deployment  

3. **Spam Filter Application**  
   - Connects to an email inbox (via IMAP)  
   - Sends test emails to simulate spam delivery  
   - Classifies incoming emails using trained models  
   - Displays confusion matrices and evaluation reports  

### Model Architecture Example – Autoencoder  

- Trained only on **non-spam** samples  
- Detects spam by measuring **reconstruction error thresholds**  

---

## ⚙️ Installation  

### Requirements  
- Python 3.10 or newer  
- `pip` package manager  

### Install dependencies

```bash
pip install -r requirements.txt
```


## 🚀 Usage  

Step-by-step on how to run your scripts:

```bash
# 1️⃣ Preprocess the datasets
python training_and_tunning_models/data_preprocessing.py

# 2️⃣ Train and tune models
python training_and_tunning_models/train_and_parametertuning.py

# 3️⃣ Run the spam filter application
python application/ac_filter.py
```

## 📊 Evaluation Metrics  

Models are evaluated with **Accuracy**, **Precision**, **Recall**, **Weighted F1-score**, and a **Confusion Matrix**.

| Model               | Weighted F1 | Precision | Recall | Accuracy |
|---------------------|-------------|-----------|---------|----------|
| Random Forest       | 0.86 | 0.87 | 0.84 | 0.86 |
| Logistic Regression | 0.80 | 0.82 | 0.78 | 0.81 |
| AdaBoost            | 0.78 | 0.76 | 0.79 | 0.78 |
| Gradient Boost      | 0.77 | 0.74 | 0.78 | 0.77 |
| Naive Bayes         | 0.75 | 0.73 | 0.76 | 0.75 |
| KNN                 | 0.71 | 0.69 | 0.70 | 0.71 |
| Autoencoder         | 0.68 | 0.66 | 0.70 | 0.68 |

---

## 🧩 Repository Structure

```text
no-phishing-zone/
├── application/
│   ├── ac_filter.py
│   ├── connect_to_email.py
│   └── sending_mail.py
│
├── tests/
│   ├── test_integration.py
│   └── test_unit.py
│
├── training_and_tunning_models/
│   ├── ac_parameter_tuning.py
│   ├── autoencoder.py
│   ├── data_preprocessing.py
│   ├── parametertuning_table.py
│   ├── split_data.py
│   └── train_and_parametertuning.py
│
├── spam_filter.py
├── start_parametertuning.py
├── .gitattributes
├── .gitignore
└── README.md
```

## 📬 Contact

Janice Sakina Akiza Nizigama

🔗 LinkedIn: https://www.linkedin.com/in/janice-nizigama

💻 GitHub: https://github.com/JSanizi

## 🧾 License

© 2025 Aalborg University — Academic / Non-Commercial use permitted with citation.
For other uses, please contact the author.
