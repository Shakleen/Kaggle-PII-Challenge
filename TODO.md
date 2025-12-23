# PII Data Detection - Kaggle Bronze Medal Solution (Top 8%)

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/data?select=train.json)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange?style=for-the-badge)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

## 🏆 Achievement
![Medal](images/pii-medal-dark.png)

* **Rank:** [173rd out of 2,048 (Top 8%)](https://www.kaggle.com/certification/competitions/ishfar/pii-detection-removal-from-educational-data)
* **Award:** Kaggle Bronze Medal 🥉
* **Private LB Score:** 0.76 (F5 Metric)

## 📌 Project Overview
This project was developed for the Kaggle "PII Data Detection" competition. The objective was to identify seven types of **Personally Identifiable Information (PII)** in student-submitted essays: Name, Address, Phone Number, Email, URL, ID Number, and Username.

### The Challenge: BIO Strategy
Since PII can span multiple tokens, I implemented a **BIO (Beginning, Intermediate, Outside)** tagging strategy. For example, the name "Shakleen Ishfar is a student" is classified as:
* **Shakleen**: `B-NAME`
* **Ishfar**: `I-NAME`
* **is**: `O`
* **a**: `O`
* **student**: `O`



### Data Constraints
* **Extreme Imbalance:** 5,862 out of 6,807 rows contained no PII.
* **Scarcity:** Certain classes like `Username` and `Address` had fewer than 5 original examples.
* **The F5 Metric:** The evaluation metric prioritized **Recall** 5x more than **Precision**. Missing a single PII token (False Negative) was significantly more costly than a False Positive.

---

## 🚀 Technical Strategy

![Flow chart](images/pii-flow-dark.png)

### 1. Synthetic Data Generation
To bridge the data gap, I leveraged **LLMs (Gemini, ChatGPT, Mistral, and LLaMa)** to generate **24,000 additional training examples**. This was crucial for teaching the model the nuances of rare classes like addresses and usernames.

### 2. Negative Sampling (Downsampling)
To prevent the model from becoming biased toward the "Outside" (non-PII) class:
1. Split the original data into **Positive** (contains PII) and **Negative** sets.
2. Created 3 distinct folds by pairing the entire Positive set with 1/3 of the Negative set.
3. Trained an ensemble of **3 DeBERTa models** on these balanced folds.

### 3. F5-Optimized Loss & Thresholding
* **Custom Loss Weights:** I assigned a significantly lower weight to the `O` class in the loss function. This forced the model to update weights more aggressively when it misclassified a PII token as `O`.
* **Confidence Thresholding:** Through hyperparameter tuning, I found that a **70% threshold** for the `O` class worked best. If the model’s confidence for "no PII" was below 70%, it would default to the most likely PII category.

### 4. Ensemble & Post-Processing
* **Weighted Ensemble:** Predictions from the 3 DeBERTa models were combined based on their confidence scores.
* **Rule-Based Refinement:** I used **Regex matching** for Phone Numbers and Usernames to ensure high-precision detection of structured strings.
* **Tagging Script:** A custom script handled the conversion of raw token predictions into valid BIO format.

---

## 🛠️ Tools & Technologies
* **Models:** DeBERTa-v3
* **LLMs for Augmentation:** Gemini, GPT-4, Mistral, LLaMa
* **Libraries:** PyTorch, Transformers, Scikit-Learn, Pandas, NumPy, Regex
* **Metric:** Custom F5-Score Implementation

---

## 📊 Results
| Metric | Score |
| :--- | :--- |
| **Private Leaderboard (F5)** | **0.76** |
| **Final Rank** | **173 / 2048** |
| **Medal** | **Bronze 🥉** |

---