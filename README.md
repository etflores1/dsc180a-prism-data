# **dsc180a-prism-data**

This repository contains all of our work for the **DSC 180A: Prism Data Project (Quarter 1)**.
The focus of this project is cleaning financial transaction data, engineering features, building multiple classification models, and analyzing consumer income patterns.

---

## **Repository Structure**

```
00_eda.ipynb
01_memos.ipynb
01_memos_func_renamed.ipynb
02_baseline_models.ipynb
03_preprocessing.ipynb
03.1_preprocessing_revisited_with_education_fix.ipynb
04_hugging_face_model.ipynb
05_DistilBERT.ipynb
05_HistGradientBoostingClassifier.ipynb
05_more_adv_models.ipynb
05_tiny_transformer_fusion_classifier.ipynb
06_income_eda.ipynb
README.md
.ipynb_checkpoints/
```

---

## **1. Data Cleaning & Preprocessing**

Performed in **03_preprocessing.ipynb** and the updated version **03.1_preprocessing_revisited_with_education_fix.ipynb**.

Steps include:

* Parsing and cleaning transaction memos
* Fixing dates and numeric fields
* Handling rare categories
* Building a reusable `DateAmountFeaturizer` (day of week, DOM, hour, whole-dollar, binned amounts)
* Full preprocessing pipeline for modeling

---

## **2. Models Implemented**

### **Baseline Models**

* Logistic Regression
* Linear SVC
* Complement Naive Bayes

### **Tree-Based Model**

* **HistGradientBoostingClassifier**

  * TF-IDF → SVD → Boosted Trees
  * Good speed and performance

### **Neural + Transformer Models**

* Tiny Transformer Fusion Model
* Feed-Forward MLP
* **DistilBERT Embeddings + Logistic Regression**
* **BERT / RoBERTa (HuggingFace transformers)**

Model notebooks include accuracy, macro-F1, latency, classification reports, and confusion matrices.

Relevant notebooks:

* `02_baseline_models.ipynb`
* `04_hugging_face_model.ipynb`
* `05_DistilBERT.ipynb`
* `05_more_adv_models.ipynb`
* `05_tiny_transformer_fusion_classifier.ipynb`
* `05_HistGradientBoostingClassifier.ipynb`

---

## **3. Income EDA & Regularity Detection**

Performed in **06_income_eda.ipynb**.

Includes:

### **Consumer-Level Statistics**

* Transactions per consumer
* Total inflow dollars
* Average & std inflow amount
* Active days (first → last transaction)
* Percentile-based income buckets

### **Major Income Sources**

Primary income categories identified:

* PAYCHECK
* INVESTMENT_INCOME
* OTHER_BENEFITS
* UNEMPLOYMENT_BENEFITS

### **Regular Income Detection**

A custom algorithm detects:

* Weekly
* Biweekly
* Monthly
* “Every *N* Days” patterns

Using:

* Amount clustering
* Interval stability
* Thresholds for tolerance

**Results**

* 1318 regular income patterns found
* 1046 unique consumers with regular income cycles

---

## **📝 Key Questions Addressed**

### **What counts as income?**

PAYCHECK, BENEFITS, and INVESTMENT_INCOME were treated as true income.
Transfers, loans, refunds, and cash deposits were **not** treated as income.

### **Does regularity matter for all income?**

No — benefits and investment income are often irregular but still count as income.

### **How do we detect “regular” paychecks?**

Using:

* Grouped amounts (± tolerance)
* Differences between payment dates
* Classifying cycles (weekly / biweekly / monthly)

---

## **How to Run This Project**

1. Run `03_preprocessing.ipynb` to build the cleaned dataframe.
2. Run any of the modeling notebooks to train models.
3. Use `06_income_eda.ipynb` for income and consumer-level analysis.

---

## **Notes**

* All work was run on UCSD Datahub.
* Transformer models may require additional RAM/time.
* HistGradientBoostingClassifier is the best non-deep-learning model for speed vs accuracy.

