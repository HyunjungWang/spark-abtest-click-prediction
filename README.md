# 🚀 Digital Ad-Click Prediction & Placement Optimization

This project focuses on analyzing user behavior logs to optimize ad placement strategies and build a high-performance predictive model for ad engagement. Using **PySpark**, I processed large-scale data, conducted statistical hypothesis testing, and improved model performance from **AUC 0.63 to 0.805**.

## 📌 Project Overview
The objective was to identify the most effective ad positions and user demographics to maximize Click-Through Rate (CTR) and marketing ROI.

## 🛠️ Tech Stack
* **Engine:** PySpark (Spark SQL & MLlib)
* **Statistical Analysis:** Scipy (Chi-square Test)
* **Machine Learning:** Random Forest, Gradient Boosted Trees (GBT/XGBoost)
* **Infrastructure:** Databricks / Spark Cluster

---

## 📈 Key Analysis & Findings

### 1. Data Cleaning & Imputation
* **Challenge:** Encountered 45% missing values in categorical columns.
* **Solution:** Implemented business-logic-driven imputation (Mode for categorical, Median for numerical) and introduced 'Unknown' labels to preserve data patterns.

### 2. Statistical Evidence (A/B Testing)
To verify if ad position truly matters, I performed a **Chi-square Test of Independence**.
* **Observation:** The `Bottom` position yielded the highest CTR (**68.7%**), followed by `Top` (**63.5%**).
* **Statistical Result:** $p < 0.05$, confirming that the difference in CTR across positions is statistically significant and not due to random chance.



### 3. Feature Importance (Why do users click?)
I utilized a Random Forest model to rank the influence of various features on click probability.
* **Age (41.2%)**: The most critical factor. 
* **Browsing History (18.0%)**: Past behavior significantly outweighs gender in predicting clicks.
* **Gender (8.3%)**: Found to have the lowest impact, confirming my initial hypothesis.



### 4. Model Performance Leap: RF to GBT
I significantly improved the model's predictive accuracy by moving from Bagging to Boosting algorithms.
* **Baseline (Random Forest):** $AUC = 0.6300$
* **Optimized (Gradient Boosted Trees):** **$AUC = 0.8051$**
* **Inference:** The GBT model successfully captured non-linear interactions between user age groups and content consumption patterns at the bottom of the page.



---

## 🎯 Business Recommendations
1. **Target Segment:** Focus ad budget on the **20-30 age group**, which demonstrated the highest engagement (**68.4% CTR**).
2. **Placement Strategy:** Prioritize **Bottom placements** for long-form content, as users finishing the content show higher intent.
3. **Efficiency:** Use the GBT model to score users in real-time; focusing on the top decile of predicted users can potentially reduce ad waste by **~30%**.



