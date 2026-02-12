# 🚀 Ad-Placement Optimization: A/B Testing & Real-time ML API

This project is an end-to-end machine learning system designed to analyze large-scale user behavior logs, predict ad-click probabilities, and dynamically determine the optimal ad placement using a FastAPI real-time decision engine.

## 📌 Project Overview
The objective was to identify the most effective ad positions and user demographics to maximize Click-Through Rate (CTR) and marketing ROI. The system bridges the gap between raw data analysis and real-time business decision-making.

## 🛠️ Tech Stack
* **Engine:** PySpark (Spark SQL & MLlib)
* **Big Data & ML**: PySpark (Spark MLlib), Scipy (Statistical Analysis)
* **Infrastructure:** Databricks / Spark Cluster
* **MLOps: MLflow** (Tracking, Model Registry)
* **Serving & Validation**: FastAPI (Real-time Simulation Logic)

---

## 📈 Key Analysis & Findings

### 1. Data Cleaning & Imputation
* **Challenge:** Encountered 45% missing values in categorical columns.
* **Solution:** Implemented business-logic-driven imputation (Mode for categorical, Median for numerical) and introduced 'Unknown' labels to preserve data patterns.

### 2. Statistical Evidence (A/B Testing)
To verify if ad position truly matters, I performed a **Chi-square Test of Independence**.
* **Observation:** The `Bottom` position yielded the highest CTR (**68.7%**), followed by `Top` (**63.5%**).
  
* **Statistical Result:** $p < 0.05$, confirming that the difference in CTR across positions is statistically significant and not due to random chance.
  
**🔍 Deep Dive: Segment Analysis by Device Type**
  
To gain granular insights, I performed a Chi-square Test of Independence segmented by `device type`. This analysis reveals where ad placement strategies are most effective.

* Significant Segments (Placement Matters):
  * Mobile ($p \approx 0.0000$): The most significant impact. Due to limited screen real estate on mobile devices, the physical position of the ad is a critical determinant of user engagement.
  * Desktop ($p = 0.0255$): A statistically significant difference was found, confirming that optimized placement strategies effectively drive higher CTR on larger screens.
* Non-Significant Segments (Consistent Performance):
  * Tablet ($p = 0.3432$): No significant difference found. Users' clicking behavior remains consistent regardless of the ad's position.
  * Unknown ($p = 0.5815$): No statistical evidence that placement affects CTR for this group.

### 3. Model Selection & Hyperparameter Tuning

I conducted iterative experiments across different algorithms and fine-tuned the best-performing model using **MLflow** to balance predictive power and inference efficiency.

#### Phase 1: Algorithm Comparison

I measured the performance gap between a streamlined feature set (Age Bucketization) and a **Full Feature** approach to determine the optimal input vector.

| Algorithm | Feature Strategy | AUC | Result & Trade-off |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | Top 100 Category Truncation | 0.5559 | **Lightweight**: Fast, but low predictive power. |
| **Random Forest** | Standard Bagging | 0.6043 | **Baseline**: Robust but average performance. |
| **GBT (Age Optimized)** | **Age Bucketization** | 0.7136 | **Efficient**: Good balance, but missed some interactions. |
| **GBT (Full Feature)** | **All Available Features** | **0.7523** | **Selected**: Best capture of multi-dimensional patterns. |

#### Phase 2: Hyperparameter Tuning (MLflow Tracking)
Using the GBT model, I performed a grid search on `max_depth` and `step_size` while also measuring the impact of different feature sets.

| Model Version | Step Size | AUC | Accuracy | Precision | F1-Score | Result |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| GBT_depth5 | 0.1 | 0.6963 | 0.7102 | 0.7102 | 0.6678 | Under-fitted |
| GBT_depth6 | 0.05 | 0.7022 | 0.7190 | 0.7255 | 0.6772 | Slow Convergence |
| GBT_depth7 | 0.05 | 0.7246 | 0.7161 | 0.7126 | 0.6813 | Balanced (Low LR) |
| **GBT_depth7 (Age Optimized)** | **0.1** | **0.7136** | - | - | - | **Feature Test** |
| **GBT_depth7 (Full Feature)** | **0.1** | **0.7523** | **0.7395** | **0.7360** | **0.7171** | **Final Selection** |
| GBT_depth8 | 0.05 | 0.7370 | 0.7254 | 0.7202 | 0.6982 | High Complexity |
| GBT_depth9 | 0.1 | 0.7517 | 0.7322 | 0.7228 | 0.7186 | Performance Plateau |
| GBT_depth8 | 0.1 | 0.7534 | 0.7273 | 0.7185 | 0.7075 | Max AUC |

Final Decision: I finalized the GBT model (Depth: 7, Step: 0.1). It strikes the optimal balance—delivering high accuracy (74%) while maintaining a lightweight structure for low-latency inference.


### 4. Model Interpretation

To understand the decision-making process of the model, I analyzed the **Feature Importance** and derived actionable business insights.

#### 📊 Feature Importance Ranking
#### Phase 1: Initial Analysis (Random Forest)
In the baseline stage, Random Forest was used to identify the primary drivers of click probability.

**Age** (41.2%): Confirmed as the most critical factor.

**Browsing History** (18.0%): Proved to outweigh demographic data.

**Gender** (8.3%): Lowest impact, supporting the hypothesis that behavior is a stronger predictor than gender.

#### Phase 2: Final Model Interpretation (GBT)
Final Model Interpretation (GBT) The GBT model identified **Age** and **Time of Day** as the most critical features, accounting for over 40% of the predictive power.

| Feature | Importance | Description |
| :--- | :---: | :--- |
| **Age** | **23.5%** | Primary driver for user conversion patterns. |
| **Time of Day** | **17.1%** | Key temporal factor for ad engagement. |
| Browsing History | 16.2% | User interest and behavioral intent. |
| Device Type | 15.0% | Technical environment of the user. |
| Gender | 14.3% | Demographic baseline. |
| Ad Position | 14.0% | Physical placement on the page. |


#### Deep Dive 1: Why "Age" is the Top Feature
The data reveals a clear "peak" in engagement for users in their 20s and 30s, who also represent the vast majority of the traffic.

| Age Group | CTR (%) | Total Users | Strategic Insight |
| :--- | :---: | :---: | :--- |
| 10s | 55.0% | 160 | Moderate interest; limited sample size. |
| **20s** | **68.4%** | **1,220** | **Highest CTR**: Most responsive demographic. |
| **30s** | **66.6%** | **6,003** | **Core Segment**: Largest user base with high engagement. |
| 40s | 63.5% | 1,114 | Stable engagement; reliable secondary target. |
| 50s | 60.9% | 1,074 | Gradual decline in ad receptivity. |
| 60s | 51.3% | 429 | Lowest engagement; less reactive to ads. |

#### Deep Dive 2: Impact of "Time of Day"
The model prioritized temporal factors because user responsiveness shifts significantly throughout the day, peaking during the afternoon.
| Time of Day | CTR (%) | Total Users | Strategic Insight |
| :--- | :---: | :---: | :--- |
| **Afternoon** | **68.6%** | **2,016** | **Highest Engagement**: Peak time for ad responsiveness. |
| Morning | 66.5% | 2,126 | Large user volume with stable click-through rates. |
| Unknown | 64.2% | 2,000 | Baseline performance for unidentified time slots. |
| Evening | 63.0% | 1,958 | Moderate receptivity during evening hours. |
| Night | 62.5% | 1,900 | Lowest CTR; users are least active or responsive. |


Key takeaway: While Age is the dominant factor for prediction, the Ad Position is a controllable strategic variable that significantly shifts CTR, especially for the high-traffic Mobile segment.


### 5. Real-time Inference & Smart Ad Placement (FastAPI)

I developed a high-performance serving layer using **FastAPI** that goes beyond simple prediction. The service performs a **real-time simulation** to determine the optimal ad placement for each user, maximizing both engagement and operational efficiency.


#### 🛠️ Smart Decision Engine Logic

* **Multi-Position Simulation**: For every request, the service dynamically simulates three potential ad positions (`Top`, `Side`, `Bottom`). It calculates the click probability for each position based on the user's real-time profile.
* **Profitability Thresholding**: Implemented a **Threshold (0.4)** logic to ensure cost-effectiveness. 
    * If the maximum predicted probability is **lower than 0.4**, the service returns `NO_AD`. 
    * This prevents wasted ad spend and protects the user experience from irrelevant content.
* **Dynamic Serving**: Loads the model via MLflow Model Registry and utilizes a Spark-FastAPI hybrid approach for real-time feature vectorization.

---
### 6. Business Impact & Applications
This system can be directly integrated into digital marketing platforms to achieve:
* Cost Optimization: The 0.4 Probability Threshold acts as a automated filter, reducing ad-spend waste on low-conversion segments.
* Dynamic ROI Maximization: By identifying that Mobile users ($p \approx 0.0000$) and Afternoon/20s groups have higher sensitivity to placement, the system can prioritize high-value slots for these specific users.
* Scalable Decision-making: Replaced manual placement rules with a ML-driven simulation engine that adapts to user behavior in real-time.
