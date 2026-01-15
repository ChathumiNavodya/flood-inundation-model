# 🌊 Flood Impact Prediction & Relief Priority Decision Support System (Sri Lanka)

An end-to-end **machine learning–based flood disaster decision support system** designed for **Sri Lanka**, combining flood occurrence prediction, inundation impact estimation, and relief priority ranking with an **interactive web dashboard**.

This project goes beyond simple flood prediction by supporting **real-world disaster response planning** through explainable models and actionable location/district rankings.

---

## 📌 Project Overview

Floods are one of the most frequent natural disasters in Sri Lanka, affecting both urban and rural communities.  
This system helps decision-makers by:

- Predicting the **probability of flood occurrence**
- Estimating **expected inundation area (m²)**
- Ranking **locations and districts by relief priority**
- Providing a dashboard to **upload data, view maps, and download outputs**

---

## 🎯 Key Objectives

- Build accurate flood prediction models using realistic features  
- Quantify flood impact using inundation area prediction  
- Prioritize relief allocation using a multi-criteria scoring framework  
- Provide **model transparency** using explainability  
- Deliver results through an **interactive web dashboard (Streamlit)**

---

## 🧠 Methodology

### 1️⃣ Flood Occurrence Prediction (Classification)
- **Target:** `flood_occurrence_current_event` (Yes/No → 1/0)
- **Models Compared:**
  - Logistic Regression (Best)
  - Random Forest Classifier
- **Metrics Used:**
  - Accuracy, Precision, Recall, F1-score, ROC–AUC  
- **Outputs:**
  - Confusion Matrix
  - ROC Curve

---

### 2️⃣ Inundation Area Prediction (Regression)
- **Target:** `inundation_area_sqm`
- **Models Compared:**
  - Linear Regression (Best)
  - Random Forest Regressor
- **Technique:** Log-transform target using `log1p()` to stabilize large value ranges
- **Output:**
  - Actual vs Predicted scatter plot

---

### 3️⃣ Relief Priority Recommendation
A **relief priority score** is calculated using:

- Flood probability (classification output)
- Predicted inundation severity (regression output)
- Population density
- Infrastructure weakness
- Hospital and evacuation accessibility

Locations are categorized into:
- **HIGH**
- **MEDIUM**
- **LOW** priority

Outputs include:
- Top 20 priority locations
- District-wise summary ranking
- Score-based and category-based maps

---

### 4️⃣ Explainability
The project includes **model explainability** by extracting:

- Logistic Regression coefficients  
- Top contributing features to flood occurrence prediction

This helps explain:  
✅ *why a location is flagged as high-risk* and improves trust in decision-making.

---

## 📊 Dataset

- **Records:** 25,000  
- **Attributes:** 32  
- Includes:
  - Environmental (rainfall, elevation, vegetation indices)
  - Geographic (lat, lon, distance to river)
  - Infrastructure (road quality, utilities, evacuation access)
  - Vulnerability indicators (population density, hospital distance)

---

## 🖥️ Interactive Dashboard (Streamlit)

### Features
- Upload CSV for new locations
- Adjustable flood probability threshold (controls inundation prediction)
- Flood probability prediction
- Inundation area estimation
- Relief priority ranking table
- District urgency summary
- Maps:
  - Relief Priority Score Map
  - District Priority Category Map
- Download predicted results as CSV

---

## 🛠️ Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib  
- Streamlit  
- Joblib  

