# 👕 STYLEPREDICT — Mini AI for Sustainable Fashion

**STYLEPREDICT** is a small Machine Learning project that simulates a recommendation system for sustainable fashion.  
The goal is to predict how much a clothing item will appeal to a user, combining personal style, eco-consciousness, and budget.

---

## 🚀 Features

- Generate a synthetic dataset (300 user-item ratings)
- Exploratory Data Analysis (EDA) with charts
- Train a regression model (**Random Forest**)
- Evaluate model performance (MAE, R²)
- Final simulator to predict the liking of new items

---

## 🧠 How it works

The system simulates users and clothing items with these characteristics:

### User
- `casual_user` → style preference (0 = formal, 1 = casual)
- `eco_user` → sensitivity to sustainability
- `budget` → available budget (€)

### Item
- `casual_item` → item style
- `sustainability` → eco score (30–100)
- `price` → price (€)

### Target
- `liking` (1–10), calculated based on:
  - style alignment between user and item
  - sustainability
  - budget compatibility
  - random noise

---

## 📊 Output

### 1. EDA Chart
Shows average liking according to style alignment:

**Generated file:** `stylepredict_chart.png`

### 2. Model Performance

- **MAE (Mean Absolute Error)** → average prediction error
- **R² Score** → percentage of variance explained

### 3. Results Chart

Comparison of actual vs predicted ratings:

**Generated file:** `stylepredict_results.png`

---

## 🤖 Model

- Algorithm: `RandomForestRegressor`
- Library: scikit-learn
- Key parameters:
  - `n_estimators = 50`
  - `random_state = 42`

---

## 🧪 Final Simulator

The model can predict the liking of new items.  
Example output:

| Item               | Predicted Liking |
|-------------------|----------------|
| Organic Linen Jacket | 8.7 / 10      |
| Hemp Sneakers        | 6.5 / 10      |
| Tencel Dress         | 9.1 / 10      |

---

## 📦 Requirements

Install dependencies with:

```bash
pip install numpy pandas matplotlib scikit-learn

---
By Uhmaizil Rahim, Nida Ajmal, Beatrice Maria Lo Muto.
---
