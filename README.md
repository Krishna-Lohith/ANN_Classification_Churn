<h1 align="center">🏦 Customer Churn Prediction</h1>

<p align="center">
  <b>Predict whether a bank customer will leave - Trained by a powerful Artificial Neural Network</b>
</p>

<p align="center">
  <a href="https://annclassificationchurn-srgf6gbtvlvj3aipqmz8r9.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀 Live Demo-Streamlit App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/>
</p>

---

## 📌 Problem Statement

Banks lose customers silently. By the time a customer leaves, it's too late to act.
This project builds an ANN model that predicts **which customers are likely to churn**, so the bank can act early and retain them.

---

## 📊 Dataset

| Detail | Info |
|--------|------|
| Source | [Kaggle — Churn Modelling Dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) |
| Rows | 10,000 customers |
| Features | 11 (Age, Balance, Geography, Gender, etc.) |
| Target | `Exited` - 1 = Churned, 0 = Stayed |
| Churn Rate | 20.37% |

---

## ⚙️ How It Works

```
Raw Data → Drop irrelevant columns → Encode categorical features
         → Scale numeric features → Train ANN → Predict churn probability
```

**Preprocessing:**
- `Gender` → Label Encoding (Female = 0, Male = 1)
- `Geography` → One-Hot Encoding (France / Germany / Spain)
- All features → StandardScaler (same scale for the ANN)
- Split: 80% Train / 20% Test

**Model Architecture:**

```
Input (12 features)
    ↓
Dense — 64 neurons — ReLU
    ↓
Dense — 32 neurons — ReLU
    ↓
Output — 1 neuron — Sigmoid  →  Churn Probability
```

- Optimizer: Adam | Loss: Binary Crossentropy
- Early Stopping (patience = 25) | TensorBoard for monitoring
- **Validation Accuracy: 86.35%**

---

## 🧠 Two Models — What's the Difference?

| File | Description |
|------|-------------|
| `model.h5` | Standard ANN trained with fixed architecture. This is the model used in the **Streamlit web app**. |
| `HyperParameterTuning_ANN_best_model.h5` | ANN trained using **hyperparameter tuning** — the number of layers, neurons per layer, and epochs are searched automatically to find the best combination. Produces a better model but uses significantly more CPU/GPU resources. Not used in the app for performance reasons. |

> **In short:** `model.h5` = fast and deployable. `HyperParameterTuning_ANN_best_model.h5` = more optimized but heavier.

---

## 🗂️ Project Files

```
📁 ANN_Classification_Churn
├── Churn_Modelling.csv               ← Dataset
├── Churn_Training.ipynb              ← Data preprocessing + ANN training
├── Churn_Prediction.ipynb            ← Single customer prediction
├── HyperParameterTuning_ANN.ipynb    ← Hyperparameter tuning notebook
├── app.py                            ← Streamlit web application
├── model.h5                          ← Trained ANN model (used in app)
├── HyperParameterTuning_ANN_best_model.h5  ← Best model from tuning
├── label_encoder_gender.pkl          ← Encoder for Gender
├── onehot_encoder_geo.pkl            ← Encoder for Geography
├── scaler.pkl                        ← StandardScaler
├── requirements.txt                  ← Python dependencies
└── runtime.txt                       ← Python runtime version
```

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Krishna-Lohith/ANN_Classification_Churn.git
cd ANN_Classification_Churn

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

---

## 🌐 Live App

👉 **[Click here to try the Churn Predictor](https://annclassificationchurn-srgf6gbtvlvj3aipqmz8r9.streamlit.app/)**

Enter a customer's details — the app instantly returns the churn probability and a clear prediction.
