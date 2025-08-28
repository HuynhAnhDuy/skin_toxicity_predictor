# ☣️ CARCINOGENICITY PREDICTOR TOOL – USER GUIDE (Streamlit Version)

## 🧬 Overview

This web-based tool predicts whether a chemical compound is **carcinogenic** based on its SMILES structure using an ensemble of machine learning and deep learning models.

It is powered by a **consensus framework** combining:
- LightGBM using RDKit fingerprints
- Random Forest using EState descriptors
- BiLSTM deep learning model using MACCS fingerprints

The tool is implemented in Python and deployed as an interactive **Streamlit application**.

---

## 🔑 Key Features

- Predicts **probability of carcinogenicity** for compounds based on SMILES input
- Consensus from 3 predictive models ensures robustness
- Supports:
  - ✅ Single SMILES input (via text box)
  - ✅ Batch prediction via `.csv` upload
- Downloadable `.csv` output for batch processing
- Simple, browser-based usage – no installation needed

---

## 🚀 Getting Started

🔗 **Access the web application here**:  
👉 [https://carcinogenicity-predictor.streamlit.app/]  

---

### ✅ Option 1: Single SMILES Prediction

1. Enter a chemical SMILES string (e.g., `NC(=O)CCCCC(N)=O`)
2. Click **"Predict from SMILES"**
3. The output includes:
   - Canonicalized SMILES
   - Probability from each model
   - Average probability
   - Final predicted label: **Carcinogen** or **Non-Carcinogen**

---

### ✅ Option 2: Batch Prediction via CSV Upload

1. Prepare a `.csv` file with a column named `SMILES`
2. Upload the file in the **CSV Upload** tab
3. Click **"Run batch prediction"**
4. A result table will be shown and available for **CSV download**

---

## 📤 Output Example

### 🧪 Single Prediction Result

| Model                     | Probability |
|--------------------------|-------------|
| LightGBM (RDKit)         | 0.7864      |
| Random Forest (EState)   | 0.7421      |
| BiLSTM (MACCS)           | 0.8123      |
| **Average Probability**  | 0.7803      |
| **Prediction**           | ☣️ Carcinogen |

---

### 📄 Batch CSV Output Example

| Input_SMILES        | Canonical_SMILES     | Prob_LightGBM | Prob_RF | Prob_BiLSTM | Average_Probability | Prediction       |
|---------------------|----------------------|----------------|---------|--------------|----------------------|------------------|
| CCOC(=O)c1ccccc1     | CCOC(=O)c1ccccc1      | 0.43           | 0.39    | 0.46         | 0.427                | Non-Carcinogen   |
| NC(=O)CCCCC(N)=O     | NC(=O)CCCCC(N)=O      | 0.81           | 0.77    | 0.85         | 0.810                | Carcinogen       |

---

## 🛠️ Technology Stack

- Python 3
- RDKit (SMILES parsing, fingerprint generation)
- Scikit-learn (Random Forest)
- LightGBM
- TensorFlow / Keras (BiLSTM)
- Streamlit (web deployment)

---

## 👨‍🔬 Authors & Acknowledgment

- **Huynh Anh Duy** (1,2)  
  - Can Tho University, Vietnam  
  - PhD Candidate, Khon Kaen University, Thailand  
  - *Cheminformatics, QSAR Modeling, Computational Drug Discovery and Toxicity Prediction*

- **Asst. Prof. Dr. Tarapong Srisongkram** (2)  
  - Faculty of Pharmaceutical Sciences  
  - Khon Kaen University, Thailand  
  - *Cheminformatics, QSAR Modeling, Computational Drug Discovery and Toxicity Prediction*

📬 Contact:
- 📧 huynhanhduy.h@kkumail.com | haduy@ctu.edu.vn  
- 📧 tarasri@kku.ac.th

---

## 📌 Version Info

- **Version**: 1.0  
- **Last updated**: June 2025  
- **License**: Academic / Research use only

---

**Thank you for using Carcinogenicity Predictor Tool!**
