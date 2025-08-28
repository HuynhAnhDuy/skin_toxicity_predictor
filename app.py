import os
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from tokenizer import CustomTokenizer
import preprocess as pp
from PIL import Image
import json
from functools import reduce

st.set_page_config(
    page_title="Skin Toxicity Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Cache heavy objects
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_model(path: str):
    return load_model(path)

@st.cache_resource(show_spinner=False)
def get_tokenizer():
    return CustomTokenizer()

# ---------------------------
# Model paths
# ---------------------------
ENDPOINT2MODEL = {
    "corrosion": "models/corrosion_model.keras",
    "irritation": "models/irritation_model.keras",
    "sensitization": "models/sensitization_model.keras",
}

# ---------------------------
# Preprocessing
# ---------------------------
def pipeline_clean_df(df: pd.DataFrame, smiles_col: str = "SMILES") -> pd.DataFrame:
    df = pp.remove_missing_data(df, smiles_col, smiles_col)
    df = pp.canonical_smiles(df, smiles_col)
    df = pp.remove_inorganic(df, 'canonical_smiles')
    df = pp.remove_mixtures(df, 'canonical_smiles')
    df = pp.process_duplicate(df, 'canonical_smiles', remove_duplicate=True)
    return df

# ---------------------------
# Prediction logic
# ---------------------------
def predict_rows(df: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    model_path = ENDPOINT2MODEL[endpoint]
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return pd.DataFrame()

    model = get_model(model_path)
    rows = []

    if endpoint in ("corrosion", "irritation"):
        for _, r in df.iterrows():
            vec = pp.get_mac_phys_feature(r['canonical_smiles'])
            if vec is None:
                rows.append({"SMILES": r["SMILES"], "canonical_smiles": r['canonical_smiles'], "prob": None, "label": None})
                continue
            X = np.array(vec, dtype=float).reshape(1, 1, -1)
            prob = float(model.predict(X, verbose=0).ravel()[0])
            label = int(prob >= 0.5)
            rows.append({"SMILES": r["SMILES"], "canonical_smiles": r['canonical_smiles'], "prob": prob, "label": label})

    elif endpoint == "sensitization":
        tok = get_tokenizer()
        for _, r in df.iterrows():
            vec = pp.get_token_fp_feature(r['canonical_smiles'], tok)
            if vec is None:
                rows.append({"SMILES": r["SMILES"], "canonical_smiles": r['canonical_smiles'], "prob": None, "label": None})
                continue
            X = np.array(vec, dtype=float).reshape(1, 1, -1)
            prob = float(model.predict(X, verbose=0).ravel()[0])
            label = int(prob >= 0.5)
            rows.append({"SMILES": r["SMILES"], "canonical_smiles": r['canonical_smiles'], "prob": prob, "label": label})
    else:
        st.error("Unknown endpoint")
        return pd.DataFrame()

    return pd.DataFrame(rows)

def predict_all_endpoints(df: pd.DataFrame) -> pd.DataFrame:
    results = []

    for endpoint in ["corrosion", "irritation", "sensitization"]:
        res = predict_rows(df, endpoint)
        if not res.empty:
            res = res[["SMILES", "canonical_smiles", "prob", "label"]].copy()
            res.columns = ["SMILES", "canonical_smiles", f"{endpoint}_prob", f"{endpoint}_label"]
            results.append(res)

    if results:
        merged = reduce(lambda left, right: pd.merge(left, right, on=["SMILES", "canonical_smiles"], how="outer"), results)

        def interpret(row):
            cor = "❌ Corrosive" if row["corrosion_label"] == 1 else "✅ Non-corrosive"
            irr = "❌ Irritant" if row["irritation_label"] == 1 else "✅ Non-irritant"
            sen = "❌ Sensitizer" if row["sensitization_label"] == 1 else "✅ Non-sensitizer"
            return f"{cor}, {irr}, {sen}"

        merged["Summary"] = merged.apply(interpret, axis=1)
        return merged

    return pd.DataFrame()

def download_csv_button(df: pd.DataFrame, fname: str, label: str = "Download CSV"):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv_bytes, file_name=fname, mime="text/csv")

# ---------------------------
# UI
# ---------------------------
st.title("☣️ DeepSkinTox: Skin Toxicity Predictor")
st.markdown(
    """
    <div style='text-align:left; color:red; font-size:22px; font-weight:bold;'>
        Multi-endpoint skin-toxicity prediction with BiLSTM/LSTM and conjoint features.
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("""        
**Endpoints and Models:**

● Skin Corrosion — predicted using a BiLSTM model trained on combined MACCS fingerprints and physicochemical descriptors.

● Skin Irritation — modeled with an LSTM architecture incorporating MACCS fingerprints and physicochemical features.

● Skin Sensitization — evaluated using a BiLSTM network with tokenized molecular representations and RDKit descriptors. 
             
**Developers:** Huynh Anh Duy<sup>1,2</sup>, Tarapong Srisongkram<sup>2</sup>  
**Affiliations:** <sup>1</sup>Can Tho University, Vietnam; <sup>2</sup>Khon Kaen University, Thailand
            """, unsafe_allow_html=True)
# Input mode sau Affiliations

st.markdown("**Input mode**")
mode = st.radio("", ["Single SMILES", "Batch CSV"], index=0, horizontal=True)

with st.sidebar:
    st.header("🧾 Instructions")
    st.markdown("""
    1. Paste a SMILES string or upload a CSV file.
    2. Click **Predict** or **Run batch prediction**.
    3. Results for all 3 endpoints will be shown.
    """)

    st.markdown("🔍 **Prediction rule:**")
    st.markdown("""
    - **Probability >= 0.5** → **Toxic** 
    - **Probability < 0.5** → **Non-toxic**
    """)

    st.markdown("🧪 **Label meanings:**")
    st.markdown("""
    - ❌ Corrosive / Irritant / Sensitizer  
    - ✅ Non-corrosive / Non-irritant / Non-sensitizer  
    """)

if mode == "Single SMILES":
    smi = st.text_input("Enter a SMILES", value="CC(=O)Oc1ccccc1C(=O)O")
    if st.button("Predict", type="primary"):
        if not smi.strip():
            st.warning("Please enter a SMILES string.")
        else:
            df = pd.DataFrame({"SMILES": [smi]})
            df = pipeline_clean_df(df)
            if df.empty:
                st.error("SMILES was removed by preprocessing.")
            else:
                res = predict_all_endpoints(df)
                if not res.empty:
                    st.success("✓ Prediction completed for all 3 endpoints")
                    st.dataframe(res, use_container_width=True)
                    download_csv_button(res, "single_prediction_all_endpoints.csv", label="Download result CSV")

                    for _, row in res.iterrows():
                        st.markdown(f"**{row['SMILES']}** → {row['Summary']}")

else:
    up = st.file_uploader("Upload CSV with a 'SMILES' column", type=["csv"])
    if st.button("Run batch prediction", type="primary"):
        if up is None:
            st.warning("Please upload a CSV file first.")
        else:
            try:
                df_in = pd.read_csv(up)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
                st.stop()
            if 'SMILES' not in df_in.columns:
                st.error("CSV must contain a 'SMILES' column.")
                st.stop()
            df_clean = pipeline_clean_df(df_in.copy())
            if df_clean.empty:
                st.error("All rows were filtered out by preprocessing.")
                st.stop()
            res = predict_all_endpoints(df_clean)
            if not res.empty:
                st.success(f"✓ Predicted {len(res)} rows for all endpoints")
                st.dataframe(res, use_container_width=True)
                download_csv_button(res, "batch_predictions_all_endpoints.csv", label="Download results CSV")

                for _, row in res.iterrows():
                    st.markdown(f"**{row['SMILES']}** → {row['Summary']}")
# === Author Section ===
st.markdown("---")
st.subheader("👨‍🔬 About the Authors")

col1, col2 = st.columns(2)

with col1:
    image1 = Image.open("assets/duy.jpg")
    st.image(image1, caption="Huynh Anh Duy", width=160)
    st.markdown("""
    **Huynh Anh Duy**  
    Can Tho University, Vietnam  
    PhD Candidate, Khon Kaen University, Thailand  
    *Cheminformatics, QSAR Modeling, Computational Drug Discovery and Toxicity Prediction*  
    📧 [huynhanhduy.h@kkumail.com](mailto:huynhanhduy.h@kkumail.com), [haduy@ctu.edu.vn](mailto:haduy@ctu.edu.vn)
    """)

with col2:
    image2 = Image.open("assets/tarasi.png")
    st.image(image2, caption="Tarapong Srisongkram", width=160)
    st.markdown("""
    **Asst Prof. Dr. Tarapong Srisongkram**  
    Faculty of Pharmaceutical Sciences  
    Khon Kaen University, Thailand  
    *Cheminformatics, QSAR Modeling, Computational Drug Discovery and Toxicity Prediction*  
    📧 [tarasri@kku.ac.th](mailto:tarasri@kku.ac.th)
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; font-size:14px; color:gray; line-height:1.6;'>"
    "⚠️ <b>Disclaimer:</b> This platform is intended for <i>research purposes only</i>. "
    "The information provided here does not substitute for professional medical advice, "
    "diagnosis, or treatment. <br><br>"
    "🧪 <b>Skin toxicity data and analyses</b> are experimental and should be interpreted "
    "with caution. Use of this tool is restricted to qualified research personnel. <br><br>"
    "📄 <b>Version:</b> 1.0.0 &nbsp; | &nbsp; <b>Created on:</b> August 28, 2025 <br>"
    "© 2025 QSAR Lab &nbsp; | &nbsp; "
    "</div>",
    unsafe_allow_html=True
)
