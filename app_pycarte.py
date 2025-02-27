import streamlit as st
import pandas as pd
import os
import plotly.express as px
from ydata_profiling import ProfileReport  
import streamlit.components.v1 as components
from pycaret.classification import setup as cls_setup, compare_models as cls_compare, pull as cls_pull, save_model as cls_save
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, pull as reg_pull, save_model as reg_save
from pycaret.clustering import setup as clu_setup, create_model, assign_model, pull as clu_pull, save_model as clu_save

# ====== Page Configuration ======
st.set_page_config(page_title="AutoStreamML", layout="wide", page_icon="🤖")


# ====== Custom CSS for Colors ======

custom_css = """
<style>
/* Full dark background with a visible gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0f1e, #252b3a, #3a2c50) !important; 
}

/* Sidebar with a glowing effect */
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #161d2f, #1e2743) !important; 
    box-shadow: 0px 0px 10px rgba(50, 80, 150, 0.4) !important;
}

/* Remove header and toolbar background */
[data-testid="stHeader"], [data-testid="stToolbar"] {
    background: transparent !important;
}

/* Improve text readability */
h1, h2, h3, h4, h5, h6, p, span, label {
    color: #d6d9e0 !important;  /* Softer white for less eye strain */
}

/* Modern dark container styling */
[data-testid="stBlock"] {
    background: rgba(40, 44, 60, 0.85) !important; /* Dark purple-gray with slight transparency */
    border-radius: 12px !important;
    padding: 15px !important;
    box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.3);
}

/* Metallic dark blue button styling */
button, .stButton > button {
    background: linear-gradient(145deg, #1e2a3f, #2e3d5f) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    font-size: 16px !important;
    font-weight: bold !important;
    border: 1px solid #3e5c9a !important;
}

/* Button hover effect */
.stButton > button:hover {
    background: linear-gradient(145deg, #293850, #405282) !important;
    box-shadow: 0px 0px 12px rgba(74, 120, 200, 0.6) !important;
}

/* Tables and DataFrames */
[data-testid="stDataFrame"] {
    border-radius: 8px !important;
    overflow: hidden !important;
    background: rgba(50, 54, 70, 0.9) !important; /* Dark bluish-gray */
    color: #f1f1f1 !important; /* Softer light text */
}

/* Input fields with a modern dark touch */
.stTextInput, .stSelectbox, .stNumberInput {
    background-color: #252b3a !important;
    color: white !important;
    border-radius: 8px !important;
    border: 1px solid #4b5f89 !important;
}

/* Subtle glowing effect for inputs on focus */
.stTextInput:focus, .stSelectbox:focus, .stNumberInput:focus {
    box-shadow: 0px 0px 10px rgba(80, 140, 255, 0.5) !important;
}

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)



# ====== Sidebar Navigation ======
with st.sidebar:
    st.image("https://d112y698adiu2z.cloudfront.net/photos/production/software_photos/002/225/744/datas/gallery.jpg", use_container_width=True)
    st.title("🚀 AutoStreamML")
    choice = st.radio("📌 Select a Task",["Upload", "Profiling", "ML", "Download"])
    st.info("🔹 This app allows you to build an automated ML pipeline easily!")

# ====== Load Data If Exists ======
if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv")

# ====== Upload Data Page ======
if choice == "Upload":
    st.title("📂 Upload Your Dataset")
    file = st.file_uploader("📌 **Upload a CSV file**", type=["csv"])

    if file:
        df = pd.read_csv(file)
        df.to_csv("sourcedata.csv", index=False)
        st.success("✅ **File uploaded successfully!**")
        st.dataframe(df)

# ====== Data Profiling Page ======
if choice == "Profiling":
    st.title("📊 Automated Exploratory Data Analysis")

    if 'df' in globals():
        with st.spinner("⏳ Generating report..."):
            profile = ProfileReport(df, explorative=True)
            report_html = profile.to_html()
            components.html(report_html, height=1200, scrolling=True)
    else:
        st.warning("⚠️ **Please upload a dataset first!**")

# ====== Machine Learning Page ======
if choice == "ML":
    st.title("🤖 Machine Learning Pipeline")

    if 'df' in globals():
        col1, col2 = st.columns(2)  # Split layout into two columns

        with col1:
            ml_type = st.selectbox("⚙️ **Select ML Task**", ["Classification", "Regression", "Clustering"])

        if ml_type in ["Classification", "Regression"]:
            with col2:
                target = st.selectbox("🎯 **Select Target Column**", df.columns)

        if st.button("🚀 Train Model"):
            with st.spinner("🔄 Training in progress..."):
                if ml_type == "Classification":
                    df = df.dropna(subset=[target])
                    cls_setup(df, target=target, verbose=False)
                    setup_df = cls_pull()
                    st.info("📌 **Experiment Settings**")
                    st.dataframe(setup_df)

                    best_model = cls_compare()
                    compare_df = cls_pull()
                    st.info("🏆 **Model Comparison Results**")
                    st.dataframe(compare_df)

                    st.success("🔥 **Best Classification Model:**")
                    st.write(best_model)
                    cls_save(best_model, "best_model")

                elif ml_type == "Regression":
                    df = df.dropna(subset=[target])
                    reg_setup(df, target=target, verbose=False)
                    setup_df = reg_pull()
                    st.info("📌 **Experiment Settings**")
                    st.dataframe(setup_df)

                    best_model = reg_compare()
                    compare_df = reg_pull()
                    st.info("🏆 **Model Comparison Results**")
                    st.dataframe(compare_df)

                    st.success("🔥 **Best Regression Model:**")
                    st.write(best_model)
                    reg_save(best_model, "best_model")

                elif ml_type == "Clustering":
                    clu_setup(df, verbose=False)
                    setup_df = clu_pull()
                    st.info("📌 **Experiment Settings**")
                    st.dataframe(setup_df)

                    best_model = create_model("kmeans")  # Using K-Means by default
                    clustered_df = assign_model(best_model)
                    st.info("📊 **Clustered Data**")
                    st.dataframe(clustered_df)

                    # 🔹 Interactive Scatter Plot for Clusters
                    fig = px.scatter(clustered_df, x=clustered_df.columns[0], y=clustered_df.columns[1], color='Cluster')
                    st.plotly_chart(fig)

                    st.success("🔥 **Best Clustering Model:**")
                    st.write(best_model)
                    clu_save(best_model, "best_model")

                st.success("✅ **Model saved successfully!**")
    else:
        st.warning("⚠️ **Please upload a dataset first!**")

# ====== Download Trained Model Page ======
if choice == "Download":
    st.title("📥 Download Trained Model")

    if os.path.exists("best_model.pkl"):
        with open("best_model.pkl", "rb") as f:
            st.download_button("⬇️ Download Model", f, "best_model.pkl")
    else:
        st.warning("⚠️ **No trained model found. Please train a model first!**")
