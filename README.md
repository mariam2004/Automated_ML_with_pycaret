# AutoStreamML

## 🚀 Overview  
AutoStreamML is a **no-code** automated machine learning (AutoML) tool built using **Streamlit** and **PyCaret**. It enables users to **upload datasets, perform exploratory data analysis (EDA), train machine learning models, and download the best model**—all through an intuitive user interface.

## 🔹 Features  
- 📂 **Upload & Explore Data**: Easily upload CSV files and preview datasets.  
- 📊 **Automated Data Profiling**: Generate an interactive EDA report using `ydata-profiling`.  
- 🤖 **Train & Compare ML Models**: Supports **Classification, Regression, and Clustering** with PyCaret.  
- 📥 **Download the Best Model**: Save the trained model for future use.  
- 🎨 **Dark Mode UI**: Modern, sleek, and user-friendly interface.

## 📌 How It Works  
1. **Upload**: Load your dataset in CSV format.  
2. **Profiling**: Generate an automated data analysis report.  
3. **ML Training**: Select ML task (Classification, Regression, or Clustering) and train models.  
4. **Model Comparison**: View the best-performing model based on evaluation metrics.  
5. **Download Model**: Export the best model as a `.pkl` file.

## 🛠 Installation & Usage  
### 🔧 Prerequisites  
- Python 3.8+  
- Required libraries: `streamlit`, `pandas`, `pycaret`, `ydata-profiling`, `plotly`

### 🚀 Setup & Run  
Clone the repository and install dependencies:  
```bash  
git clone https://github.com/yourusername/AutoStreamML.git  
cd AutoStreamML  
pip install -r requirements.txt  
streamlit run app.py  
```

