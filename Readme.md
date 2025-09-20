🤖 DataES AI: The Intelligent Data Quality & Analysis Platform

![alt text](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)
<!-- Replace with your deployment link -->

DataES AI is a comprehensive, AI-powered Streamlit application designed to automate and enhance the entire data quality and exploratory data analysis (EDA) workflow. It transforms the tedious, manual process of data validation into an interactive, intelligent, and insightful experience.

The platform provides an end-to-end solution that helps data professionals ensure data integrity, discover hidden patterns, identify anomalies, and understand data lineage with unprecedented ease and transparency.

✨ Core Features

This platform is built around two primary, interconnected modules:

🏠 1. The AI Data Quality Assistant

The first step in any data project is ensuring the quality of your data. This module acts as a proactive "Data Guardian".

🧠 Intelligent Onboarding: Upload any CSV, and the app instantly performs a deep scan.

🎯 Precise Diagnostics: Instead of generic advice, it provides evidence-based findings, using statistical tests to identify specific issues like:

Constant Value Columns (Zero Variance)

Statistically Significant Outliers (using IQR)

Inconsistent Casing & Whitespace

High Cardinality vs. Unique Identifiers

🗣️ RAG-Powered AI Analyst: Upload an optional data dictionary to provide business context. The built-in AI Analyst, powered by a local LLM (via LM Studio), uses this context (Retrieval-Augmented Generation) to give you hyper-specific, domain-aware remediation advice.

📈 2. The Advanced Analysis Dashboard

Once your data is clean, this multi-tab dashboard allows you to move seamlessly from validation to insight generation.

📊 Comprehensive EDA: A full suite of interactive visualizations:

Target Analysis: Dynamically select and visualize the distribution of your target variable.

Correlation Heatmaps: Instantly spot relationships between features.

Feature vs. Target Plots: Analyze how feature distributions differ across outcomes.

🔬 Anomaly Detection: Use the powerful Isolation Forest algorithm to identify multivariate outliers that might represent errors or fraud.

AI Root Cause Analysis: Select any detected anomaly and ask the AI to hypothesize its potential root cause.

🔗 End-to-End Data Lineage: A two-level visualization engine for ultimate transparency:

Source-to-Table View: Upload multiple related CSVs to visualize the entire data pipeline and join relationships.

Feature Recipe View: Select an engineered feature to see its exact formula and source columns.

🤖 AI Interpretation Everywhere: "Ask AI" buttons are integrated throughout the dashboard, allowing you to get instant, business-focused interpretations of any chart or finding.

🚀 How It Works: The Technology
<p align="center">
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn"/>
<img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly"/>
</p>


Frontend: Built entirely with Streamlit for a fast, interactive user interface.

Data Manipulation: Powered by Pandas for efficient data processing.

Machine Learning:

Anomaly Detection: Uses IsolationForest from Scikit-learn.

Statistical Analysis: Leverages NumPy and Pandas for diagnostics.

Visualizations: Interactive charts are created with Plotly Express and Graphviz for lineage graphs.

AI/LLM Integration: Communicates with any local Large Language Model served via LM Studio, which mimics the OpenAI API.

🛠️ Getting Started

Follow these steps to run the DataES AI platform on your local machine.

1. Prerequisites

Python 3.8+

LM Studio: Download and install from lmstudio.ai.

Graphviz: (For Lineage Graphs)

Windows: Download and install from the official website and add it to your system's PATH.

macOS (Homebrew): brew install graphviz

Linux (APT): sudo apt-get install graphviz

2. Installation

Clone the repository and install the required Python packages:

code
Bash
download
content_copy
expand_less

# Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Install dependencies
pip install -r requirements.txt

(You'll need to create a requirements.txt file containing streamlit, pandas, numpy, scikit-learn, plotly, openai, graphviz)

3. Start the Local LLM Server

Open LM Studio.

Download your preferred model (e.g., Mistral, Llama 3).

Go to the Local Server tab (<->).

Click Start Server.

4. Run the Streamlit Application

Open your terminal in the project's root directory and run the following command. The --server.maxUploadSize flag is important for handling large datasets.

code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
streamlit run 🏠_Data_Quality_Analysis.py --server.maxUploadSize 1024

Your web browser will automatically open with the running application! 🎉

This project successfully implements the core requirements of the challenge:

✅ Flagging of Data Quality Issues: Implemented with a precise, evidence-based diagnostic engine.

✅ AI-Driven Suggestions: The entire platform is built around contextual AI analysis and remediation advice.

✅ Anomaly Detection: A dedicated module using IsolationForest with AI-powered root cause analysis.

✅ Root Cause Analysis: Integrated directly into the Anomaly Detection workflow.

✅ Data Lineage: A stunning two-level visualization of both inter-file and intra-file data transformations.

✅ Platform Independence: Designed to work with any CSV file, with intelligent, conditional logic for domain-specific enhancements.