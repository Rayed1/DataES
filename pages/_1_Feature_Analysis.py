import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from openai import OpenAI
from sklearn.ensemble import IsolationForest
import graphviz
import io 

# --- Configuration for LM Studio ---
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key="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")


# --- Page Setup ---
st.set_page_config(layout="wide", page_title="Advanced Data Analysis")
st.title("📈 Advanced Data Analysis Dashboard")
st.write(
    "Perform a full EDA, find anomalies, trace feature lineage, and use AI to interpret the results."
)

# --- Check for uploaded data ---
if 'df' not in st.session_state:
    st.warning("Please upload a dataset on the '🏠 Data Quality Analysis' page first.")
    st.stop()

# --- Caching Expensive Functions ---
@st.cache_data
def create_features(df):
    """
    Creates new features for the Home Credit dataset if the required columns are present.
    If not, it returns the original dataframe without changes, preventing crashes.
    """
    df_featured = df.copy()
    
    REQUIRED_COLUMNS = [
        'AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY', 
        'DAYS_EMPLOYED', 'DAYS_BIRTH'
    ]
    
    if all(col in df_featured.columns for col in REQUIRED_COLUMNS):
        # If they exist, perform the domain-specific feature engineering
        df_featured['CREDIT_INCOME_PERCENT'] = df_featured['AMT_CREDIT'] / df_featured['AMT_INCOME_TOTAL']
        df_featured['ANNUITY_INCOME_PERCENT'] = df_featured['AMT_ANNUITY'] / df_featured['AMT_INCOME_TOTAL']
        df_featured['CREDIT_TERM'] = df_featured['AMT_ANNUITY'] / df_featured['AMT_CREDIT']
        
        # Handle the special value in DAYS_EMPLOYED
        if 365243 in df_featured['DAYS_EMPLOYED'].values:
            df_featured['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
            
        df_featured['DAYS_EMPLOYED_PERCENT'] = df_featured['DAYS_EMPLOYED'] / df_featured['DAYS_BIRTH']
        
        # Return the dataframe with the new features
        return df_featured
    else:
        return df 

@st.cache_data
def calculate_correlation(_df):
    return _df.select_dtypes(include=np.number).corr()

# --- AI Interpretation Function ---
def get_llm_interpretation(analysis_type, data_summary, question):
    system_prompt = (
        "You are a senior data scientist. Your role is to interpret data analysis results for a business audience. "
        "Avoid overly technical jargon and focus on the business implications, potential risks, and actionable insights."
    )
    user_prompt = (
        f"Analysis Context: {analysis_type}\n\n"
        f"Data Summary:\n{data_summary}\n\n"
        f"My Question:\n{question}\n\n"
        "Please provide your expert interpretation:"
    )
    try:
        completion = client.chat.completions.create(
            model=st.session_state.get('selected_model', "mistralai/mistral-7b-instruct"), 
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.7,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error connecting to LM Studio: {e}"

# --- Main App UI ---
df_original = st.session_state['df']
df_featured = create_features(df_original)
st.session_state['df_featured'] = df_featured

# --- Dashboard ---
tab_list = [
    "🎯 Target Variable", 
    "🧮 Correlations", 
    "📊 Feature vs. Target", 
    "📈 Distributions",
    "🔬 Anomaly Detection",
    "🔗 Feature Lineage"
]
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_list)

with tab1:
    st.header("Analysis of the Target Variable")
    st.write("Select a column to analyze its distribution. This is typically a binary outcome or a categorical feature with a small number of unique values.")

    all_cols = df_featured.columns.tolist()
    
    default_index = all_cols.index('TARGET') if 'TARGET' in all_cols else len(all_cols) - 1

    # --- DYNAMIC WIDGET ---
    target_col = st.selectbox(
        "Select the Target Variable Column:",
        options=all_cols,
        index=default_index
    )

    if target_col:
        target_counts = df_featured[target_col].value_counts()
        num_unique_values = len(target_counts)

        # --- VISUALIZATION ---
        if num_unique_values < 15:
            st.subheader(f"Pie Chart for '{target_col}'")
            fig = px.pie(
                values=target_counts.values, 
                names=target_counts.index,
                title=f'Distribution of Outcomes for "{target_col}"'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"The column '{target_col}' has **{num_unique_values}** unique values. The most frequent value is '**{target_counts.index[0]}**', appearing in **{target_counts.iloc[0]/len(df_featured):.2%}** of the rows.")

        else:
            st.subheader(f"Bar Chart for '{target_col}' (Top 20 Values)")
            st.warning(f"'{target_col}' has too many unique values ({num_unique_values}) for a pie chart. Displaying the top 20 most frequent values instead.")
            
            top_20_counts = target_counts.head(20)
            
            fig = px.bar(
                x=top_20_counts.index,
                y=top_20_counts.values,
                title=f"Top 20 Most Frequent Values in '{target_col}'"
            )
            fig.update_layout(xaxis_title=target_col, yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)


with tab2:
    st.header("Pearson Correlation of Features")
    corr_matrix = calculate_correlation(df_featured)
    fig = px.imshow(
        corr_matrix, 
        text_auto=True, 
        aspect="auto", 
        title="Correlation Heatmap of Numerical Features",
        color_continuous_scale='RdBu_r'
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- RESTORED AI FEATURE ---
    st.subheader("AI Interpretation of Correlations")
    if st.button("Ask AI to Interpret the Heatmap"):
        with st.spinner("AI is analyzing the correlations..."):
            target_correlations = corr_matrix['TARGET'].sort_values(ascending=False).drop('TARGET')
            summary = (
                f"Top 5 features positively correlated with TARGET (default risk):\n{target_correlations.head(5).to_string()}\n\n"
                f"Top 5 features negatively correlated with TARGET (more likely to repay):\n{target_correlations.tail(5).to_string()}"
            )
            question = "What do these strong positive and negative correlations with the TARGET variable mean from a business perspective? Which features seem most important for predicting loan defaults?"
            interpretation = get_llm_interpretation("Pearson Correlation Heatmap", summary, question)
            with st.expander("View AI Analysis", expanded=True):
                st.markdown(interpretation)

with tab3:
    st.header("Relationship Between Features and Target")
    all_numeric_cols = df_featured.select_dtypes(include=np.number).columns.tolist()
    selected_feature_tab3 = st.selectbox("Select a feature to visualize:", options=all_numeric_cols, key="tab3_select")
    
    if selected_feature_tab3:
        fig = go.Figure()
        fig.add_trace(go.Violin(x=df_featured['TARGET'][df_featured['TARGET'] == 0], y=df_featured[selected_feature_tab3][df_featured['TARGET'] == 0], name='Repaid (0)', line_color='blue'))
        fig.add_trace(go.Violin(x=df_featured['TARGET'][df_featured['TARGET'] == 1], y=df_featured[selected_feature_tab3][df_featured['TARGET'] == 1], name='Defaulted (1)', line_color='orange'))
        fig.update_layout(title=f'Distribution of "{selected_feature_tab3}" vs. Loan Default Status', violinmode='group')
        st.plotly_chart(fig, use_container_width=True)

        # --- RESTORED AI FEATURE ---
        st.subheader(f"AI Interpretation for '{selected_feature_tab3}'")
        if st.button(f"Ask AI to Interpret this Relationship"):
            with st.spinner(f"AI is analyzing '{selected_feature_tab3}'..."):
                median_repaid = df_featured[df_featured['TARGET'] == 0][selected_feature_tab3].median()
                median_defaulted = df_featured[df_featured['TARGET'] == 1][selected_feature_tab3].median()
                summary = (
                    f"For the feature '{selected_feature_tab3}', the median value for loans that were REPAID is: {median_repaid:.4f}\n"
                    f"For the feature '{selected_feature_tab3}', the median value for loans that DEFAULTED is: {median_defaulted:.4f}"
                )
                question = f"Based on the difference in medians, what does this tell us about the predictive power of '{selected_feature_tab3}'? What is the business implication of this relationship?"
                interpretation = get_llm_interpretation(f"Feature '{selected_feature_tab3}' vs. Target", summary, question)
                with st.expander("View AI Analysis", expanded=True):
                    st.markdown(interpretation)

with tab4:
    st.header("Univariate Feature Distributions")
    feature_dist = st.selectbox("Select a numerical feature:", options=df_featured.select_dtypes(include=np.number).columns.tolist(), key="tab4_select")
    fig = px.histogram(df_featured, x=feature_dist, marginal="box", title=f"Distribution of {feature_dist}")
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header("🔬 Anomaly Detection using Isolation Forest")
    st.write("This tool helps identify unusual data points that don't conform to expected patterns. Select one or two features to analyze.")
    
    numerical_cols = df_featured.select_dtypes(include=np.number).columns.tolist()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        x_axis_feature = st.selectbox("Select a feature for the X-axis:", numerical_cols, index=numerical_cols.index("AMT_INCOME_TOTAL"))
    with col2:
        y_axis_feature = st.selectbox("Select a feature for the Y-axis (optional):", [None] + numerical_cols, index=1+numerical_cols.index("AMT_CREDIT"))
    with col3:
        contamination = st.slider("Select anomaly sensitivity (contamination):", min_value=0.001, max_value=0.1, value=0.01, step=0.005, help="The proportion of outliers to expect in the data.")
    
    if st.button("Find Anomalies"):
        with st.spinner("Training model and identifying anomalies..."):
            features_to_analyze = [x_axis_feature]
            if y_axis_feature:
                features_to_analyze.append(y_axis_feature)
            
            X = df_featured[features_to_analyze].dropna()
            model = IsolationForest(contamination=contamination, random_state=42)
            preds = model.fit_predict(X)
            
            st.session_state['anomaly_plot_data'] = X.copy()
            st.session_state['anomaly_plot_data']['Anomaly'] = pd.Series(preds, index=X.index).map({1: 'Normal', -1: 'Anomaly'})
            
            st.session_state['anomalies_df'] = df_featured.loc[st.session_state['anomaly_plot_data'][st.session_state['anomaly_plot_data']['Anomaly'] == 'Anomaly'].index]
            
            st.session_state['anomaly_plot_title'] = f"Anomaly Detection in {x_axis_feature}" + (f" vs. {y_axis_feature}" if y_axis_feature else "")
            st.session_state['anomaly_plot_x'] = x_axis_feature
            st.session_state['anomaly_plot_y'] = y_axis_feature

    if 'anomalies_df' in st.session_state:
        st.plotly_chart(px.scatter(
            st.session_state['anomaly_plot_data'], 
            x=st.session_state['anomaly_plot_x'], 
            y=st.session_state['anomaly_plot_y'], 
            color='Anomaly',
            color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'},
            title=st.session_state['anomaly_plot_title']
        ), use_container_width=True)
        
        st.write("Detected Anomalies:")
        anomalies_df_from_state = st.session_state['anomalies_df']
        st.dataframe(anomalies_df_from_state)
        
        st.subheader("AI Root Cause Analysis")
        if not anomalies_df_from_state.empty:
            selected_anomaly_id = st.selectbox("Select an anomaly to analyze with AI:", options=anomalies_df_from_state.index, key="anomaly_select")
            if st.button("Ask AI to Analyze Selected Anomaly"):
                with st.spinner("AI is analyzing the selected anomaly..."):
                    anomaly_data = anomalies_df_from_state.loc[selected_anomaly_id].to_dict()
                    
                    for key, value in anomaly_data.items():
                        if isinstance(value, np.generic):
                            anomaly_data[key] = value.item()
                    
                    summary = f"An anomaly was detected with the following data:\n{anomaly_data}"
                    question = "Based on this data, what could be the potential root cause of this anomaly? Is it likely a data entry error, a fraudulent case, or a rare but legitimate event? Please provide a hypothesis."
                    interpretation = get_llm_interpretation("Anomaly Root Cause Analysis", summary, question)
                    st.markdown(interpretation)
        else:
            st.info("No anomalies were detected with the current sensitivity setting.")

with tab6:
    st.header("🔗 End-to-End Data Lineage")
    st.write("Visualize the complete data flow, from source files to engineered features.")

    st.subheader("1. Source-to-Table Data Flow")
    st.write("Upload related CSV files (e.g., `bureau.csv`, `previous_application.csv`) to visualize how they join to create the final analytical dataset. File names must be exact to be recognized.")

    known_relationships = {
        'bureau.csv': {'on': 'SK_ID_CURR', 'type': 'Left Join'},
        'previous_application.csv': {'on': 'SK_ID_CURR', 'type': 'Left Join'},
        'POS_CASH_balance.csv': {'on': 'SK_ID_PREV', 'type': 'Joins to previous_application'},
        'installments_payments.csv': {'on': 'SK_ID_PREV', 'type': 'Joins to previous_application'},
        'credit_card_balance.csv': {'on': 'SK_ID_PREV', 'type': 'Joins to previous_application'},
        'bureau_balance.csv': {'on': 'SK_ID_BUREAU', 'type': 'Joins to bureau'}
    }

    related_files = st.file_uploader(
        "Upload related data sources", 
        type="csv", 
        accept_multiple_files=True
    )

    dot_flow = graphviz.Digraph('DataFlow', graph_attr={'rankdir': 'LR', 'splines': 'ortho'})
    dot_flow.attr('node', shape='box', style='rounded,filled')

    dot_flow.node('main_df', 'Main Application Data\n(application_train.csv)', fillcolor='orange')

    if related_files:
        uploaded_filenames = [f.name for f in related_files]
        
        for filename in uploaded_filenames:
            if filename in known_relationships:
                rel = known_relationships[filename]
                dot_flow.node(filename, f"Source: {filename}", fillcolor='lightblue')
                
                if rel['type'] == 'Left Join':
                    dot_flow.edge(filename, 'main_df', label=f"  {rel['type']} on\n  {rel['on']}")
                elif rel['type'] == 'Joins to bureau' and 'bureau.csv' in uploaded_filenames:
                     dot_flow.edge(filename, 'bureau.csv', label=f"  Joined on\n  {rel['on']}")
                elif rel['type'] == 'Joins to previous_application' and 'previous_application.csv' in uploaded_filenames:
                    dot_flow.edge(filename, 'previous_application.csv', label=f"  Joined on\n  {rel['on']}")

    st.graphviz_chart(dot_flow)
    st.info("This is a simulated lineage graph. It shows the intended join relationships between the recognized source files without performing the actual, computationally expensive merge operation.")

    st.divider()

    st.subheader("2. Feature Creation Recipe")
    st.write("After the source tables are joined, we create new features. Select a feature to see how it was derived from other columns in the final table.")

    feature_recipes = {
        'CREDIT_INCOME_PERCENT': {'inputs': ['AMT_CREDIT', 'AMT_INCOME_TOTAL'], 'logic': 'AMT_CREDIT / AMT_INCOME_TOTAL', 'description': 'Ratio of the loan amount to the client\'s total income.'},
        'ANNUITY_INCOME_PERCENT': {'inputs': ['AMT_ANNUITY', 'AMT_INCOME_TOTAL'], 'logic': 'AMT_ANNUITY / AMT_INCOME_TOTAL', 'description': 'Ratio of the loan annuity to the client\'s income.'},
        'CREDIT_TERM': {'inputs': ['AMT_ANNUITY', 'AMT_CREDIT'], 'logic': 'AMT_ANNUITY / AMT_CREDIT', 'description': 'Approximation of the loan term.'},
        'DAYS_EMPLOYED_PERCENT': {'inputs': ['DAYS_EMPLOYED', 'DAYS_BIRTH'], 'logic': 'DAYS_EMPLOYED / DAYS_BIRTH', 'description': 'Ratio of days employed to days since birth.'}
    }

    selected_lineage_feature = st.selectbox("Select an engineered feature to see its recipe:", options=list(feature_recipes.keys()))
    
    if selected_lineage_feature:
        recipe = feature_recipes[selected_lineage_feature]
        
        dot_recipe = graphviz.Digraph('FeatureRecipe')
        dot_recipe.attr('node', shape='box', style='rounded,filled')
        
        for input_col in recipe['inputs']:
            dot_recipe.node(input_col, f"Source Column:\n{input_col}", fillcolor='lightblue')
            dot_recipe.edge(input_col, 'transformation_logic')
            
        dot_recipe.node('transformation_logic', f"Transformation Logic:\n{recipe['logic']}", shape='diamond', fillcolor='lightgreen')
        dot_recipe.node(selected_lineage_feature, f"Engineered Feature:\n{selected_lineage_feature}", fillcolor='orange')
        dot_recipe.edge('transformation_logic', selected_lineage_feature)
        
        st.graphviz_chart(dot_recipe)

        if st.button("Ask AI to Explain this Feature's Importance"):
            summary = (f"Feature Name: {selected_lineage_feature}\n"
                       f"How it's made: It is calculated using the formula '{recipe['logic']}'.\n"
                       f"Business Description: {recipe['description']}")
            question = "In the context of credit risk, why is this an important feature to engineer? What kind of predictive insight might it provide that the original columns alone could not?"
            interpretation = get_llm_interpretation("Feature Lineage and Importance", summary, question)
            st.markdown(interpretation)
