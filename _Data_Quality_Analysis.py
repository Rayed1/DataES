import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI

# --- Configuration for LM Studio ---
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key="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")

# --- Initial AI Suggestion Engine (Unchanged) ---
def get_precise_suggestions(df):
    """
    Analyzes a DataFrame and provides precise, evidence-based data quality suggestions.
    It only reports issues when there is statistical evidence that they exist.
    """
    suggestions = {}
    # Use a larger sample for more accurate statistical checks, but cap it for performance
    sample_df = df.head(5000)
    total_rows = len(sample_df)

    for column in sample_df.columns:
        col_suggestions = []
        col_series = sample_df[column]
        dtype = str(col_series.dtype)

        # --- Check 1: Missing Values (More nuanced) ---
        if col_series.isnull().any():
            missing_percentage = col_series.isnull().mean() * 100
            if missing_percentage > 40:
                # Flag high-levels of missing data as a major concern
                col_suggestions.append(f"**Major Concern - High Missing Values:** **{missing_percentage:.1f}%** of the data is missing. This column may need to be dropped or require advanced imputation.")
            else:
                col_suggestions.append(f"**Missing Values Found:** {missing_percentage:.1f}% of the data is missing. Consider a suitable imputation strategy (e.g., mean, median, mode).")

        # --- Checks for Numerical Columns ---
        if pd.api.types.is_numeric_dtype(col_series):
            # Check 2: Zero or Near-Zero Variance
            if col_series.nunique() == 1:
                col_suggestions.append(f"**Constant Value (Zero Variance):** This column contains only one value (`{col_series.iloc[0]}`). It offers no predictive value and can likely be dropped.")
            
            # Check 3: Outlier Detection using the IQR method (more robust than generic suggestion)
            # Only check for outliers if it's not a binary or low-cardinality integer column
            if col_series.nunique() > 10:
                Q1 = col_series.quantile(0.25)
                Q3 = col_series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = col_series[(col_series < lower_bound) | (col_series > upper_bound)]
                if not outliers.empty:
                    col_suggestions.append(f"**Potential Outliers Detected:** Found **{len(outliers)}** value(s) outside the 1.5x Interquartile Range (e.g., `{outliers.iloc[0]}`). These extreme values could skew statistical analysis and model performance.")

        # --- Checks for Categorical/Object Columns ---
        elif pd.api.types.is_object_dtype(col_series):
            # Check 4: High Cardinality (more nuanced)
            num_unique = col_series.nunique()
            cardinality_ratio = num_unique / total_rows
            if cardinality_ratio > 0.95 and num_unique > 1000:
                # This is very likely a unique identifier
                col_suggestions.append(f"**Unique Identifier:** This column has a unique value for nearly every row ({num_unique} unique values). It should be treated as an ID, not a categorical feature.")
            elif num_unique > 50:
                col_suggestions.append(f"**High Cardinality:** Contains **{num_unique}** unique text categories. This may require feature engineering (e.g., grouping, target encoding) before use in some models.")

            # Check 5: Inconsistent Formatting (Whitespace and Case)
            # Leading/Trailing Whitespace
            if col_series.astype(str).str.strip().nunique() < num_unique:
                col_suggestions.append("**Inconsistent Whitespace:** Found values with leading or trailing whitespace (e.g., ' value' vs. 'value'). These should be standardized by trimming whitespace.")
            
            # Mixed Case
            if col_series.nunique() > col_series.str.lower().nunique():
                 # Find an example
                lower_counts = col_series.str.lower().value_counts()
                example_lower = lower_counts[lower_counts > 1].index[0]
                example_cases = col_series[col_series.str.lower() == example_lower].unique()
                col_suggestions.append(f"**Inconsistent Casing:** Found values that differ only by case (e.g., `{example_cases[0]}` vs. `{example_cases[1]}`). Standardize by converting the column to a consistent case (e.g., lowercase).")

        if col_suggestions:
            suggestions[column] = col_suggestions
            
    return suggestions


with st.sidebar:
    st.header("🤖 AI Model Configuration")
    # A selection of popular models available on OpenRouter
    selected_model = st.selectbox(
        "Choose an AI Model:",
        [
            "mistralai/mistral-7b-instruct",
            "google/gemini-flash-1.5",
            "meta-llama/llama-3-8b-instruct",
            "openai/gpt-3.5-turbo",
            "anthropic/claude-3-haiku"
        ],
        help="Select a model from OpenRouter to power the AI analysis."
    )
    # Store the selected model in the session state so other pages can access it
    st.session_state['selected_model'] = selected_model
    st.info(f"Using **{selected_model}** for all AI tasks.")


# --- Column Profile Generator (Unchanged) ---
def generate_column_profile(df, column_name):
    column_series = df[column_name]
    dtype = str(column_series.dtype)
    profile = [f"Data Type: {dtype}", f"Missing Values: {column_series.isnull().sum()} ({column_series.isnull().mean():.2%})"]
    if 'int' in dtype or 'float' in dtype:
        profile.append("Statistics:")
        profile.append(str(column_series.describe().to_dict()))
    elif 'object' in dtype:
        profile.append(f"Number of Unique Categories: {column_series.nunique()}")
        profile.append("Top 5 Most Frequent Categories:")
        profile.append(str(column_series.value_counts().head(5).to_dict()))
    return "\n".join(profile)

# --- RAG-Enabled LLM Function (Unchanged) ---
def get_llm_analysis(df, column_name, suggestions, user_query, data_dictionary={}):
    system_prompt = (
        "You are a world-class data quality analyst. You will be given a statistical profile of a data column, "
        "business context from a data dictionary, a list of potential issues, and a user's question. "
        "Your task is to synthesize ALL this information to provide a hyper-specific, business-aware analysis. "
        "Directly reference the business context to explain the real-world impact of the data quality issues."
    )
    
    column_profile = generate_column_profile(df, column_name)
    formatted_suggestions = "\n- ".join(suggestions)
    
    business_context = data_dictionary.get(column_name)
    context_section = ""
    if business_context:
        context_section = f"--- Business Context (from Data Dictionary) ---\n{business_context}\n\n"

    user_prompt = (
        "Please analyze the following data column based on all the information provided.\n\n"
        f"--- Data Column Profile for '{column_name}' ---\n{column_profile}\n\n"
        f"{context_section}"
        f"--- Automated Analysis Findings ---\n- {formatted_suggestions}\n\n"
        f"--- My Specific Question ---\n'{user_query}'\n\n"
        "Provide your expert analysis, making sure to connect the statistical data with the business context:"
    )

    try:
        completion = client.chat.completions.create(
            # --- CRITICAL CHANGE ---
            # It now gets the model name from the session state
            model=st.session_state.get('selected_model', "mistralai/mistral-7b-instruct"), 
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.7,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error connecting to LM Studio. Please ensure the server is running. Details: {e}"

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("🤖 DataEs: The Context-Aware Assistant")
st.write("Upload a dataset and an optional data dictionary to get automated checks, visualizations, and business-aware AI analysis.")

col1, col2 = st.columns(2)
with col1:
    main_file = st.file_uploader("1. Upload your main dataset (CSV) which Contains Training Data", type="csv")
with col2:
    dictionary_file = st.file_uploader("2. Upload Official Data Dictionary (CSV)", type="csv")

if dictionary_file is not None:
    try:
        dict_df = pd.read_csv(dictionary_file, encoding='latin1') 
        if 'Row' in dict_df.columns and 'Description' in dict_df.columns:
            st.session_state['data_dictionary'] = pd.Series(dict_df.Description.values, index=dict_df.Row).to_dict()
            st.success("Official Data Dictionary loaded successfully!")
        else:
            st.error("The dictionary file must have 'Row' and 'Description' columns.")
    except Exception as e:
        st.error(f"Error processing dictionary file: {e}")

# This block now handles the ONE-TIME processing when a new file is uploaded.
if main_file is not None and 'df' not in st.session_state:
    with st.spinner("Reading and analyzing your dataset... This may take a moment."):
        df = pd.read_csv(main_file, nrows=20000)
        # Store the main dataframe in the session state
        st.session_state['df'] = df
        
        # --- CRITICAL FIX: Analyze ONCE and store suggestions in session state ---
        st.session_state['suggestions'] = get_precise_suggestions(df)
        st.success("Dataset loaded and initial analysis complete!")

if 'suggestions' in st.session_state:
    suggestions = st.session_state['suggestions']
    
    with st.expander("View Automated Data Quality Suggestions", expanded=True):
        if not suggestions:
            st.success("✅ Our analysis found no immediate data quality concerns based on the initial scan.")
        else:
            st.info(f"Found potential data quality issues in {len(suggestions)} columns. Click on a column to see the details.")
            
            # Loop through each column that has suggestions
            for col, sug_list in suggestions.items():
                # Create a nested expander for each column's findings
                with st.expander(f"Findings for Column: **{col}**"):
                    # Loop through the list of suggestions for that column
                    for suggestion in sug_list:
                        # Use st.markdown to render the suggestions as a bulleted list
                        st.markdown(f"- {suggestion}")

if 'df' in st.session_state:
    st.header("🔍 Visual Deep Dive & AI Analyst Chat")
    st.write("Select a column to visualize its distribution and ask our AI Analyst for a detailed explanation.")
    
    df = st.session_state['df']
    suggestions = st.session_state['suggestions']
    
    column_to_analyze = st.selectbox("Select a column to analyze:", options=list(df.keys()))

    if column_to_analyze:
        col_suggestions = suggestions.get(column_to_analyze, [])
        viz_col, chat_col = st.columns([2, 3])

        with viz_col:
            st.subheader(f"Analysis for: `{column_to_analyze}`")
            column_series = df[column_to_analyze]
            dtype = str(column_series.dtype)

            # --- NEW: Dynamic Statistics Display ---
            st.markdown("##### Key Column Statistics")
            
            if 'int' in dtype or 'float' in dtype:
                # Stats for numerical columns
                stats = column_series.describe()
                stat_cols = st.columns(4)
                stat_cols[0].metric("Total Rows", f"{int(stats['count']):,}")
                stat_cols[1].metric("Mean", f"{stats['mean']:.2f}")
                stat_cols[2].metric("Min", f"{int(stats['min']):,}")
                stat_cols[3].metric("Max", f"{int(stats['max']):,}")
            else:
                # Stats for categorical columns
                stats = column_series.describe()
                stat_cols = st.columns(3)
                stat_cols[0].metric("Total Rows", f"{int(stats['count']):,}")
                stat_cols[1].metric("Unique Values", f"{int(stats['unique']):,}")
                stat_cols[2].metric("Top Value", str(stats['top']))
            
            missing_count = column_series.isnull().sum()
            missing_percent = column_series.isnull().mean() * 100
            st.metric("Missing Values", f"{missing_count:,} ({missing_percent:.2f}%)")
            st.divider()
            # --- END of New Section ---

            st.markdown("##### Visualizations")
            if 'int' in dtype or 'float' in dtype:
                st.plotly_chart(px.histogram(df, x=column_to_analyze, title=f"Distribution of {column_to_analyze}"), use_container_width=True)
                st.plotly_chart(px.box(df, y=column_to_analyze, title=f"Box Plot of {column_to_analyze}"), use_container_width=True)
            elif 'object' in dtype:
                value_counts = column_series.value_counts().nlargest(20).reset_index()
                value_counts.columns = [column_to_analyze, 'count']
                st.plotly_chart(px.bar(value_counts, x=column_to_analyze, y='count', title=f"Top 20 Categories in {column_to_analyze}"), use_container_width=True)

        with chat_col:
            st.subheader("Chat with your AI Data Analyst")
            st.info(f"**Automated Findings for `{column_to_analyze}`:**\n- " + '\n- '.join(col_suggestions))
            
            user_question = st.text_input("Ask a question about this column's quality...", value=f"Explain the business impact of these issues for '{column_to_analyze}' and suggest fixes.")
            
            if st.button("Get AI Analysis"):
                with st.spinner("🤖 The context-aware analyst is thinking..."):
                    data_dictionary = st.session_state.get('data_dictionary', {})
                    response = get_llm_analysis(df, column_to_analyze, col_suggestions, user_question, data_dictionary)
                    st.markdown(response)