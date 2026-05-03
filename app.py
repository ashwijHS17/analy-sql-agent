import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sqlalchemy import create_engine
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI SQL Data Analyst", page_icon="🚀", layout="wide")
st.title("🚀 AI SQL Data Analyst Agent")
st.markdown("Upload a CSV and ask questions in natural language. This agent will convert your question to SQL and execute it.")

# --- SIDEBAR: API KEY ---
with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Enter Groq API Key", type="password")
    model_name = st.selectbox("Select Model", ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"])
    st.info("Get your key at: [console.groq.com](https://console.groq.com/)")

# --- DATA LOADING ---
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head(5))

    # Create SQLite Engine (In-Memory)
    # Using 'data_table' as the fixed table name for the agent to find easily
    engine = create_engine("sqlite:///temp_data.db")
    df.to_sql("data_table", engine, index=False, if_exists="replace")
    db = SQLDatabase(engine)

    # --- AGENT SETUP ---
    if groq_api_key:
        try:
            llm = ChatGroq(
                groq_api_key=groq_api_key, 
                model_name=model_name,
                temperature=0
            )

            # Create LangChain SQL Agent 
            # Note: agent_type is passed as a string to avoid versioning import errors
            agent_executor = create_sql_agent(
                llm=llm,
                db=db,
                agent_type="zero-shot-react-description",
                verbose=True,
                handle_parsing_errors=True
            )

            # --- CHAT INTERFACE ---
            user_question = st.text_input("Ask a question about your data (e.g., 'What is the total revenue?' or 'Show a bar chart of sales by product')")

            if user_question:
                with st.spinner("Analyzing data..."):
                    # Execute the query
                    response = agent_executor.invoke({"input": user_question})
                    
                    st.success("Analysis Complete!")
                    
                    # Layout for Results
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.markdown("### 🤖 Answer")
                        st.write(response["output"])

                    with col2:
                        st.markdown("### 📊 Visualization")
                        # Basic logic to trigger a plot if keywords are mentioned
                        if any(word in user_question.lower() for word in ["chart", "plot", "graph", "visualize"]):
                            try:
                                numeric_cols = df.select_dtypes(include=['number']).columns
                                if len(numeric_cols) >= 1:
                                    # Defaulting to a histogram for distribution if not specified
                                    fig = px.histogram(df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}")
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("No numeric columns found to create a chart.")
                            except Exception as viz_err:
                                st.error(f"Could not generate visual: {viz_err}")
                        else:
                            st.info("Tip: Ask to 'show a chart' to see data visualizations.")

        except Exception as e:
            st.error(f"Agent Error: {e}")
    else:
        st.warning("Please enter your Groq API Key in the sidebar to begin.")

else:
    st.info("Please upload a CSV file to start the analysis.")
