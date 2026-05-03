import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sqlalchemy import create_engine
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_types import AgentType
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI SQL Data Analyst", page_icon="🚀", layout="wide")
st.title("🚀 AI SQL Data Analyst Agent")
st.markdown("Upload a CSV and ask questions in natural language.")

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
            agent_executor = create_sql_agent(
                llm=llm,
                db=db,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True
            )

            # --- CHAT INTERFACE ---
            user_question = st.text_input("Ask a question about your data (e.g., 'What is the average price?' or 'Show me a bar chart of sales by region')")

            if user_question:
                with st.spinner("Analyzing..."):
                    # 1. Get Answer and SQL Query
                    # The agent will automatically find the table name 'data_table'
                    response = agent_executor.invoke({"input": user_question})
                    
                    st.success("Analysis Complete!")
                    
                    # Layout for Results
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.markdown("### 🤖 Answer")
                        st.write(response["output"])

                    with col2:
                        st.markdown("### 📊 Suggested Visualization")
                        # Basic heuristic for charts
                        if "chart" in user_question.lower() or "plot" in user_question.lower() or "graph" in user_question.lower():
                            try:
                                # We use the dataframe directly for visualization to ensure stability
                                numeric_cols = df.select_dtypes(include=['number']).columns
                                if len(numeric_cols) >= 1:
                                    fig = px.histogram(df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}")
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("No numeric columns found for plotting.")
                            except Exception as e:
                                st.error(f"Error generating chart: {e}")
                        else:
                            st.info("Add 'show me a chart' to your prompt to trigger a visualization.")

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter your Groq API Key in the sidebar.")

else:
    st.info("Waiting for CSV upload...")
