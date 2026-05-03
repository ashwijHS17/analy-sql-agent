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
st.set_page_config(page_title="AI SQL Analyst", page_icon="🚀", layout="wide")

# --- STYLE CUSTOMIZATION ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 AI SQL Data Analyst Agent")
st.markdown("Upload a CSV and chat with your data using natural language.")

# --- API KEY & CONFIGURATION ---
# Check Streamlit Secrets first, then fallback to Sidebar input
groq_api_key = None

if "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]
else:
    with st.sidebar:
        st.header("🔑 API Configuration")
        groq_api_key = st.text_input("Enter Groq API Key", type="password")
        if not groq_api_key:
            st.warning("Please provide an API key to continue.")
            st.info("Tip: Add 'GROQ_API_KEY' to your Streamlit Secrets to skip this.")

with st.sidebar:
    model_name = st.selectbox(
        "Select Model", 
        ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"],
        index=1
    )
    st.divider()
    st.markdown("### How to use")
    st.write("1. Upload a CSV file")
    st.write("2. Ask a question (e.g. 'What is the average price?')")
    st.write("3. Mention 'chart' to see a visual.")

# --- DATA LOADING ---
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Load and Preview Data
    df = pd.read_csv(uploaded_file)
    with st.expander("👀 View Data Preview"):
        st.dataframe(df.head(10), use_container_width=True)

    # Create SQLite Engine (In-Memory)
    # We name the table 'data_table' so the Agent can find it easily
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

            # Create LangChain SQL Agent using string-based type for stability
            agent_executor = create_sql_agent(
                llm=llm,
                db=db,
                agent_type="zero-shot-react-description",
                verbose=True,
                handle_parsing_errors=True
            )

            # --- CHAT INTERFACE ---
            st.divider()
            user_question = st.text_input("💬 Ask your data a question:", placeholder="e.g., Which product has the highest sales?")

            if user_question:
                with st.spinner("🤖 Agent is thinking and writing SQL..."):
                    try:
                        # Invoke Agent
                        response = agent_executor.invoke({"input": user_question})
                        
                        # Layout for Results
                        col1, col2 = st.columns([1, 1])

                        with col1:
                            st.subheader("🤖 Analysis")
                            st.info(response["output"])

                        with col2:
                            st.subheader("📊 Visualization")
                            # Trigger chart logic if keywords exist
                            if any(word in user_question.lower() for word in ["chart", "plot", "graph", "visualize"]):
                                numeric_cols = df.select_dtypes(include=['number']).columns
                                if len(numeric_cols) >= 1:
                                    # Logic for a simple auto-chart
                                    fig = px.bar(df.head(20), x=df.columns[0], y=numeric_cols[0], 
                                                 title=f"{numeric_cols[0]} by {df.columns[0]}")
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("No numeric data found for visualization.")
                            else:
                                st.write("No chart requested. Try asking: 'Show me a bar chart of [column].'")

                    except Exception as e:
                        st.error(f"Execution Error: {e}")

        except Exception as e:
            st.error(f"Agent Initialization Error: {e}")
else:
    st.info("Waiting for a CSV file to be uploaded...")

# --- FOOTER ---
st.divider()
st.caption("Built with LangChain, Groq, and Streamlit.")
