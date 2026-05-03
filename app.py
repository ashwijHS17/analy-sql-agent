import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI SQL Analyst", page_icon="🚀", layout="wide")

st.title("🚀 AI SQL Data Analyst Agent")
st.markdown("Upload a CSV and chat with your data. **Model: Llama 3.3 70B Versatile**")

# --- API KEY & CONFIGURATION ---
groq_api_key = None

# Prioritize Streamlit Secrets for security
if "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]
else:
    with st.sidebar:
        st.header("🔑 API Configuration")
        groq_api_key = st.text_input("Enter Groq API Key", type="password")
        if not groq_api_key:
            st.warning("Please provide an API key to continue.")

with st.sidebar:
    st.header("⚙️ Model Info")
    # Locked to llama-3.3-70b-versatile as requested
    st.success("Using: llama-3.3-70b-versatile")
    st.divider()
    st.info("This model is optimized for complex SQL generation and data reasoning.")

# --- DATA LOADING ---
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    with st.expander("👀 View Data Preview"):
        st.dataframe(df.head(10), use_container_width=True)

    # In-Memory SQLite Setup
    engine = create_engine("sqlite:///temp_data.db")
    df.to_sql("data_table", engine, index=False, if_exists="replace")
    db = SQLDatabase(engine)

    # --- AGENT SETUP ---
    if groq_api_key:
        try:
            # Initializing specifically with Llama 3.3 70B
            llm = ChatGroq(
                groq_api_key=groq_api_key, 
                model_name="llama-3.3-70b-versatile",
                temperature=0
            )

            # LangChain Agent utilizing string-based identification for stability
            agent_executor = create_sql_agent(
                llm=llm,
                db=db,
                agent_type="zero-shot-react-description",
                verbose=True,
                handle_parsing_errors=True
            )

            # --- CHAT INTERFACE ---
            st.divider()
            user_question = st.text_input("💬 Ask your data a question:", placeholder="e.g., Which category has the highest average price?")

            if user_question:
                with st.spinner("🤖 Llama 3.3 is analyzing..."):
                    try:
                        response = agent_executor.invoke({"input": user_question})
                        
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.subheader("🤖 Analysis")
                            st.info(response["output"])

                        with col2:
                            st.subheader("📊 Visualization")
                            if any(word in user_question.lower() for word in ["chart", "plot", "graph"]):
                                # SMARTER VISUALIZATION LOGIC
                                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

                                # Filter out "ID" columns for the Y-axis
                                filtered_numeric = [c for c in numeric_cols if 'id' not in c.lower()]
                                
                                if len(numeric_cols) >= 1:
                                    x_axis = categorical_cols[0] if categorical_cols else numeric_cols[0]
                                    y_axis = filtered_numeric[0] if filtered_numeric else numeric_cols[0]

                                    fig = px.bar(
                                        df.head(20), 
                                        x=x_axis, 
                                        y=y_axis, 
                                        title=f"{y_axis} by {x_axis}",
                                        color=x_axis if categorical_cols else None,
                                        template="plotly_dark"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("No numeric columns available.")
                            else:
                                st.write("Ask for a 'chart' to see visuals.")

                    except Exception as e:
                        st.error(f"Execution Error: {e}")

        except Exception as e:
            st.error(f"Agent Initialization Error: {e}")
else:
    st.info("Please upload a CSV file to begin.")
