import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent


# Load environment variables
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    st.error("Error: HUGGINGFACEHUB_API_TOKEN environment variable not set.")
    st.stop()

# Streamlit page setup
st.set_page_config(page_title="AI CSV Data Analyzer", layout="wide")

st.title("AI CSV Data Analyzer")
st.write("Upload a CSV file and ask questions about your dataset.")

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    st.write("Dataset shape:", df.shape)
    st.write("Columns:", list(df.columns))

    # Initialize ChatOpenAI using HuggingFace router
    llm = ChatOpenAI(
        model="mistralai/Mistral-7B-Instruct-v0.2:featherless-ai",
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
        temperature=0
    )

    # Create dataframe agent
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True
    )
    question = st.text_input("Ask a question about the dataset")
    if question:
        with st.spinner("Analyzing dataset..."):
            try:
                response = agent.invoke({"input":question})
                st.subheader("Answer")
                st.write(response["output"])
            except Exception as e:
                st.error(f"Error during analysis: {e}")