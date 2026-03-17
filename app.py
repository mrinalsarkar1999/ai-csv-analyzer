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

    # Define a strict, token-efficient prompt with formatting and plotting rules
    custom_prefix = """You are an expert data analyst working with a pandas dataframe named `df`. 
The dataframe is already loaded as `df`. Do not mock data.

CRITICAL FORMATTING RULES:
1. You have access to a tool called `python_repl_ast`. You MUST format your actions EXACTLY like the examples below. 
2. NEVER put descriptive text in the Action field. The Action field MUST be exactly "python_repl_ast".
3. NEVER output an "Action" and a "Final Answer" in the same response. If you output an Action, stop generating and wait for the observation.

--- CALCULATION EXAMPLE ---
Thought: I need to calculate the mean of the Age column.
Action: python_repl_ast
Action Input: df['Age'].mean()
-----------------------

--- PLOTTING EXAMPLE ---
Thought: I need to create a bar chart of survival rates.
Action: python_repl_ast
Action Input: import matplotlib.pyplot as plt; df.groupby('Sex')['Survived'].sum().plot(kind='bar'); plt.savefig('temp_plot.png'); plt.clf()
------------------------

STREAMLIT PLOTTING RULES:
If the user asks for a chart or graph, you MUST include `plt.savefig('temp_plot.png')` in your Action Input code before calling `plt.clf()`.
Only AFTER the tool returns a successful observation should you write: "Final Answer: Plot saved as temp_plot.png".
"""

    # Create dataframe agent with error handling and custom prefix
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,
        handle_parsing_errors="Check your output and make sure it conforms! Do not output an Action and a Final Answer in the same step.",
        max_iterations=5,            # Stops runaway loops to save tokens
        prefix=custom_prefix         # Applies the strict rules defined above
    )
    
    question = st.text_input("Ask a question about the dataset")
    if question:
        
        # Clean up any old plot before starting a new analysis
        if os.path.exists("temp_plot.png"):
            os.remove("temp_plot.png")

        with st.spinner("Analyzing dataset..."):
            try:
                response = agent.invoke({"input": question})
                
                st.subheader("Answer")
                st.write(response["output"])
                
                # Check if the agent created a plot during its thought process
                if os.path.exists("temp_plot.png"):
                    st.image("temp_plot.png", caption="Generated Plot")
                    
            except Exception as e:
                st.error(f"Error during analysis: {e}")