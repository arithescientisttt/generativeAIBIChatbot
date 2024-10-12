import os 
import streamlit as st
import pandas as pd
import sqlite3
import logging
import ast  # For parsing string representations of lists

from langchain_community.chat_models import ChatOpenAI
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize conversation history
if 'history' not in st.session_state:
    st.session_state.history = []

# OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is set
if not openai_api_key:
    st.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Step 1: Upload CSV data file (or use default)
st.title("Business Data Insights Chatbot: Automating SQL Generation & Insights Extraction")
st.write("Upload a CSV file to get started, or use the default dataset.")

csv_file = st.file_uploader("Upload your CSV file", type=["csv"])
if csv_file is None:
    data = pd.read_csv("default_data.csv")  # Ensure this file exists
    st.write("Using default_data.csv file.")
    table_name = "default_table"
else:
    data = pd.read_csv(csv_file)
    table_name = csv_file.name.split('.')[0]
    st.write(f"Data Preview ({csv_file.name}):")
    st.dataframe(data.head())

# Display column names
st.write("**Available Columns:**")
st.write(", ".join(data.columns.tolist()))

# Step 2: Load CSV data into SQLite database
db_file = 'my_database.db'
conn = sqlite3.connect(db_file)
data.to_sql(table_name, conn, index=False, if_exists='replace')
conn.close()

# Create SQLDatabase instance
db = SQLDatabase.from_uri(f"sqlite:///{db_file}", include_tables=[table_name])

# Initialize the LLM
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)

# Initialize the SQL Agent
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_executor_kwargs={"return_intermediate_steps": True}
)

# Step 3: Sample Questions
st.write("**Sample Questions:**")
sample_questions = [
    "Summarize the data for me.",
    "Do you notice any correlations in the datasets?",
    "Can you offer any recommendations based on the datasets?",
    "Provide an analysis of some numbers across some categories."
]

def set_sample_question(question):
    st.session_state['user_input'] = question
    process_input()

for question in sample_questions:
    st.button(question, on_click=set_sample_question, args=(question,))

# Step 4: Define the callback function
def process_input():
    user_prompt = st.session_state.get('user_input', '')

    if user_prompt:
        try:
            # Append user message to history
            st.session_state.history.append({"role": "user", "content": user_prompt})

            # Use the agent to get the response
            with st.spinner("Processing..."):
                response = agent_executor(user_prompt)

            # Extract the final answer and the data from intermediate steps
            final_answer = response['output']
            intermediate_steps = response['intermediate_steps']

            # Initialize an empty list for SQL result
            sql_result = []

            # Find the SQL query result
            for step in intermediate_steps:
                if step[0].tool == 'sql_db_query':
                    # The result is a string representation of a list
                    sql_result = ast.literal_eval(step[1])
                    break

            # Convert the result to a DataFrame for better formatting
            if sql_result:
                df_result = pd.DataFrame(sql_result)
                sql_result_formatted = df_result.to_markdown(index=False)
            else:
                sql_result_formatted = "No results found."

            # Include the data in the final answer
            assistant_response = f"{final_answer}\n\n**Query Result:**\n{sql_result_formatted}"

            # Append the assistant's response to the history
            st.session_state.history.append({"role": "assistant", "content": assistant_response})

            # Generate insights based on the response
            insights_template = """
            You are an expert data analyst. Based on the user's question and the response provided below, generate a concise analysis that includes key data insights and actionable recommendations. Limit the response to a maximum of 150 words.

            User's Question: {question}

            Response:
            {response}

            Concise Analysis:
            """
            insights_prompt = PromptTemplate(template=insights_template, input_variables=['question', 'response'])
            insights_chain = LLMChain(llm=llm, prompt=insights_prompt)

            insights = insights_chain.run({'question': user_prompt, 'response': assistant_response})

            # Append the assistant's insights to the history
            st.session_state.history.append({"role": "assistant", "content": insights})
        except Exception as e:
            logging.error(f"An error occurred: {e}")

            # Check for specific errors related to missing columns
            if "no such column" in str(e).lower():
                assistant_response = "Error: One or more columns referenced do not exist in the data."
            else:
                assistant_response = f"Error: {e}"

            st.session_state.history.append({"role": "assistant", "content": assistant_response})

        # Reset user input
        st.session_state['user_input'] = ''

# Step 5: Display conversation history
st.write("## Conversation History")
for message in st.session_state.history:
    if message['role'] == 'user':
        st.markdown(f"**User:** {message['content']}")
    elif message['role'] == 'assistant':
        st.markdown(f"**Assistant:** {message['content']}")

# Input field
st.text_input("Enter your message:", key='user_input', on_change=process_input)
