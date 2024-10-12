  Business Data Insights Chatbot - README

Business Data Insights Chatbot: Automating SQL Generation & Insights Extraction
===============================================================================

Welcome to the Business Data Insights Chatbot, an application that automates the generation of SQL queries and extracts meaningful insights from your data. Powered by OpenAI's language models, this chatbot helps you interact with your datasets using natural language, making data analysis more intuitive and efficient.

Features
--------

*   Upload your CSV datasets or use the provided default dataset.
*   Automatic SQL generation and execution to answer your data-related questions.
*   Provides detailed insights, correlations, and recommendations based on the dataset.
*   Conversation history that tracks user queries and chatbot responses.

Setup Instructions
------------------

### Requirements

Before running the project, ensure you have the following Python packages installed:

        streamlit==1.21.0
        openai>=1.10.0,<2.0.0
        sqlparse==0.4.4
        langchain
        langchain-openai
        langchain-community
        faiss-cpu==1.7.3
        python-dotenv==1.0.0
        tiktoken==0.7.0
        sql-metadata==2.13.0
        tabulate
        scikit-learn==1.3.0
        statsmodels==0.14.0
    

### Running the Application

1.  Clone the repository from GitHub.
2.  Ensure the `default_data.csv` file is in the project directory (or upload your own CSV).
3.  Install the required packages using `pip install -r requirements.txt`.
4.  Set your OpenAI API key as an environment variable:
    
    export OPENAI\_API\_KEY='your-api-key'
    
5.  Run the Streamlit application:
    
    streamlit run your\_app.py
    
6.  Upload a CSV file or use the default dataset to start interacting with the chatbot.

Usage
-----

Once the application is running, you can upload your CSV data, or use the default dataset provided.

### Sample Questions to Ask the Chatbot:

*   Summarize the data for me.
*   Do you notice any correlations in the datasets?
*   Can you offer any recommendations based on the datasets?
*   Provide an analysis of some numbers across some categories.

Demo
----

You can try the demo of the Business Data Insights Chatbot by following this link: [AI Business Insights Chatbot Demo](https://www.arithescientist.com/genaibichatbot).

Files in the Repository
-----------------------

*   `app.py`: Main application code for the chatbot.
*   `requirements.txt`: List of required packages.
*   `default_data.csv`: Default dataset for demo purposes.

License
-------

This project is licensed under the MIT License.