import os
#from langchain_core.language_models import ChatModel
from langchain_groq import ChatGroq
from dotenv import load_dotenv 
load_dotenv() 

# Set your API key securely (consider using environment variables or a .env file in production)
groq_api_key = os.getenv("GROQ_API_KEY")  # Replace with your real key

# Instantiate the LLM
llm = ChatGroq(
    api_key=groq_api_key,
    temperature=0.3,
    model="llama3-8b-8192"
)
