import os
#from langchain_core.language_models import ChatModel
from langchain_groq import ChatGroq 
from langchain_core.language_models.llms import LLM
from dotenv import load_dotenv 
load_dotenv() 


# Set your API key securely (consider using environment variables or a .env file in production)
groq_api_key = os.getenv("GROQ_API_KEY")  # Replace with your real key

# Instantiate the LLM
def get_groq_llm() -> LLM:
    """
    Initializes and returns the Groq LLM using LLaMA3-8B model.
    Make sure your GROQ_API_KEY is set in the environment.
    """
    return ChatGroq(model="llama3-8b-8192", api_key=groq_api_key)
