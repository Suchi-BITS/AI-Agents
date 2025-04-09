from llm_groq import get_groq_llm

llm = get_groq_llm()
response = llm.invoke("What is the capital of France?")
print(response)