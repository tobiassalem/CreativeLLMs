import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI

# Env setup
load_dotenv()
huggingface_api_key = os.getenv("HUGGINFACE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
serpapi_api_key = os.getenv("SERP_API_KEY")

# Style setup
yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

# Load our document
loader = TextLoader("docs/personal_knowledge.txt")
documents = loader.load()

# Specify OpenAI as the LLM that we want to use in our chain
openAI = OpenAI(temperature=0, openai_api_key=openai_api_key)
qa_chain = load_qa_chain(llm=openAI)

# Define questions that can only be answered by accessing the personal document
queries = ["Hej, vad heter det äldsta huset på Mölntorps Gård, och när byggdes det?",
           "Vad heter det speciella huset på Mölntorps Gård?",
           "Vad kallas huset som Magnus och Lotta bor i?",
           "Vad är min fästmös smeknamn?",
           "Vad är min äldsta brother-in-laws smeknamn?",
           "Vad är min yngsta brother-in-laws smeknamn?",
           "Vad är huvudstaden i Frankrike?"]

# Run the queries in a chain, with access to the personal knowledge document.
for q in queries:
    response = qa_chain.invoke({"input_documents": documents, "question": q})
    print(f"{green}Question: {q}")
    print(f"{yellow}Response: {white}{response["output_text"]}")
