import os

from dotenv import load_dotenv
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain_openai import OpenAI

load_dotenv()
huggingface_api_key = os.getenv("HUGGINFACE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
serpapi_api_key = os.getenv("SERP_API_KEY")

llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

toolkit = load_tools(["serpapi"], llm=llm, serpapi_api_key=serpapi_api_key)
agent = initialize_agent(toolkit, llm, agent="zero-shot-react-description", verbose=True, return_intermediate_steps=True)

# Define questions that can only be answered by Train of Thought, and access to the toolkit.
queries = ["What was the first album of the band that Natalie Bergman is a part of?",
           "What is the current president of the country where JFK once was president?",
           "How far is it between earth and the largest planet in the solar system?"]

# Run the queries in a chain, with access to the toolkit.
for q in queries:
    response = agent({"input": q})
    print(f"Question: {q}")
    print(f"Final Answer: {response["output"]}")
