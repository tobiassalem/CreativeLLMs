import os

from dotenv import load_dotenv
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain_openai import OpenAI

load_dotenv()
huggingface_api_key = os.getenv("HUGGINFACE_API_KEY")
openai_api_key = os.getenv("OPEN_AI_API_KEY")
serpapi_api_key = os.getenv("SERP_API_KEY")

llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

toolkit = load_tools(["serpapi"], llm=llm, serpapi_api_key=serpapi_api_key)
agent = initialize_agent(toolkit, llm, agent="zero-shot-react-description", verbose=True, return_intermediate_steps=True)
response = agent({"input": "What was the first album of the band that Natalie Bergman is a part of?"})