import os
import sys

from colorama import Style
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# Load environment
load_dotenv()

# Load the documents, by file type
documents = []
for file in os.listdir('docs'):
    if file.endswith('.pdf'):
        pdf_path = './docs/' + file
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    elif file.endswith('.docx') or file.endswith('.doc'):
        doc_path = './docs/' + file
        loader = Docx2txtLoader(doc_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        text_path = './docs/' + file
        loader = TextLoader(text_path, encoding='utf-8')
        documents.extend(loader.load())

# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)

# Convert the document chunks to embedding and save them to the vector store
vectordb = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
    persist_directory='./data'
)
vectordb.persist()

# create our Q&A chain
qa_chain = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo'),
    retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
    return_source_documents=True,
    verbose=False
)

#
# import schema for chat messages and ChatOpenAI in order to query chatmodels GPT-3.5-turbo or GPT-4

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)


def my_agent(input):
    messages = [
        SystemMessage(content="You are a funny and helpful assistant."),
        HumanMessage(content=input),
    ]
    response = chat.invoke(messages)
    return response.content


def is_oblivious(answer):
    return answer == "I don't know." or "I don't have that information." or "I do not know."


# Chatbot setup
yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

# The chain run command accepts the chat_history as a parameter.
# So first of all, let’s enable a continuous conversation via the terminal by  nesting the stdin and stout commands
# inside a While loop. Next, we must manually build up this list based on our conversation with the LLM.
# The chain does not do this out of the box.
# So for each question and answer, we will build up a list called chat_history,
# which we will pass back into the chain run command each time.
chat_history = []
print(f"{yellow}")
print("---------------------------------------------------------------------------------")
print("Welcome to the DocBot. You are now ready to start interacting with your documents")
print("---------------------------------------------------------------------------------")
print(f"{Style.RESET_ALL}")
while True:
    # this prints to the terminal, and waits to accept an input from the user
    query = input('Prompt (q to quit): ')
    # give us a way to exit the script
    if query == "exit" or query == "quit" or query == "q":
        print('Exiting')
        sys.exit()
    # we pass in the query to the LLM, and print out the response. As well as
    # our query, the context of semantically relevant information from our
    # vector store will be passed in, as well as list of our chat history
    # TODO: if answer is not found in the documents - perform a normal OpenAI chat query.
    result = qa_chain.invoke({'question': query, 'chat_history': chat_history})
    answer = result['answer']
    if is_oblivious(answer):
        print("Sorry I cannot find the answer in the documents, but will access the AI network, hang on!")
        answer = my_agent(query)
    print(f"Answer: {answer}")
    # we build up the chat_history list, based on our question and response
    # from the LLM, and the script then returns to the start of the loop
    # and is again ready to accept user input.
    chat_history.append((query, result['answer']))
