import os
import sys

from colorama import Style
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from my_helpers import load_vectorstore, is_oblivious, my_agent

# Load environment
load_dotenv()

# Load our vectorstore (with our parsed, chunked and embedded knowledge documents).
vectorstore = load_vectorstore()

# Create our retrieval Q&A chain

# ConversationalRetrievalChain: uses input keys "question", "chat_history", and output key: "answer"
# Oddly enough fails to answer "What was my last question?" Should it not be able to?
# Ref. https://stackoverflow.com/questions/76722077/why-doesnt-langchain-conversationalretrievalchain-remember-the-chat-history-ev
qa_chain = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo'),
    retriever=vectorstore.as_retriever(search_kwargs={'k': 6}),
    return_source_documents=True,
    verbose=False
)

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
print("------------------------------------------------------------------------------------------------")
print("Welcome to the conversational DocBot. You are now ready to start interacting with your documents")
print("------------------------------------------------------------------------------------------------")
print(f"{Style.RESET_ALL}")
while True:
    # This prints to the terminal, and waits to accept an input from the user (with a way to exit).
    question = input(f"{green}Prompt (q to quit): ")
    if question == "exit" or question == "quit" or question == "q":
        print('Ok, exiting')
        sys.exit()
    # we pass in the query to the LLM, and print out the response. As well as
    # our query, the context of semantically relevant information from our
    # vector store will be passed in, as well as list of our chat history
    result = qa_chain.invoke({'question': question, 'chat_history': chat_history})
    answer = result['answer']

    # If answer is not found in the documents - perform a normal OpenAI chat query.
    if is_oblivious(answer):
        print(f"{yellow}Sorry I cannot find the answer in the documents, but will access the AI network, hang on!")
        answer = my_agent(question)
    print(f"{yellow}Answer: {answer}")
    # We build up the chat_history list, based on our question and response from the LLM,
    # and the script then returns to the start of the loop, and is again ready to accept user input.
    chat_history.append((question, answer))
