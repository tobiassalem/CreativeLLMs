import sys

from colorama import Style
from dotenv import load_dotenv
from langchain.chains import (
    StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
)
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

from my_helpers import load_vectorstore, is_oblivious, my_agent

# Load environment
load_dotenv()


# Create our Q&A chain
# qa_chain = ConversationalRetrievalChain.from_llm(
#     ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo'),
#     retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
#     return_source_documents=True,
#     verbose=False
# )

# combine_docs_chain = StuffDocumentsChain(...)
# This controls how each document will be formatted. Specifically,
# it will be passed to `format_document` - see that function for more details.
document_prompt = PromptTemplate(
    input_variables=["page_content"],
    template="{page_content}"
)
document_variable_name = "context"
llm = OpenAI()
# The prompt here should take as an input variable the `document_variable_name`
summarize_prompt = PromptTemplate.from_template(
    "Summarize this content: {context}"
)
llm_chain = LLMChain(llm=llm, prompt=summarize_prompt)
combine_docs_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_prompt=document_prompt,
    document_variable_name=document_variable_name
)

# This controls how the standalone question is generated.
# Should take `chat_history` and `question` as input variables.
combine_template = (
    "Combine the chat history and follow up question into "
    "a standalone question. Chat History: {chat_history}"
    "Follow up question: {question}"
)
combine_prompt = PromptTemplate.from_template(combine_template)
question_generator_chain = LLMChain(llm=llm, prompt=combine_prompt)

# Load our vectorstore (with our parsed, chunked and embedded knowledge documents).
vectorstore = load_vectorstore()

# Create our Q&A chain: given the vectorstore, combine docs chain, and question generator chain
# ConversationalRetrievalChain: uses input keys "question", "chat_history", and output key: "answer"
qa_chain = ConversationalRetrievalChain(
    combine_docs_chain=combine_docs_chain,
    retriever=vectorstore.as_retriever(),
    question_generator=question_generator_chain,
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
print("---------------------------------------------------------------------------------------------")
print("Welcome to the summarizing DocBot. You are now ready to start interacting with your documents")
print("---------------------------------------------------------------------------------------------")
print(f"{Style.RESET_ALL}")
while True:
    # This prints to the terminal, and waits to accept an input from the user (with a way to exit).
    query = input(f"{green}Prompt (q to quit): ")
    if query == "exit" or query == "quit" or query == "q":
        print('Ok, exiting')
        sys.exit()
    # we pass in the query to the LLM, and print out the response. As well as
    # our query, the context of semantically relevant information from our
    # vector store will be passed in, as well as list of our chat history
    result = qa_chain.invoke({'question': query, 'chat_history': chat_history})
    answer = result['answer']

    # If an answer is not found in the documents - perform a normal OpenAI query.
    if is_oblivious(answer):
        print(f"{white}Sorry I cannot find the answer in the documents, but will access the AI network, hang on!")
        answer = my_agent(query)
    print(f"{yellow}Answer: {answer}")
    # We build up the chat_history list, based on our question and response from the LLM,
    # and the script then returns to the start of the loop, and is again ready to accept user input.
    chat_history.append((query, answer))
