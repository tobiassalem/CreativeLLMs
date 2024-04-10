import os
# import schema for chat messages and ChatOpenAI in order to query chat models GPT-3.5-turbo or GPT-4
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# import classes for Document management
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader

# import for vector storage
from langchain_community.vectorstores import Chroma

# NB! Use full path to our directories since the script can be run from another location.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOC_DIR = BASE_DIR + "/docs"
DB_DIR = BASE_DIR + "/data"


# From the defined DOC_DOR, load and split documents, with chunk overlap for context precision.
def load_documents() -> list[Document]:
    documents = []
    for file in os.listdir('docs'):
        if file.endswith('.pdf'):
            pdf_path = DOC_DIR + '/' + file
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):
            doc_path = DOC_DIR + '/' + file
            loader = Docx2txtLoader(doc_path)
            documents.extend(loader.load())
        elif file.endswith('.txt') or file.endswith('md'):
            text_path = DOC_DIR + '/' + file
            loader = TextLoader(text_path, encoding='utf-8')
            documents.extend(loader.load())

        print(f"Loaded {file} from {DOC_DIR}. We now have {len(documents)} documents")

    print(f"Loaded {len(documents)} documents from {DOC_DIR}")

    # Split the documents into smaller chunks.
    # The chunk_overlap helps to give better results and contain the context of the information between chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(documents)
    print(f"CharacterTextSplitter complete. We now have {len(documents)} document chunks.")
    return documents


# Convert the given document chunks to embedding and save them to the vector store, in the defined DB_DIR
def persist_docs_in_vectorstore(documents):
    vector_db = Chroma.from_documents(
        documents,
        embedding=OpenAIEmbeddings(),
        persist_directory=DB_DIR
    )
    vector_db.persist()
    print(f"Persisted docs in vectorstore: {DB_DIR}")
    return vector_db


# Load documents from our vectorstore, in the defined DB_DIR
def load_docs_from_vectorstore():
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=OpenAIEmbeddings())
    print(f"Loaded docs from vectorstore: {DB_DIR}")
    return vector_db


# Return true if our vector store in DB_DIR exist and is non-empty, false otherwise.
# NB! The responses can act strange when you change the docs, chains and code setup without
# recreating the data in the vectorstore.
def vectorstore_exists():
    if len(os.listdir(DB_DIR)) == 0:
        print(f"DB Directory {DB_DIR} is empty. This is fine and expected on first run.")
        return False
    else:
        print(f"DB Directory {DB_DIR} is NOT empty. If you add/change docs it is recommended to clear it.")
        return True


def load_vectorstore():
    if vectorstore_exists():
        vectordb = load_docs_from_vectorstore()
    else:
        vectordb = persist_docs_in_vectorstore(load_documents())
    return vectordb


# Return true if our vector store in DB_DIR is empty, false otherwise.
# NB! The responses can act strange when you change the docs, chains and code setup without
# deleting the data created from the previous setups.
def is_vectorstore_empty() -> bool:
    if len(os.listdir(DB_DIR)) == 0:
        print(f"DB Directory {DB_DIR} is empty. This is fine and expected on first run.")
        return True
    else:
        print(f"DB Directory {DB_DIR} is NOT empty. If you add/change docs it is recommended to clear it.")
        return False


# Checks is the given answer is considered oblivious, i.e. the LLM does not know the answer.
def is_oblivious(given_answer) -> bool:
    oblivious_markers = ["I don't know", "I do not know", "don't have that information",
                         "I do not have that information", "cannot answer that", "Jag vet inte",
                         "Jag har ej information", "Jag kan ej härleda"]
    return any(marker in given_answer for marker in oblivious_markers)


# Defined a simple OpenAI agent to use as fallback when the answer cannot be found in our knowledge documents.
def my_agent(given_input) -> str | list[str | dict]:
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
    messages = [
        SystemMessage(content="You are a funny and helpful assistant."),
        AIMessage(content="I will do my best to find the true answer for you!"),
        HumanMessage(content=given_input),
    ]
    response = chat.invoke(messages)
    return response.content


# Test my function
if __name__ == '__main__':
    test_answers = ["I don't know.", "I do not know.", "I don't have that information...", "I cannot answer that",
                    "Sorry, I don't have that information.", "I really don't have that information", "I now know"]
    for answer in test_answers:
        print(f"Is oblivious: {answer} : " + str(is_oblivious(answer)))
