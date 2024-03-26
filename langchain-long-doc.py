# Import and load environment
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

load_dotenv()

# Style setup
yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

# Load the document
loader = PyPDFLoader('./docs/RachelGreenCV.pdf')
documents = loader.load()

# Split the data into chunks of 1,000 characters, with an overlap
# of 200 characters between the chunks, which helps to give better results
# and contain the context of the information between chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)

# Create our vectorDB, using the OpenAIEmbeddings transformer to create
# embeddings from our text chunks. We set all the db information to be stored
# inside the ./data directory, so it doesn't clutter up our source files
vectordb = Chroma.from_documents(
  documents,
  embedding=OpenAIEmbeddings(),
  persist_directory='./data_one'
)
vectordb.persist()

# Set up our QA chain. RetrievalQA: uses input key: "query", output key: "result"
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=vectordb.as_retriever(search_kwargs={'k': 4}),
    return_source_documents=True
)

# We can now execute queries against our Q&A chain
# Define questions that can only be answered by accessing the personal document
queries = ["Who is the CV about?",
           "At what university did she study?",
           "What is the Dissertation title?",
           "Give examples of some Honors and Awards?",
           "What is the capital of France?"]

# Run the queries in a chain, with access to the personal knowledge document.
for q in queries:
    response = qa_chain.invoke({"query": q})
    print(f"{green}Question: {q}")
    print(f"{yellow}Response: {white}{response["result"]}")

