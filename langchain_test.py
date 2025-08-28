# --- Install required packages ---
# pip install langchain openai faiss-cpu chromadb tiktoken
import warnings
warnings.filterwarnings("ignore")
import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA



# Set your OpenAI API key
os.environ["GROQ_API_KEY"] = "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"

# Step 1: Prepare Telecom Knowledge Base (documents, FAQs, chat logs)
telecom_docs = [
    Document(page_content="How to activate SIM card? To activate, dial *123# or call customer care."),
    Document(page_content="What are roaming charges? Incoming calls cost Rs.1/min, outgoing Rs.1.5/min."),
    Document(page_content="How to check data balance? Send SMS 'DATA' to 199 or use the mobile app."),
    Document(page_content="How to port number? Send SMS 'PORT <MobileNumber>' to 1900."),]


# Step 2: Split into chunks (if needed for large docs)
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = splitter.split_documents(telecom_docs)


# Step 3: Embed & Index using Chroma
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Step 4: Store in FAISS
#vectorstore = FAISS.from_documents(docs, embeddings)
from langchain_community.vectorstores import Chroma
# Store in Chroma
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")

# Step 5: Groq LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# Step 6: Retrieval QA
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Step 7: Ask
#query = "How do I activate my new SIM?"
import sys
while True:
    print("Hi....:-")
    query = input("Enter the query (type 'Exit' to quit): ")
    
    if query.lower() == "exit":
        print("Exiting program...")
        break  # Exit the loop and program
    
    if len(query) < 2:
        print("No input provided.")
    else:
        # Assuming qa_chain is already defined
        result = qa_chain({"query": query})
        print("Answer:", result["result"])
