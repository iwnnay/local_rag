from langchain_community.document_loaders import UnstructuredPDFLoader
import os


pdf_file = "./data/pca_tutorial.pdf"
embedding_model_name = "nomic-embed-text"
llm_model_name = "llama3.2"


# Load and Ingest a PDF
if os.path.exists(pdf_file):
    pdf_loader = UnstructuredPDFLoader(file_path=pdf_file)
    documents = pdf_loader.load()
    print("[✓] PDF file loaded successfully.")
else:
    raise FileNotFoundError("PDF file not found. Please upload a valid file.")


# Split the PDF into Chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter


splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
doc_chunks = splitter.split_documents(documents)
print(f"[✓] Document split into {len(doc_chunks)} chunks.")


# Embed Chunks into Vector Database
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import ollama


print("Pulling embedding model...")
ollama.pull(embedding_model_name)


vector_store = Chroma.from_documents(
    documents=doc_chunks,
    embedding=OllamaEmbeddings(model=embedding_model_name),
    collection_name="pca_tutorial_collection",
)
print("[✓] Chunks embedded and stored in vector database.")


#  Setup the Retriever with Multi-Query
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough



llm = ChatOllama(model=llm_model_name)



multi_query_prompt = PromptTemplate(
    input_variables=["question"],
    template=(
        "You are an AI assistant. Generate 5 different versions of the user question "
        "to improve document retrieval from a vector database.\n"
        "Original question: {question}"
    ),
)



retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(),
    llm=llm,
    prompt=multi_query_prompt,
)


print("[✓] Retriever with multi-query setup complete.")


# Ask Questions with RAG Pipeline
from langchain_core.output_parsers import StrOutputParser


context_prompt = ChatPromptTemplate.from_template(
    "Answer the question using ONLY the context below:\n{context}\n\nQuestion: {question}"
)


rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | context_prompt
    | llm
    | StrOutputParser()
)


# === Example Query ===
user_question = "What is the goal of PCA?"
response = rag_chain.invoke(user_question)


print("\n=== RAG Response ===\n")
print(response)
