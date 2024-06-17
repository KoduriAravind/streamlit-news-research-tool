%%writefile app.py
import os
import streamlit as st
import pickle
import time
import langchain
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

# Set up Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
llm = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", model_kwargs={"temperature": 0.2, "max_length": 512})

# Streamlit UI setup
st.title("News Research Tool")
st.sidebar.title("News Article URLs")

# Input URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_index.pkl"
main_placeholder = st.empty()

if process_url_clicked:
    # Load data from URLs
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading... started...")
    data = loader.load()

    # Split data into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', '!', '?', ',', ' '],
        chunk_size=512,  # Adjust chunk size to fit within model's max context length
        chunk_overlap=50  # Overlap to maintain context across chunks
    )
    main_placeholder.text("Text splitting... started...")

    try:
        docs = text_splitter.split_documents(data)
    except ValueError as e:
        st.error(f"Error during text splitting: {e}")
        st.stop()

    # Create embeddings and build FAISS index
    embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding vector... started...")
    time.sleep(2)

    # Save the vectorstore for later use
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

    st.success("URLs processed and vectorstore created successfully!")

# Input query for processing
query = main_placeholder.text_input("Enter your query")

if query and os.path.exists(file_path):
    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)

    # Create retrieval chain
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
    result = chain({"question": query}, return_only_outputs=True)

    # Display results
    st.header("Answer")
    st.write(result["answer"])

    # Display sources
    st.subheader("Sources")
    st.write(result["sources"])
