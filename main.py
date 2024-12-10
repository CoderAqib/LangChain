from getpass import getpass
import streamlit as st
import time
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()


def get_api_key():
    # Get the API key from environment variables or user input
    try:
        return os.getenv("GOOGLE_API_KEY")
    except Exception:
        return getpass("Please enter your Google API key:")


st.title("News Research Tool")
st.sidebar.title("News Article URL")

# Accept a single URL
url = st.sidebar.text_input("Enter URL")

process_url_clicked = st.sidebar.button("Process URL")

file_path = "faiss_store_vectors.pkl"
main_placeholder = st.empty()
llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.9, max_tokens=500)

if process_url_clicked:
    # Ensure URL is provided
    if not url:
        st.error("Please enter a valid URL.")
    else:
        # Load data
        loader = UnstructuredURLLoader(urls=[url])
        main_placeholder.text("Data loading... started")
        data = loader.load()

        # Split data
        text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
        main_placeholder.text("Text splitter... started")
        docs = text_splitter.split_documents(data)

        # Create embeddings and save them to FAISS index
        api_key = get_api_key()
        if not api_key:
            st.error("API key not found. Please check your environment variables.")
            exit()

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=api_key)
        vectors = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Vectors embedding started...")
        time.sleep(2)

        # Save FAISS index
        vectors.save_local("faiss_index")

query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists("faiss_index"):
        # Recreate embeddings for retriever
        api_key = get_api_key()
        if not api_key:
            st.error("API key not found. Please check your environment variables.")
            exit()

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=api_key)

        # Load FAISS index from local file
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        # Create the retrieval chain
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

        # Run the query
        result = chain({"question": query}, return_only_outputs=True)

        # Display the answer
        st.header("Answer")
        st.write(result["answer"])

        # Display sources if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources: ")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)
