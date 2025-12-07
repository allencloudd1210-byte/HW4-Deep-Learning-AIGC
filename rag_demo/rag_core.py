import os
import shutil
from typing import List, Optional

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

class RAGSystem:
    def __init__(self, 
                 persist_directory: str = "faiss_index",
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        
        self.persist_directory = persist_directory
        # Default to HF embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
            
        self.vector_store = None
        self.load_vector_store()

    def load_documents(self, file_paths: List[str]):
        documents = []
        for file_path in file_paths:
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith(".txt"):
                loader = TextLoader(file_path)
            elif file_path.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(file_path)
            else:
                continue
            documents.extend(loader.load())
        return documents

    def create_vector_store(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        if self.vector_store:
            self.vector_store.add_documents(texts)
        else:
            self.vector_store = FAISS.from_documents(texts, self.embeddings)
            
        self.vector_store.save_local(self.persist_directory)
        return self.vector_store

    def load_vector_store(self):
        if os.path.exists(self.persist_directory):
            try:
                self.vector_store = FAISS.load_local(self.persist_directory, self.embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                print(f"Error loading vector store: {e}")
                self.vector_store = None

    def clear_vector_store(self):
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        self.vector_store = None

    def get_retrieval_chain(self, llm_api_key: Optional[str] = None, llm_provider: str = "ollama", model_name: str = "llama3"):
        if not self.vector_store:
            return None
            
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        if llm_provider == "google":
            if not llm_api_key:
                raise ValueError("API Key required for Google")
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=llm_api_key, temperature=0.7)
        elif llm_provider == "openai":
            if not llm_api_key:
                raise ValueError("API Key required for OpenAI")
            llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=llm_api_key, temperature=0.7)
        elif llm_provider == "ollama":
            llm = ChatOllama(model=model_name, temperature=0.7)
        else:
            raise ValueError("Unsupported LLM provider")

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            verbose=True
        )
        return chain
