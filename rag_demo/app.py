import streamlit as st
import os
import tempfile
from rag_core import RAGSystem
from utils import get_prescription_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

st.set_page_config(page_title="Mental Health RAG Robot", page_icon="🤖", layout="wide")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_system" not in st.session_state:
    st.session_state.rag_system = None

if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    llm_provider = st.selectbox("Select LLM Provider", ["ollama", "google", "openai"])
    
    api_key = None
    model_name = "llama3"
    
    if llm_provider == "ollama":
        st.markdown("### Local LLM (Ollama)")
        model_name = st.text_input("Model Name", value="llama3")
        
        # Check Ollama Status
        import requests
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=1)
            if response.status_code == 200:
                st.success("🟢 Ollama is running")
            else:
                st.warning("🟡 Ollama reachable but returned error")
        except:
            st.error("🔴 Ollama NOT running")
            st.info("Run `ollama serve` in terminal")
    else:
        api_key = st.text_input(f"{llm_provider.capitalize()} API Key", type="password")
        if not api_key:
            st.warning(f"Please enter your {llm_provider} API Key")

    st.divider()
    
    st.subheader("📚 Knowledge Base")
    uploaded_files = st.file_uploader("Upload Documents (PDF, TXT, DOCX)", accept_multiple_files=True)
    
    if st.button("Build/Update Vector DB"):
        if not uploaded_files:
            st.warning("Please upload files first.")
        else:
            with st.spinner("Initializing AI Engine & Processing documents..."):
                # Save uploaded files to temp dir
                temp_dir = tempfile.mkdtemp()
                file_paths = []
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)
                
                # Initialize RAG System (Lazy Load)
                RAGSystemClass = get_rag_system()
                st.session_state.rag_system = RAGSystemClass()
                documents = st.session_state.rag_system.load_documents(file_paths)
                st.session_state.rag_system.create_vector_store(documents)
                st.success(f"Processed {len(documents)} document chunks!")
                
    if st.button("Clear Knowledge Base"):
        if st.session_state.rag_system:
            st.session_state.rag_system.clear_vector_store()
            st.session_state.conversation_chain = None
            st.success("Knowledge base cleared.")

# Main Chat Interface
st.title("🤖 Mental Health Prescription Robot")
st.markdown("Chat with me about your worries, and I'll try to help based on the knowledge base.")

# Initialize Chain if ready
if st.session_state.rag_system and (api_key or llm_provider == "ollama") and not st.session_state.conversation_chain:
    try:
        st.session_state.conversation_chain = st.session_state.rag_system.get_retrieval_chain(
            llm_api_key=api_key,
            llm_provider=llm_provider,
            model_name=model_name
        )
    except Exception as e:
        st.error(f"Failed to initialize chain: {e}")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("How are you feeling today?"):
    if llm_provider != "ollama" and not api_key:
        st.error("Please enter your API Key in the sidebar.")
    elif not st.session_state.conversation_chain:
        st.warning("Please build the knowledge base first.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.conversation_chain.invoke({"question": prompt})
                    answer = response['answer']
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {e}")
                    if llm_provider == "ollama" and "Connection refused" in str(e):
                        st.error("💡 Tip: Is Ollama running? Try running `ollama serve` in your terminal.")

# Prescription Feature
st.divider()
if st.button("💊 Generate Mental Health Prescription"):
    if not st.session_state.messages:
        st.warning("No conversation history to analyze.")
    elif llm_provider != "ollama" and not api_key:
        st.error("Please enter your API Key.")
    else:
        with st.spinner("Writing prescription..."):
            # Prepare chat history string
            history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
            
            # Initialize LLM for prescription
            try:
                if llm_provider == "google":
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
                elif llm_provider == "openai":
                    from langchain_openai import ChatOpenAI
                    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)
                elif llm_provider == "ollama":
                    from langchain_ollama import ChatOllama
                    llm = ChatOllama(model=model_name)
                    
                prescription_chain = LLMChain(llm=llm, prompt=get_prescription_prompt())
                prescription = prescription_chain.run(chat_history=history_text)
                
                st.markdown("### 📝 Your Prescription")
                st.markdown(prescription)
            except Exception as e:
                st.error(f"Failed to generate prescription: {e}")
                if llm_provider == "ollama" and "Connection refused" in str(e):
                    st.error("💡 Tip: Is Ollama running? Try running `ollama serve` in your terminal.")
