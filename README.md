# Mental Health Prescription Robot (心靈處方籤機器人)

A local RAG (Retrieval-Augmented Generation) application built with Streamlit and Ollama. This application allows users to build a personal knowledge base from their own documents and chat with a "Mental Health Robot" that can generate personalized advice.

## Features

-   **Local Privacy**: Runs entirely locally using **Ollama** (no API keys required).
-   **RAG Technology**: Retrieves relevant information from your uploaded documents (PDF, TXT, DOCX).
-   **Interactive Chat**: Conversational interface with memory.
-   **Prescription Generator**: Generates a structured "Mental Health Prescription" based on the conversation.

## Prerequisites

1.  **Python 3.8+**
2.  **Ollama**: Download and install from [ollama.com](https://ollama.com/).
3.  **Llama 3 Model**: Run `ollama pull llama3` in your terminal.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/allencloudd1210-byte/HW4-Deep-Learning-AIGC.git
    cd HW4-Deep-Learning-AIGC
    ```

2.  Install dependencies:
    ```bash
    pip install -r rag_demo/requirements.txt
    ```

## Usage

1.  **Start Ollama Service**:
    Make sure Ollama is running in the background.
    ```bash
    ollama serve
    ```

2.  **Run the Application**:
    ```bash
    streamlit run rag_demo/app.py
    ```

3.  **In the App**:
    -   **Configuration**: Enter "llama3" (or your downloaded model name) in the sidebar.
    -   **Upload**: Upload documents from the `sample_docs/` folder or your own files.
    -   **Build DB**: Click "Build/Update Vector DB".
    -   **Chat**: Start chatting!
    -   **Prescription**: Click the button to generate a summary advice card.

## Project Structure

-   `rag_demo/`: Source code.
    -   `app.py`: Main Streamlit application.
    -   `rag_core.py`: RAG system logic.
    -   `utils.py`: Helper functions.
-   `sample_docs/`: Sample text files for testing.
