import sys
try:
    from langchain_ollama import ChatOllama
    print("ChatOllama imported successfully")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
