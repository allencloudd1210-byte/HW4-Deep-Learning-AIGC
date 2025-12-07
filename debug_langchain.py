import sys
try:
    import langchain
    print(f"LangChain version: {langchain.__version__}")
    print(f"LangChain file: {langchain.__file__}")
    
    import langchain.chains
    print("langchain.chains imported successfully")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
