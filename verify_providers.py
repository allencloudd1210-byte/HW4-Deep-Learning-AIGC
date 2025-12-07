import sys
try:
    import langchain_google_genai
    print("langchain_google_genai imported successfully")
    import langchain_openai
    print("langchain_openai imported successfully")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
