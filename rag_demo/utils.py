from langchain.prompts import PromptTemplate

PRESCRIPTION_TEMPLATE = """
You are a compassionate and professional mental health assistant. 
Based on the following conversation history between a user and a mental health robot, 
please generate a "Mental Health Prescription" card.

Conversation History:
{chat_history}

The prescription should include:
1.  **Patient's Main Concern**: A brief summary of what the user is struggling with.
2.  **Doctor's Advice**: A warm, encouraging, and actionable piece of advice based on the context.
3.  **Recommended Actions**: 1-3 simple steps the user can take (e.g., deep breathing, talking to a friend).
4.  **Encouragement**: A short uplifting quote or message.

Format the output as a clean Markdown card.
"""

def get_prescription_prompt():
    return PromptTemplate(
        input_variables=["chat_history"],
        template=PRESCRIPTION_TEMPLATE
    )

def format_prescription(text):
    # Optional: Add HTML styling if needed, for now return raw markdown
    return text
