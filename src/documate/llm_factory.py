# src/documate/llm_factory.py

import os
from typing import Any

def get_chat_model() -> Any:
    """
    Factory function to get the appropriate chat model based on environment variables.
    
    Returns:
        An instance of a chat model (e.g., ChatGoogleGenerativeAI) configured for streaming.
    
    Raises:
        ValueError: If the provider is not supported or required keys are missing.
    """
    provider = os.getenv("CHAT_PROVIDER", "google").lower()
    print(f"Chat provider selected: {provider}")

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY is not set in the environment.")
        
        # Gemini Pro is a good general-purpose model
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite", 
            google_api_key=google_api_key,
            temperature=0.1,
            convert_system_message_to_human=True # Helps with some prompt formats
        )

    elif provider == "azure":
        raise NotImplementedError("Azure chat provider is not yet implemented.")

    else:
        raise ValueError(f"Unsupported chat provider: '{provider}'.")