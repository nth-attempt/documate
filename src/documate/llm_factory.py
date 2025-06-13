# src/documate/llm_factory.py

import os
from typing import Any
from .azure_auth import AzureADTokenManager

def get_chat_model() -> Any:
    """
    Factory function to get the appropriate chat model based on environment variables.
    """
    provider = os.getenv("CHAT_PROVIDER", "google").lower()
    print(f"Chat provider selected: {provider}")

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY is not set in the environment.")
        return ChatGoogleGenerativeAI(
            model="gemini-pro", google_api_key=google_api_key,
            temperature=0.1, convert_system_message_to_human=True
        )

    elif provider == "azure":
        # --- NEW SECTION FOR AZURE ---
        from langchain_openai import AzureChatOpenAI

        # Create a single token manager instance
        token_manager = AzureADTokenManager()

        return AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT_NAME"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            # LangChain will call this function to get a token
            azure_ad_token_provider=token_manager.get_token,
            temperature=0.1
        )

    else:
        raise ValueError(f"Unsupported chat provider: '{provider}'.")