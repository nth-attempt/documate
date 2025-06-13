# src/documate/embeddings_factory.py

import os
from typing import Any
from .azure_auth import AzureADTokenManager

def get_embedding_model() -> Any:
    """
    Factory function to get the appropriate embedding model based on environment variables.
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "google").lower()
    print(f"Embedding provider selected: {provider}")

    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY is not set in the environment.")
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

    elif provider == "azure":
        # --- NEW SECTION FOR AZURE ---
        from langchain_openai import AzureOpenAIEmbeddings

        # Create a single token manager instance
        token_manager = AzureADTokenManager()

        return AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            # LangChain will call this function to get a token
            azure_ad_token_provider=token_manager.get_token,
        )

    else:
        raise ValueError(f"Unsupported embedding provider: '{provider}'. Please use 'google' or 'azure'.")