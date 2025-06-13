import os
from typing import Any

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
        # Placeholder for your future Azure setup.
        raise NotImplementedError("Azure provider is configured but not yet fully implemented.")

    elif provider == "openai":
        # Placeholder for standard OpenAI
        raise NotImplementedError("OpenAI provider is configured but not yet fully implemented.")

    else:
        raise ValueError(f"Unsupported embedding provider: '{provider}'. Please use 'google', 'azure', or 'openai'.")