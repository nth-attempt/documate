# src/documate/callbacks/streamlit_callback.py

from typing import Any, Dict, List
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

class StreamlitCallbackHandler(BaseCallbackHandler):
    """
    A custom LangChain callback handler that streams LLM outputs to a Streamlit container.
    """
    def __init__(self, container, initial_text=""):
        """
        Initializes the callback handler.

        Args:
            container: The Streamlit container (e.g., st.empty()) to write the output to.
            initial_text: Initial text to display in the container.
        """
        self.container = container
        self.text = initial_text
        self.container.markdown(self.text)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """
        Appends the new token to the text and updates the container.
        """
        self.text += token
        self.container.markdown(self.text)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """
        Actions to take when the LLM finishes generating.
        (Could be used for final formatting, but not needed for basic streaming.)
        """
        pass

    def on_llm_error(self, error: Exception | KeyboardInterrupt, **kwargs: Any) -> None:
        """
        Displays an error message in the container if the LLM fails.
        """
        self.container.error(f"LLM Error: {error}")