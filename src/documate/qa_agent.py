# src/documate/qa_agent.py

import os
from typing import Any, List, Dict
from operator import itemgetter

from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

class QAAgent:
    """
    An agent that answers questions about a repository using a RAG pipeline.
    This version uses LCEL for more control and includes source file citations.
    """
    def __init__(self, chat_model: Any, embeddings: Any, vector_db_path: str):
        self.chat_model = chat_model
        self.embeddings = embeddings
        self.vector_db_path = vector_db_path
        print("QAAgent (LCEL-powered) initialized.")

    def _get_retriever(self, repo_name: str):
        persist_directory = os.path.join(self.vector_db_path, repo_name)
        if not os.path.exists(persist_directory):
            raise FileNotFoundError(f"Vector store for repository '{repo_name}' not found.")
        
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        # Retrieve more documents to give the LLM better context for citation.
        return vector_store.as_retriever(search_kwargs={"k": 8})

    def _format_docs_with_sources(self, docs: List[Document]) -> str:
        """
        Formats documents into a single string with clear source file markers.
        """
        formatted_docs = []
        for i, doc in enumerate(docs):
            # Extract the relative path from the full path in metadata
            source_path = doc.metadata.get("source", "Unknown Source")
            # We can try to make the path relative to the repo root if possible
            # This is a heuristic and might need adjustment based on repo structure
            try:
                # Assuming the path is like .../cloned_repos/repo_name/src/file.py
                parts = source_path.split(os.sep)
                repo_name_index = parts.index(os.path.basename(doc.metadata.get("repo_path", "")))
                relative_path = os.path.join(*parts[repo_name_index + 1:])
            except (ValueError, KeyError, TypeError):
                relative_path = source_path

            content = doc.page_content
            formatted_docs.append(f"--- START OF {relative_path} ---\n{content}\n--- END OF {relative_path} ---")
        
        return "\n\n".join(formatted_docs)

    def get_answer(self, question: str, repo_name: str, callbacks: List[BaseCallbackHandler] | None = None) -> Dict[str, Any]:
        print(f"Querying repository '{repo_name}' with question: '{question}'")
        
        retriever = self._get_retriever(repo_name)
        repo_path = os.path.join("cloned_repos", repo_name) # Heuristic for getting repo path

        # This is our new, more explicit prompt template
        prompt_template = """
        You are an expert AI assistant for the codebase: **{repo_name}**.
        Your goal is to provide helpful, accurate, and detailed answers to a developer's questions.

        Use the following context, which consists of code snippets from different files, to answer the user's question.

        **CRITICAL INSTRUCTIONS:**
        1.  When you provide code snippets or refer to specific logic, you **MUST CITE** the source file path. The file path is provided in the context markers (e.g., `--- START OF src/main.py ---`).
        2.  Cite files directly in your answer, for example: "As seen in `src/utils/helpers.py`, the `format_data` function...".
        3.  If the user asks for a high-level overview, synthesize information from multiple files and cite them all.
        4.  If you don't know the answer or the context is insufficient, clearly state that you couldn't find the information in the provided files. DO NOT make up answers.

        **CONTEXT FROM THE CODEBASE:**
        {context}

        **QUESTION:**
        {question}

        **DETAILED AND CITED ANSWER:**
        """
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question", "repo_name"]
        )

        # We inject the repo_path into each document's metadata before formatting
        def add_repo_path_to_docs(docs: List[Document]) -> List[Document]:
            for doc in docs:
                doc.metadata["repo_path"] = repo_path
            return docs

        # This is the LCEL (LangChain Expression Language) chain
        rag_chain = (
            {
                "context": retriever | RunnableLambda(add_repo_path_to_docs) | RunnableLambda(self._format_docs_with_sources),
                "question": RunnablePassthrough(),
                "repo_name": lambda x: repo_name
            }
            | prompt
            | self.chat_model
            | StrOutputParser()
        )

        # We pass the original question straight into the chain
        result = rag_chain.invoke(question, config={"callbacks": callbacks})
        
        # The LCEL chain just returns the final string, so we package it
        # for compatibility with our existing UI.
        return {"result": result}