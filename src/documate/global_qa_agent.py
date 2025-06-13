# src/documate/global_qa_agent.py

import os
from typing import Any, List, Dict

from langchain.retrievers import MergerRetriever
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class GlobalQAAgent:
    """
    An agent that can answer questions by searching across ALL available
    repositories (both code and wikis).
    """
    def __init__(self, chat_model: Any, embeddings: Any, vector_db_path: str):
        self.chat_model = chat_model
        self.embeddings = embeddings
        self.vector_db_path = vector_db_path
        print("GlobalQAAgent initialized.")

    def _get_all_retrievers(self) -> MergerRetriever | None:
        """
        Scans the vector store path and creates a merged retriever
        for every available code and wiki index.
        """
        all_retrievers = []
        repo_names = os.listdir(self.vector_db_path)

        for repo_name in repo_names:
            repo_base_path = os.path.join(self.vector_db_path, repo_name)
            if not os.path.isdir(repo_base_path):
                continue

            # Add code retriever if it exists
            code_index_path = os.path.join(repo_base_path, "code")
            if os.path.exists(code_index_path):
                code_store = Chroma(persist_directory=code_index_path, embedding_function=self.embeddings)
                # We add the repo_name to the metadata of each document before returning it
                retriever = code_store.as_retriever(search_kwargs={"k": 2})
                retriever.name = f"{repo_name} (Code)"
                all_retrievers.append(retriever)

            # Add wiki retriever if it exists
            wiki_index_path = os.path.join(repo_base_path, "wiki")
            if os.path.exists(wiki_index_path):
                wiki_store = Chroma(persist_directory=wiki_index_path, embedding_function=self.embeddings)
                retriever = wiki_store.as_retriever(search_kwargs={"k": 2})
                retriever.name = f"{repo_name} (Wiki)"
                all_retrievers.append(retriever)
        
        if not all_retrievers:
            return None

        print(f"Found and merged {len(all_retrievers)} retrievers for global search.")
        # The merger retriever will query all these retrievers in parallel
        return MergerRetriever(retrievers=all_retrievers)
    
    def _format_docs_for_global_search(self, docs: List[Document]) -> str:
        """
        Formats documents and explicitly states the repository of origin.
        """
        formatted_docs = []
        for doc in docs:
            # We need to find which retriever this doc came from.
            # This is a bit of a workaround as MergerRetriever doesn't add this metadata.
            # We infer it from the source path.
            source_path = doc.metadata.get("source", "Unknown Source")
            parts = source_path.split(os.sep)
            try:
                # Assuming path is like .../vector_stores/repo_name/code/...
                vstores_index = parts.index("vector_stores")
                repo_name = parts[vstores_index + 1]
            except ValueError:
                repo_name = "Unknown Repo"

            source_type = "Wiki Page" if ".md" in source_path else "Source Code"
            filename = os.path.basename(source_path)
            
            context_header = f"[From Repo: {repo_name}, Type: {source_type}, File: {filename}]"
            content = doc.page_content
            formatted_docs.append(f"--- START OF CONTEXT {context_header} ---\n{content}\n--- END OF CONTEXT ---")
        
        return "\n\n".join(formatted_docs)

    def get_answer(self, question: str, callbacks: List[Any] | None = None) -> Dict[str, Any]:
        retriever = self._get_all_retrievers()
        if not retriever:
            return {"result": "No repositories have been indexed yet. Please add a repository first."}

        prompt_template = """
        You are a top-tier AI assistant acting as a knowledge discovery engine for an entire organization's codebase.
        Your goal is to answer a user's question by finding the most relevant information from ANY available repository.

        **CRITICAL INSTRUCTIONS:**
        1.  Analyze the following context, which contains code and documentation from MULTIPLE different repositories.
        2.  You **MUST CITE** the repository and the specific file for every piece of information you use. The context provides this information in the format `[From Repo: <repo_name>, Type: <Source Code/Wiki Page>, File: <filename>]`.
        3.  Synthesize a single, cohesive answer. If multiple repositories have relevant information, combine them. For example: "In the `fastapi` repository, you can do X (see `[From Repo: fastapi, ...]`). Alternatively, the `tqdm` library offers Y for a similar purpose (see `[From Repo: tqdm, ...]`)"
        4.  If the most relevant information comes from a CodeWiki, prioritize that for the high-level explanation.
        5.  If you don't know the answer, state that clearly.

        **CONTEXT FROM ALL REPOSITORIES:**
        {context}

        **USER'S GLOBAL QUESTION:** {question}

        **COMPREHENSIVE, CITED ANSWER:**
        """
        prompt = PromptTemplate.from_template(prompt_template)
        
        rag_chain = (
            {"context": retriever | self._format_docs_for_global_search, "question": RunnablePassthrough()}
            | prompt
            | self.chat_model
            | StrOutputParser()
        )

        result = rag_chain.invoke(question, config={"callbacks": callbacks})
        return {"result": result}