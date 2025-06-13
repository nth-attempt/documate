import os
from typing import Any, List, Dict
from langchain.retrievers import MergerRetriever
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class QAAgent:
    """
    An agent that answers questions using a hybrid RAG pipeline,
    searching over both source code and generated wiki documentation.
    """
    def __init__(self, chat_model: Any, embeddings: Any, vector_db_path: str):
        self.chat_model = chat_model
        self.embeddings = embeddings
        self.vector_db_path = vector_db_path
        print("QAAgent (Hybrid RAG) initialized.")

    def _get_retrievers(self, repo_name: str) -> MergerRetriever:
        """Creates and merges retrievers for code and wiki vector stores."""
        code_index_path = os.path.join(self.vector_db_path, repo_name, "code")
        wiki_index_path = os.path.join(self.vector_db_path, repo_name, "wiki")

        retrievers = []
        if os.path.exists(code_index_path):
            code_store = Chroma(persist_directory=code_index_path, embedding_function=self.embeddings)
            retrievers.append(code_store.as_retriever(search_kwargs={"k": 5}))
            print("Code retriever loaded.")
        
        if os.path.exists(wiki_index_path):
            wiki_store = Chroma(persist_directory=wiki_index_path, embedding_function=self.embeddings)
            retrievers.append(wiki_store.as_retriever(search_kwargs={"k": 3}))
            print("Wiki retriever loaded.")

        if not retrievers:
            raise FileNotFoundError(f"No vector stores found for repository '{repo_name}'.")

        # The merger retriever runs all retrievers and combines the results.
        return MergerRetriever(retrievers=retrievers)

    def _format_docs_with_sources(self, docs: List[Document]) -> str:
        """Formats docs with clear source markers (code vs. wiki)."""
        formatted_docs = []
        for doc in docs:
            source_path = doc.metadata.get("source", "Unknown Source")
            # Determine if it's a code or wiki source
            source_type = "Wiki Page" if ".md" in source_path else "Source Code"
            
            # Use just the filename for cleaner citations
            filename = os.path.basename(source_path)
            content = doc.page_content
            formatted_docs.append(f"--- START OF CONTEXT FROM: [{source_type}: {filename}] ---\n{content}\n--- END OF CONTEXT ---")
        
        return "\n\n".join(formatted_docs)

    def get_answer(self, question: str, repo_name: str, callbacks: List[Any] | None = None) -> Dict[str, Any]:
        retriever = self._get_retrievers(repo_name)
        
        prompt_template = """
        You are an expert AI assistant for the codebase: **{repo_name}**.
        Your goal is to provide helpful, accurate answers by synthesizing information from TWO sources: the raw source code and the human-written CodeWiki documentation.

        **CRITICAL INSTRUCTIONS:**
        1.  Analyze the following context, which contains chunks from both source code and wiki pages.
        2.  You **MUST CITE** the source of your information using the markers provided in the context, for example: `[Source Code: main.py]` or `[Wiki Page: 01_Introduction.md]`.
        3.  Prioritize information from the CodeWiki for high-level explanations and overviews.
        4.  Use information from the Source Code for specific implementation details, code snippets, and function definitions.
        5.  Combine information from both sources for the most comprehensive answer. If they conflict, mention it.
        6.  If you don't know the answer, state that clearly. DO NOT make up answers.

        **CONTEXT FROM REPOSITORY:**
        {context}

        **QUESTION:** {question}

        **DETAILED AND CITED ANSWER:**
        """
        prompt = PromptTemplate.from_template(prompt_template)
        
        rag_chain = (
            {
                "context": retriever | self._format_docs_with_sources,
                "question": RunnablePassthrough(),
                "repo_name": lambda x: repo_name
            }
            | prompt
            | self.chat_model
            | StrOutputParser()
        )

        result = rag_chain.invoke(question, config={"callbacks": callbacks})
        return {"result": result}