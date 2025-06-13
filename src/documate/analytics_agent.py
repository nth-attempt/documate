import os
import json
import fnmatch
from typing import List, Dict, Any

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

class VectorStoreInterface:
    def from_documents(self, documents: List[Document], embeddings: Any, **kwargs):
        raise NotImplementedError

class ChromaStore(VectorStoreInterface):
    def from_documents(self, documents: List[Document], embeddings: Any, **kwargs):
        persist_directory = kwargs.get("persist_directory")
        if not persist_directory:
            raise ValueError("ChromaStore requires 'persist_directory' in kwargs")
        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )

class AnalyticsAgent:
    """
    An agent that analyzes a repository, filters files, and indexes them.
    """
    def __init__(self, config_path: str, vector_store: VectorStoreInterface, embeddings: Any, vector_db_path: str):
        self.filters = self._load_filters(config_path)
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.vector_db_path = vector_db_path
        print("AnalyticsAgent initialized.")

    def _load_filters(self, config_path: str) -> Dict[str, Any]:
        print(f"Loading file filters from: {config_path}")
        with open(config_path, 'r') as f:
            return json.load(f)

    def _filter_files(self, repo_path: str) -> List[str]:
        allowed_extensions = self.filters.get("allowed_extensions", [])
        excluded_patterns = self.filters.get("excluded_patterns", [])
        included_files = []
        for root, _, files in os.walk(repo_path):
            for filename in files:
                filepath = os.path.join(root, filename)
                if any(fnmatch.fnmatch(filepath, os.path.join(repo_path, pattern)) for pattern in excluded_patterns):
                    continue
                if any(filename.endswith(ext) for ext in allowed_extensions):
                    included_files.append(filepath)
        print(f"Found {len(included_files)} files to analyze after filtering.")
        return included_files

    def _load_and_split_documents(self, file_paths: List[str]) -> List[Document]:
        docs = []
        for file_path in file_paths:
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                docs.extend(loader.load())
            except Exception as e:
                print(f"Warning: Could not load file {file_path}. Error: {e}")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)
        print(f"Split {len(docs)} documents into {len(split_docs)} chunks.")
        return split_docs

    def process_repository(self, repo_path: str) -> bool:
        repo_name = os.path.basename(repo_path)
        print(f"\n--- Starting Analysis for: {repo_name} ---")
        try:
            relevant_files = self._filter_files(repo_path)
            if not relevant_files:
                print("No relevant files found to process. Aborting.")
                return False
            documents = self._load_and_split_documents(relevant_files)
            if not documents:
                print("No documents could be loaded or split. Aborting.")
                return False
            persist_directory = os.path.join(self.vector_db_path, repo_name)
            print(f"Generating embeddings and persisting to: {persist_directory}")
            self.vector_store.from_documents(
                documents=documents,
                embeddings=self.embeddings,
                persist_directory=persist_directory
            )
            print(f"✅ Analysis complete for {repo_name}. Vector store created.")
            return True
        except Exception as e:
            print(f"❌ An error occurred during repository processing: {e}")
            return False