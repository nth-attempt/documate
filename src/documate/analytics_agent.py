import os
import json
import fnmatch
from typing import List, Dict, Any

from langchain_community.document_loaders import TextLoader, DirectoryLoader
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
    An agent that can analyze a directory of files, filter them,
    and index them into a specified vector store.
    """
    def __init__(self, config_path: str, vector_store: VectorStoreInterface, embeddings: Any):
        self.filters = self._load_filters(config_path)
        self.vector_store = vector_store
        self.embeddings = embeddings
        print("AnalyticsAgent (Reusable) initialized.")

    def _load_filters(self, config_path: str) -> Dict[str, Any]:
        #... (this method remains the same)
        print(f"Loading file filters from: {config_path}")
        with open(config_path, 'r') as f:
            return json.load(f)

    def _filter_files(self, source_path: str, allowed_extensions: List[str]) -> List[str]:
        #... (this method is now more generic)
        excluded_patterns = self.filters.get("excluded_patterns", [])
        included_files = []
        for root, _, files in os.walk(source_path):
            for filename in files:
                filepath = os.path.join(root, filename)
                if any(fnmatch.fnmatch(filepath, os.path.join(source_path, pattern)) for pattern in excluded_patterns):
                    continue
                if any(filename.endswith(ext) for ext in allowed_extensions):
                    included_files.append(filepath)
        print(f"Found {len(included_files)} files to analyze in '{source_path}'.")
        return included_files

    def index_codebase(self, repo_path: str, index_path: str):
        """Processes a source code repository."""
        print(f"\n--- Starting Codebase Indexing for: {os.path.basename(repo_path)} ---")
        allowed_extensions = self.filters.get("allowed_extensions", [])
        relevant_files = self._filter_files(repo_path, allowed_extensions)
        self._process_and_index_files(relevant_files, index_path)

    def index_wiki(self, wiki_path: str, index_path: str):
        """Processes a directory of generated wiki files."""
        print(f"\n--- Starting Wiki Indexing for: {os.path.basename(wiki_path)} ---")
        # For wikis, we only care about markdown files
        relevant_files = self._filter_files(wiki_path, [".md"])
        self._process_and_index_files(relevant_files, index_path)

    def _process_and_index_files(self, file_paths: List[str], persist_directory: str):
        """The generic core logic for loading, splitting, and indexing."""
        if not file_paths:
            print("No relevant files found to process. Aborting indexing.")
            return False
        
        # Load and Split
        docs = []
        for file_path in file_paths:
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                docs.extend(loader.load())
            except Exception as e:
                print(f"Warning: Could not load file {file_path}. Error: {e}")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        split_docs = text_splitter.split_documents(docs)
        print(f"Split {len(docs)} documents into {len(split_docs)} chunks.")

        if not split_docs:
            print("No documents could be loaded or split. Aborting indexing.")
            return False

        # Generate Embeddings and Store
        print(f"Generating embeddings and persisting to: {persist_directory}")
        self.vector_store.from_documents(
            documents=split_docs,
            embeddings=self.embeddings,
            persist_directory=persist_directory
        )
        print(f"✅ Indexing complete for path: {persist_directory}")
        return True