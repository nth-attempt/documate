# src/documate/wiki_agent/architect.py

import os
# --- CHANGE: Import the new Pydantic model and prompt ---
from .prompts import ARCHITECT_PROMPT_TEMPLATE, FlatWikiStructure

from typing import Any

class WikiArchitect:
    """
    An agent that designs the hierarchical structure of the wiki.
    """
    def __init__(self, llm: Any):
        # --- CHANGE: Use the new FlatWikiStructure model ---
        self.llm_with_structure = llm.with_structured_output(FlatWikiStructure)
        self.prompt = ARCHITECT_PROMPT_TEMPLATE
        print("WikiArchitect (Flat Structure) initialized.")

    def _get_file_structure(self, repo_path: str) -> str:
        # This method remains the same
        tree = []
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__']]
            level = root.replace(repo_path, '').count(os.sep)
            indent = ' ' * 4 * level
            tree.append(f"{indent}{os.path.basename(root)}/")
            sub_indent = ' ' * 4 * (level + 1)
            for f in files:
                if not f.startswith('.'):
                    tree.append(f"{sub_indent}{f}")
        return "\n".join(tree)

    # --- CHANGE: The method now returns a FlatWikiStructure object ---
    def generate_wiki_structure(self, repo_name: str, repo_path: str) -> FlatWikiStructure:
        print(f"--- Architect: Designing wiki structure for {repo_name} ---")
        file_structure = self._get_file_structure(repo_path)
        
        prompt_with_context = self.prompt.format(file_structure=file_structure)
        
        wiki_structure = self.llm_with_structure.invoke(prompt_with_context)
        
        print(f"--- Architect: Flat wiki structure designed for '{wiki_structure.title}' with {len(wiki_structure.pages)} total pages. ---")
        return wiki_structure