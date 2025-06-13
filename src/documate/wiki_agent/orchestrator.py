# src/documate/wiki_agent/orchestrator.py

from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END
import os
import json

from .architect import WikiArchitect
from .prompts import FlatWikiStructure, Page, WikiStructure
from .page_writer import PageWriter 
from ..qa_agent import QAAgent 
from ..analytics_agent import AnalyticsAgent

# --- 1. Define the State for the Orchestrator ---

class WikiGenerationState(TypedDict):
    repo_name: str
    repo_path: str
    # The flat plan from the architect
    flat_wiki_structure: FlatWikiStructure
    # The final nested structure we will save
    nested_wiki_structure: WikiStructure
    output_path: str

# --- 2. Define the Orchestrator's Nodes ---

class OrchestratorNodes:
    # --- CHANGE: __init__ now also takes an AnalyticsAgent ---
    def __init__(self, architect, page_writer, analytics_agent: AnalyticsAgent):
        self.architect = architect
        self.page_writer = page_writer
        self.analytics_agent = analytics_agent 
    
    def plan_wiki_structure(self, state: WikiGenerationState) -> WikiGenerationState:
        print("--- Orchestrator: Planning wiki structure ---")
        structure = self.architect.generate_wiki_structure(
            state["repo_name"], state["repo_path"]
        )
        # --- CHANGE: Store the flat structure first ---
        return {"flat_wiki_structure": structure}

    # --- NEW NODE: Reconstruct the hierarchy ---
    def reconstruct_hierarchy(self, state: WikiGenerationState) -> WikiGenerationState:
        """Node that converts the flat list of pages into a nested hierarchy."""
        print("--- Orchestrator: Reconstructing page hierarchy ---")
        flat_structure = state["flat_wiki_structure"]
        pages_map = {page.file: Page(**page.dict(), pages=[]) for page in flat_structure.pages}
        
        nested_pages = []
        for page_model in flat_structure.pages:
            if page_model.parent_file:
                parent = pages_map.get(page_model.parent_file)
                if parent:
                    # It's a sub-page, append it to its parent's list
                    parent.pages.append(pages_map[page_model.file])
                else:
                    # It's an orphan, treat as a top-level page
                    nested_pages.append(pages_map[page_model.file])
            else:
                # It's a top-level page
                nested_pages.append(pages_map[page_model.file])

        # Create the final nested structure object
        nested_structure = WikiStructure(title=flat_structure.title, pages=nested_pages)
        return {"nested_wiki_structure": nested_structure}
    
    def generate_all_pages(self, state: WikiGenerationState) -> WikiGenerationState:
        """Node that iterates through the plan and generates content for each page."""
        print("\n--- Orchestrator: Generating content for all wiki pages ---")
        wiki_structure = state["nested_wiki_structure"]
        repo_name = state["repo_name"]
        output_path = state["output_path"]

        # A recursive function to process pages and their sub-pages
        def process_level(pages: List[Page], parent_title: str):
            for page in pages:
                page_content = self.page_writer.write_page(
                    repo_name=repo_name,
                    page_title=page.title,
                    parent_title=parent_title
                )
                
                # Save the content to its file
                file_path = os.path.join(output_path, page.file)
                with open(file_path, "w", encoding='utf-8') as f:
                    f.write(page_content)
                print(f"   ✓ Saved page: {file_path}")

                # Recurse for sub-pages
                if page.pages:
                    process_level(page.pages, page.title)

        process_level(wiki_structure.pages, wiki_structure.title)
        return {} # No state update needed, as we're writing to files

    def save_plan(self, state: WikiGenerationState) -> WikiGenerationState:
        print("--- Orchestrator: Saving plan ---")
        output_path = os.path.join("wikis", state["repo_name"])
        os.makedirs(output_path, exist_ok=True)
        
        # --- CHANGE: Save the newly created nested structure ---
        structure_dict = state["nested_wiki_structure"].dict()
        
        plan_file_path = os.path.join(output_path, "structure.json")
        with open(plan_file_path, "w") as f:
            json.dump(structure_dict, f, indent=2)
            
        print(f"Plan saved to {plan_file_path}")
        return {"output_path": output_path}
    
    def index_generated_wiki(self, state: WikiGenerationState) -> WikiGenerationState:
        """Node that triggers the AnalyticsAgent to index the new wiki files."""
        print("\n--- Orchestrator: Indexing generated wiki content ---")
        repo_name = state["repo_name"]
        wiki_source_path = state["output_path"] # The 'wikis/repo_name' directory
        wiki_index_path = os.path.join(os.getenv("VECTOR_DB_PATH", "vector_stores"), repo_name, "wiki")
        
        self.analytics_agent.index_wiki(
            wiki_path=wiki_source_path,
            index_path=wiki_index_path
        )
        return {}

# --- 3. Define the Main Orchestrator Class ---

class WikiOrchestrator:
    # --- CHANGE: __init__ now needs both qa_agent and analytics_agent ---
    def __init__(self, llm: any, qa_agent: QAAgent, analytics_agent: AnalyticsAgent):
        architect = WikiArchitect(llm)
        page_writer = PageWriter(llm, qa_agent)
        # Pass the analytics_agent to the nodes class
        self.nodes = OrchestratorNodes(architect, page_writer, analytics_agent)
        self.workflow = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(WikiGenerationState)
        # ... (add_node calls for existing nodes) ...
        graph.add_node("plan_structure", self.nodes.plan_wiki_structure)
        graph.add_node("reconstruct_hierarchy", self.nodes.reconstruct_hierarchy)
        graph.add_node("save_plan", self.nodes.save_plan)
        graph.add_node("generate_pages", self.nodes.generate_all_pages)
        # --- ADD the new indexing node ---
        graph.add_node("index_wiki", self.nodes.index_generated_wiki)
        
        # --- CHANGE the graph's data flow ---
        graph.set_entry_point("plan_structure")
        graph.add_edge("plan_structure", "reconstruct_hierarchy")
        graph.add_edge("reconstruct_hierarchy", "save_plan")
        graph.add_edge("save_plan", "generate_pages")
        graph.add_edge("generate_pages", "index_wiki") # <-- New Edge
        graph.add_edge("index_wiki", END) # <-- New Edge
        
        return graph.compile()

    def generate(self, repo_name: str):
        repo_path = os.path.join("cloned_repos", repo_name)
        if not os.path.exists(repo_path):
            raise FileNotFoundError(f"Repository path not found: {repo_path}")
            
        initial_state = {"repo_name": repo_name, "repo_path": repo_path}
        
        print(f"\n🚀 Starting CodeWiki Generation for: {repo_name} 🚀\n")
        self.workflow.invoke(initial_state, {"recursion_limit": 10})
        print(f"\n✅ CodeWiki Scaffolding Complete for: {repo_name} ✅\n")