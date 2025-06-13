import operator
from typing import TypedDict, List, Annotated

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, END

from .prompts import PLANNER_PROMPT, WRITER_PROMPT, DIAGRAM_PROMPT
from ..qa_agent import QAAgent

# --- 1. Define the State for our agent's memory ---

class ResearchNote(BaseModel):
    question: str = Field(description="The specific question that was asked.")
    answer: str = Field(description="The answer to the question based on the codebase.")

class GraphState(TypedDict):
    repo_name: str
    plan: List[str]
    research_notes: Annotated[List[ResearchNote], operator.add]
    diagram: str
    final_document: str

# --- 2. Define the Agent's Nodes (the functions that do the work) ---

class DocumentationAgentNodes:
    def __init__(self, llm: any, qa_agent: QAAgent):
        self.llm = llm
        self.qa_agent = qa_agent

    def plan_step(self, state: GraphState) -> GraphState:
        """Generates the initial research plan."""
        print("--- AGENT: Planning Step ---")
        prompt = PLANNER_PROMPT.format(repo_name=state["repo_name"])
        # The response is a single string of numbered questions
        response = self.llm.invoke(prompt)
        plan = [q.strip() for q in response.content.split('\n') if q.strip()]
        print(f"Plan: {plan}")
        return {"plan": plan}

    def research_step(self, state: GraphState) -> GraphState:
        """Executes the research plan, one question at a time."""
        print("--- AGENT: Research Step ---")
        plan = state["plan"]
        notes = []
        for i, question in enumerate(plan):
            print(f"Researching question {i+1}/{len(plan)}: {question}")
            try:
                # Use our existing QA agent as a tool
                answer_data = self.qa_agent.get_answer(question, state["repo_name"])
                answer = answer_data.get("result", "Could not find an answer.")
                notes.append(ResearchNote(question=question, answer=answer))
            except Exception as e:
                print(f"Failed to answer question: {question}. Error: {e}")
                notes.append(ResearchNote(question=question, answer="Error retrieving answer."))
        return {"research_notes": notes}

# NEW CODE in src/documate/doc_agent/agent.py
    def generate_diagram_step(self, state: GraphState) -> GraphState:
        """Generates the Mermaid diagram and validates its format."""
        print("--- AGENT: Diagram Generation Step ---")
        prompt = DIAGRAM_PROMPT.format(research_notes=state["research_notes"])
        response = self.llm.invoke(prompt)
        diagram_code = response.content.strip()

        # --- VALIDATION LOGIC ---
        # Check if the output looks like a valid Mermaid graph.
        # A simple check for the 'graph' keyword and flow arrows '-->' is a good heuristic.
        if "graph" in diagram_code and "-->" in diagram_code:
            print(f"Generated Diagram Code:\n{diagram_code}")
            # Clean up just in case the LLM still adds markdown fences
            diagram_code = diagram_code.replace("```mermaid", "").replace("```", "").strip()
            return {"diagram": diagram_code}
        else:
            # The LLM failed to generate a valid diagram. Use a fallback.
            print("--- AGENT: Diagram generation failed. Using fallback. ---")
            fallback_diagram = 'graph TD\n    A["Error: Could not generate architecture diagram."]'
            return {"diagram": fallback_diagram}

    def write_document_step(self, state: GraphState) -> GraphState:
        """Synthesizes all research into the final README."""
        print("--- AGENT: Document Writing Step ---")
        prompt = WRITER_PROMPT.format(
            repo_name=state["repo_name"],
            research_notes=state["research_notes"],
            diagram=state["diagram"],
        )
        response = self.llm.invoke(prompt)
        print("--- AGENT: Finished ---")
        return {"final_document": response.content}


# --- 3. Define the Graph and the Wrapper Class ---

class DocumentationAgent:
    def __init__(self, llm: any, qa_agent: QAAgent):
        self.nodes = DocumentationAgentNodes(llm, qa_agent)
        self.workflow = self._build_graph()

    def _build_graph(self):
        """Compiles the nodes into a runnable LangGraph workflow."""
        graph = StateGraph(GraphState)
        # Rename "plan" node to "planner"
        graph.add_node("planner", self.nodes.plan_step)
        graph.add_node("research", self.nodes.research_step)
        graph.add_node("generate_diagram", self.nodes.generate_diagram_step)
        graph.add_node("write_document", self.nodes.write_document_step)

        # Update the edges to use the new node name
        graph.set_entry_point("planner")
        graph.add_edge("planner", "research")
        graph.add_edge("research", "generate_diagram")
        graph.add_edge("generate_diagram", "write_document")
        graph.add_edge("write_document", END)

        return graph.compile()

    def generate(self, repo_name: str) -> str:
        """
        Runs the documentation generation process for a given repository.
        
        Args:
            repo_name: The name of the repository to document.

        Returns:
            The generated README.md content as a string.
        """
        initial_state = {"repo_name": repo_name, "research_notes": []}
        final_state = self.workflow.invoke(initial_state, {"recursion_limit": 10})
        return final_state.get("final_document", "Error: Document generation failed.")