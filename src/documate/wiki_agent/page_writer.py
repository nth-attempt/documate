# src/documate/wiki_agent/page_writer.py

from .prompts import PAGE_WRITER_PROMPT_TEMPLATE, DIAGRAM_TOOL_PROMPT_TEMPLATE
from ..qa_agent import QAAgent
import re

class PageWriter:
    """
    An agent that can write a detailed markdown page for a single topic.
    """
    def __init__(self, llm: any, qa_agent: QAAgent):
        self.llm = llm
        self.qa_agent = qa_agent
        print("PageWriter initialized.")

    def _generate_diagram_if_needed(self, topic: str, full_content: str) -> str:
        """Checks if a diagram is requested and generates it."""
        # A simple heuristic: if the content mentions "flow", "architecture", "structure", etc.
        keywords = ['flow', 'architecture', 'structure', 'diagram', 'interaction', 'sequence']
        if any(keyword in full_content.lower() for keyword in keywords):
            print(f"--- PageWriter: Diagram keyword found for topic '{topic}'. Attempting generation. ---")
            prompt = DIAGRAM_TOOL_PROMPT_TEMPLATE.format(topic=topic)
            diagram_code = self.llm.invoke(prompt).content.strip()

            if "IGNORE" not in diagram_code and "graph" in diagram_code:
                # Return just the raw Mermaid code
                return diagram_code
        # Return an empty string if no diagram is generated
        return ""

    def write_page(self, repo_name: str, page_title: str, parent_title: str) -> str:
        """
        Researches and writes a single documentation page.
        """
        print(f"--- PageWriter: Starting to write page for '{page_title}' ---")
        
        # This is a simplified agentic process without the overhead of a full graph for this sub-task.
        # 1. First, we prompt the LLM to think about the questions it needs to ask.
        initial_prompt = PAGE_WRITER_PROMPT_TEMPLATE.format(
            page_title=page_title,
            repo_name=repo_name,
            parent_title=parent_title or "Main"
        )
        llm_response_text = self.llm.invoke(initial_prompt).content
        
        # 2. We extract the questions from the LLM's plan.
        questions = re.findall(r'\d+\.\s*(.*)', llm_response_text)
        if not questions:
             # Fallback if the LLM doesn't format correctly
            questions = [f"Provide a detailed explanation of {page_title} including its main components and purpose."]

        print(f"--- PageWriter: Researching with questions: {questions} ---")
        
        # 3. We use our QAAgent to answer these questions.
        research_notes = ""
        for q in questions:
            try:
                answer = self.qa_agent.get_answer(q, repo_name)['result']
                research_notes += f"Q: {q}\nA: {answer}\n\n"
            except Exception as e:
                print(f"Error answering question '{q}': {e}")
                research_notes += f"Q: {q}\nA: Could not retrieve answer.\n\n"

        # 4. We synthesize the final document.
        synthesis_prompt = f"""
        You are a technical writer. Synthesize the following research notes into a final, comprehensive markdown document for the page titled "{page_title}".
        Ensure you preserve the source file citations. Start directly with the markdown heading for the title.

        **Research Notes:**
        {research_notes}

        **Final Markdown Document:**
        """
        final_content = self.llm.invoke(synthesis_prompt).content

        # 5. Generate a diagram if needed and insert a placeholder
        # The _generate_diagram_if_needed method now returns the raw code, not markdown
        diagram_code = self._generate_diagram_if_needed(page_title, final_content)
        
        if diagram_code:
            # We use a unique placeholder that's unlikely to appear naturally
            placeholder = f"\n\n%%MERMAID_DIAGRAM%%{diagram_code}%%MERMAID_DIAGRAM%%\n\n"
            # We try to insert it in a logical place, e.g., after the first major section
            first_heading_match = re.search(r'\n## .*\n', final_content)
            if first_heading_match:
                insert_pos = first_heading_match.end()
                final_content = final_content[:insert_pos] + placeholder + final_content[insert_pos:]
            else:
                # Fallback to appending at the end
                final_content += placeholder
        
        print(f"--- PageWriter: Finished page for '{page_title}' ---")
        return final_content