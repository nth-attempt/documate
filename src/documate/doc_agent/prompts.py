# src/documate/doc_agent/prompts.py

from langchain_core.prompts import PromptTemplate

# 1. Prompt for the Planner Node
PLANNER_PROMPT_TEMPLATE = """
Based on the research notes and diagram, generate a complete README.md file. The README should include the following sections:
-   `# {repo_name}`: The project title.
-   `## 🚀 Overview`: A brief, high-level summary of the project.
-   `## 🏗️ Architecture`: A description of the core components and their interactions, supported by the architecture diagram. Include the Mermaid diagram block here.
-   `## ⚙️ Setup and Usage`: Instructions on how to set up and run the project locally.
-   `## ✨ Key Features`: A brief highlight of 1-2 important features.

**IMPORTANT:** Make sure to include the source file citations (e.g., `src/main.js`) that are present in the research notes. This helps developers find the relevant code quickly.

The tone should be professional and helpful for a new developer approaching this codebase.
Do not add any text before the title or after the document. Output only the raw markdown.

Research Plan:
"""
PLANNER_PROMPT = PromptTemplate.from_template(PLANNER_PROMPT_TEMPLATE)


# 2. Prompt for the Document Writer Node
WRITER_PROMPT_TEMPLATE = """
You are a technical writer tasked with creating a high-quality, comprehensive README.md file for a software project.

You have been provided with a list of questions and their corresponding answers, which were researched from the codebase.
Your job is to synthesize this information into a clear and well-structured markdown document.

The repository name is: {repo_name}

**Research Notes (Questions and Answers):**
{research_notes}

**Diagram:**
```mermaid
{diagram}
```

Based on the research notes and diagram, generate a complete README.md file. The README should include the following sections:
-   `# {repo_name}`: The project title.
-   `## 🚀 Overview`: A brief, high-level summary of the project.
-   `## 🏗️ Architecture`: A description of the core components and their interactions, supported by the architecture diagram. Include the Mermaid diagram block here.
-   `## ⚙️ Setup and Usage`: Instructions on how to set up and run the project locally.
-   `## ✨ Key Features`: A brief highlight of 1-2 important features.

The tone should be professional and helpful for a new developer approaching this codebase.
Do not add any text before the title or after the document. Output only the raw markdown.
"""
WRITER_PROMPT = PromptTemplate.from_template(WRITER_PROMPT_TEMPLATE)


# 3. Prompt for the Diagram Generator Node
DIAGRAM_PROMPT_TEMPLATE = """
You are an expert system architect. Your task is to generate a high-level architecture diagram based on the provided research about a codebase.

**IMPORTANT INSTRUCTIONS:**
1.  The diagram's syntax MUST be Mermaid.js 'graph TD' (Top-Down) format.
2.  The output MUST be ONLY the Mermaid.js code block. Do not include any other text, explanation, or markdown formatting (like ```mermaid).
3.  The diagram should show the main components, services, or modules and the primary flow of data or control between them.
4.  Keep it high-level. Do not include every single function or class. Focus on the big picture.
5.  If you cannot determine the architecture from the context, output the text "Error: Could not determine architecture."

**Research Notes (Questions and Answers):**
{research_notes}

Based on the notes, generate the Mermaid diagram code.

Example of a GOOD output:
graph TD
    A[User Request] --> B(API Server);
    B --> C{{Database}};
    B --> D[Authentication Service];
    A --> E[Frontend Client];
"""
DIAGRAM_PROMPT = PromptTemplate.from_template(DIAGRAM_PROMPT_TEMPLATE)