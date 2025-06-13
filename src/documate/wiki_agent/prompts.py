
from typing import List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field

# --- Models for the LLM (Flat Structure) ---
# The LLM will be instructed to output this flat structure to avoid recursion issues.

class FlatPage(BaseModel):
    """
    Represents a single page in the wiki's flat structure.
    Hierarchy is defined by the 'parent_file' field.
    """
    title: str = Field(description="The clear and concise title of the page.")
    file: str = Field(
        description="A unique, numbered markdown filename for this page, e.g., '01_Introduction.md' or '03_01_User_Routes.md'."
    )
    parent_file: Optional[str] = Field(
        default=None,
        description="The filename of the parent page. Use null or omit for top-level pages."
    )

class FlatWikiStructure(BaseModel):
    """
    A flattened list of all pages in the documentation wiki.
    The hierarchy can be reconstructed from the 'parent_file' fields.
    """
    title: str = Field(description="The main title of the wiki, usually the repository name.")
    pages: List[FlatPage] = Field(description="A flat list of all pages to be included in the wiki.")


# --- Models for Internal Use (Nested Structure) ---
# After the LLM generates the flat structure, our code will reconstruct it into this
# nested format for easy use in the UI and for saving the final structure.json.
# This part is NOT sent to the LLM.

class Page(BaseModel):
    """A single page in the documentation wiki (nested format)."""
    title: str
    file: str
    # 'pages' is now optional and defaults to an empty list
    pages: Optional[List['Page']] = Field(default_factory=list)

# Update forward reference for the recursive model
Page.update_forward_refs()

class WikiStructure(BaseModel):
    """The complete hierarchical structure of the documentation wiki (nested format)."""
    title: str
    pages: List[Page]


# --- NEW Prompt for the Wiki Architect (to match the flat structure) ---
ARCHITECT_PROMPT_TEMPLATE = """
You are an expert software architect and technical writer. Your task is to analyze the file structure of a given codebase and design a comprehensive, hierarchical structure for its documentation wiki.

You must output a JSON object that provides a FLAT LIST of all pages. The hierarchy will be defined by linking pages to their parents via the `parent_file` field.

**Codebase File Structure:**
```
{file_structure}
```

**Instructions:**
1.  Analyze the file structure to identify main components, features, and logical groupings.
2.  Design a table of contents that is intuitive for a new developer.
3.  Create clear, descriptive titles for each page.
4.  Generate unique, numbered filenames for each page to ensure a logical order (e.g., `01_Setup.md`, `02_Architecture.md`).
5.  For nested pages, set the `parent_file` to the filename of the parent. For example, a page with `file: '02_01_API_Server.md'` should have `parent_file: '02_Architecture.md'`.
6.  For all top-level pages, the `parent_file` field should be null or omitted.

Based on your analysis, generate the complete JSON structure containing the flat list of all pages.
"""

PAGE_WRITER_PROMPT_TEMPLATE = """
You are an expert technical writer, assigned to write a single, detailed documentation page for a software project.

**Your Goal:** Write a comprehensive, clear, and well-structured markdown page about the specific topic: **"{page_title}"**.

**Context:**
- Repository Name: `{repo_name}`
- This page is a part of a larger section titled: `{parent_title}`. Use this to frame your explanation.
- You have access to a Q&A tool that can answer questions about the codebase.

**Instructions:**
1.  Formulate 2-4 detailed questions about the topic **"{page_title}"** that will help you gather all the necessary information.
2.  For each question, use your Q&A tool to get answers directly from the codebase.
3.  Synthesize the answers into a detailed and coherent markdown page.
4.  **Crucially, you MUST include source file citations** (e.g., `src/main.py`) provided by the Q&A tool to help developers locate the code.
5.  If relevant, include code blocks to illustrate key points.
6.  Start the document directly with the page title as a main heading (e.g., `# {page_title}`). Do not add any other introductory text.

Begin by thinking about the questions you need to ask.
"""

DIAGRAM_TOOL_PROMPT_TEMPLATE = """
You are a system architect. Your task is to generate a Mermaid.js diagram for a specific sub-topic of a codebase.

**Topic to Diagram:** "{topic}"

**Instructions:**
1.  Generate a Mermaid.js 'graph TD' (Top-Down) or 'graph LR' (Left-to-Right) diagram that illustrates the flow or structure of this specific topic.
2.  The diagram should be focused and not overly complex.
3.  Output **ONLY** the raw Mermaid.js code block. Do not include ```mermaid or any other text.
4.  If a diagram is not applicable for this topic, output the single word: "IGNORE".
"""