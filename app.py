# app.py

import os
import json
import streamlit as st
from dotenv import load_dotenv
from io import BytesIO

# Import all our modules
from src.documate.repo_manager import RepoManager
from src.documate.analytics_agent import AnalyticsAgent, ChromaStore
from src.documate.embeddings_factory import get_embedding_model
from src.documate.llm_factory import get_chat_model
from src.documate.qa_agent import QAAgent
from src.documate.callbacks.streamlit_callback import StreamlitCallbackHandler
from src.documate.doc_agent.agent import DocumentationAgent
from src.documate.wiki_agent.orchestrator import WikiOrchestrator

# --- Page Configuration and Service Initialization ---
st.set_page_config(page_title="Documate", page_icon="🤖", layout="wide")
load_dotenv()

@st.cache_resource
def get_services():
    """Initializes and caches all core services for the application."""
    clone_path = os.getenv("CLONE_PATH", "cloned_repos")
    vector_db_path = os.getenv("VECTOR_DB_PATH", "vector_stores")
    
    services = {}
    try:
        services["embeddings"] = get_embedding_model()
        services["chat_model"] = get_chat_model()
    except Exception as e:
        st.error(f"Failed to initialize AI models: {e}. Check your .env file and API keys.")
        st.stop()

    services["repo_manager"] = RepoManager(base_clone_path=clone_path)
    services["analytics_agent"] = AnalyticsAgent(
        config_path="configs/file_filters.json",
        vector_store=ChromaStore(),
        embeddings=services["embeddings"],
        vector_db_path=vector_db_path
    )
    services["qa_agent"] = QAAgent(
        chat_model=services["chat_model"],
        embeddings=services["embeddings"],
        vector_db_path=vector_db_path
    )
    services["doc_agent"] = DocumentationAgent(
        llm=services["chat_model"],
        qa_agent=services["qa_agent"]
    )
    services["wiki_orchestrator"] = WikiOrchestrator(
        llm=services["chat_model"], 
        qa_agent=services["qa_agent"]
    )
    return services

services = get_services()
manager = services["repo_manager"]
analytics_agent = services["analytics_agent"]
qa_agent = services["qa_agent"]
doc_agent = services["doc_agent"]
wiki_orchestrator = services["wiki_orchestrator"]

# --- UI Layout ---
st.title("🤖 Documate: Your AI Codebase Companion")

# --- Ingestion and Analysis Section (Collapsible) ---
with st.expander("Step 1: Ingest & Analyze a Repository", expanded=False):
    tab1, tab2 = st.tabs(["Clone from URL", "Upload ZIP File"])

    def handle_ingestion_success(local_path):
        repo_name = os.path.basename(local_path)
        with st.spinner(f"Analyzing {repo_name}... This may take a while."):
            success = analytics_agent.process_repository(local_path)
        if success:
            st.success(f"Analysis complete for **{repo_name}**! You can now interact with it below.")
            st.session_state.selected_repo = repo_name  # Auto-select the new repo
            # Clear any previously generated docs for the new repo
            if "generated_readme" in st.session_state:
                del st.session_state.generated_readme
        else:
            st.error("Analysis failed. Please check the terminal for more details.")
    
    with tab1:
        repo_url = st.text_input("Repository URL", placeholder="https://github.com/your-org/your-repo.git", key="repo_url_input")
        if st.button("Clone and Analyze"):
            if repo_url:
                with st.spinner(f"Cloning {repo_url}..."):
                    local_path = manager.clone_repo(repo_url)
                if local_path:
                    handle_ingestion_success(local_path)
                else:
                    st.error("Failed to clone. Check URL and see terminal logs.")
    
    with tab2:
        uploaded_file = st.file_uploader("Choose a ZIP file", type="zip", key="zip_uploader")
        if uploaded_file:
            if st.button("Upload and Analyze"):
                with st.spinner("Processing ZIP file..."):
                    local_path = manager.process_zip_file(BytesIO(uploaded_file.getvalue()), uploaded_file.name)
                if local_path:
                    handle_ingestion_success(local_path)
                else:
                    st.error("Failed to process ZIP. See terminal logs.")

st.divider()

# --- Main Application Section ---
st.header("Step 2: Interact with a Repository")

vector_db_path = os.getenv("VECTOR_DB_PATH", "vector_stores")
if os.path.exists(vector_db_path):
    available_repos = [d for d in os.listdir(vector_db_path) if os.path.isdir(os.path.join(vector_db_path, d))]
else:
    available_repos = []

if not available_repos:
    st.warning("No repositories have been analyzed yet. Please use the section above to ingest a repository.")
else:
    # Initialize session state for selected repo if it doesn't exist or is no longer valid
    if "selected_repo" not in st.session_state or st.session_state.selected_repo not in available_repos:
        st.session_state.selected_repo = available_repos[0]
    
    # Store the previous repo to detect changes
    prev_repo = st.session_state.selected_repo
    
    selected_repo = st.selectbox(
        "Choose a repository:",
        options=available_repos,
        index=available_repos.index(st.session_state.selected_repo)
    )
    st.session_state.selected_repo = selected_repo

    # If the repository has changed, clear the chat and generated docs
    if prev_repo != selected_repo:
        st.session_state.messages = []
        if "generated_readme" in st.session_state:
            del st.session_state.generated_readme

    # --- Create tabs for different interactions ---
    chat_tab, docs_tab, wiki_tab = st.tabs([
        "💬 Ask Documate (Chat)", 
        "📝 Generate Single README", 
        "📚 Generate DeepWiki"
    ])

    with chat_tab:
        st.subheader(f"Chat with `{selected_repo}`")
        # Initialize chat history for the current repo
        if "messages" not in st.session_state or st.session_state.get("current_repo_chat") != selected_repo:
            st.session_state.messages = []
            st.session_state.current_repo_chat = selected_repo

        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input(f"Ask a question about {selected_repo}..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                container = st.empty()
                st_callback = StreamlitCallbackHandler(container)
                try:
                    response = qa_agent.get_answer(
                        question=prompt,
                        repo_name=selected_repo,
                        callbacks=[st_callback]
                    )
                    final_answer = response.get("result", "Sorry, I couldn't find an answer.")
                    container.markdown(final_answer)
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    with docs_tab:
        st.subheader(f"Generate Full Documentation for `{selected_repo}`")
        st.markdown("Click the button below to start an AI agent that will research the codebase and generate a complete `README.md` file, including an architecture diagram.")
        
        if st.button("🚀 Generate README.md", key="generate_docs"):
            with st.spinner("🤖 The documentation agent is at work... This can take a few minutes. Check the terminal for progress."):
                try:
                    generated_readme = doc_agent.generate(selected_repo)
                    st.session_state.generated_readme = generated_readme
                except Exception as e:
                    st.error(f"An error occurred during documentation generation: {e}")
                    st.session_state.generated_readme = None

        if "generated_readme" in st.session_state and st.session_state.generated_readme:
            st.divider()
            st.subheader("Generated Document")
            
            st.markdown(st.session_state.generated_readme)
            
            st.download_button(
                label="Download README.md",
                data=st.session_state.generated_readme,
                file_name=f"{selected_repo}_README.md",
                mime="text/markdown",
            )
        st.subheader(f"Generate Full Documentation for `{selected_repo}` (Single File)")
    
    with wiki_tab:
        st.subheader(f"Generate & View a Multi-Page DeepWiki for `{selected_repo}`")

        wiki_output_path = os.path.join("wikis", selected_repo)
        plan_path = os.path.join(wiki_output_path, "structure.json")

        # --- Generation Section ---
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🚀 Generate or Re-generate DeepWiki", key="generate_wiki", use_container_width=True):
                # When we start a new generation, set the completion flag to False
                st.session_state.wiki_generation_complete = False
                with st.spinner("🤖 The full DeepWiki agent is at work... This will take several minutes."):
                    try:
                        # This runs the full plan -> save -> generate pages workflow
                        wiki_orchestrator.generate(selected_repo)
                        st.success("DeepWiki generation complete!")
                        # --- CRITICAL FIX: Set a flag upon successful completion ---
                        st.session_state.wiki_generation_complete = True
                        # Clear query params to show the intro page after generation
                        st.query_params.clear()
                        # Rerun to ensure the display section updates correctly
                        st.rerun()
                    except Exception as e:
                        st.error(f"An error occurred during wiki generation: {e}")
                        st.session_state.wiki_generation_complete = False

        with col2:
            st.info("This process invokes multiple AI agents to plan the wiki structure, write content for each page, and generate diagrams. Check your terminal for detailed progress.")

        st.divider()

        # --- Display Section ---
        # --- CRITICAL FIX: Check for BOTH the plan and the completion flag ---
        # We also check the session state flag to ensure generation isn't in progress.
        if "wiki_generation_complete" not in st.session_state:
            st.session_state.wiki_generation_complete = os.path.exists(plan_path)

        if not st.session_state.wiki_generation_complete:
            st.info("No wiki has been generated for this repository yet, or generation is in progress. Click the button above to start.")
        else:
            # If the flag is true, we can safely assume the files exist.
            try:
                with open(plan_path, 'r') as f:
                    wiki_structure = json.load(f)

                query_params = st.query_params
                current_page_file = query_params.get("page", [None])[0]

                default_page_file = wiki_structure["pages"][0]["file"] if wiki_structure["pages"] else None
                if current_page_file is None:
                    current_page_file = default_page_file

                nav_col, content_col = st.columns([1, 3])

                with nav_col:
                    st.header(wiki_structure.get("title", "Wiki Navigation"))
                    
                    def render_nav(pages, level=0):
                        for page in pages:
                            # Indent sub-pages
                            prefix = " " * (level * 4) + "- "
                            if st.button(page['title'], key=page['file'], use_container_width=True):
                                st.query_params["page"] = page['file']
                                st.rerun()
                            if page.get("pages"):
                                render_nav(page["pages"], level + 1)
                    
                    render_nav(wiki_structure["pages"])

                with content_col:
                    content_file_path = os.path.join(wiki_output_path, current_page_file) if current_page_file else None
                    if content_file_path and os.path.exists(content_file_path):
                        with open(content_file_path, 'r', encoding='utf-8') as f:
                            page_content = f.read()
                        st.markdown(page_content, unsafe_allow_html=True)
                    elif default_page_file and os.path.exists(os.path.join(wiki_output_path, default_page_file)):
                        # Fallback to default if current page is somehow invalid
                        with open(os.path.join(wiki_output_path, default_page_file), 'r', encoding='utf-8') as f:
                            page_content = f.read()
                        st.markdown(page_content, unsafe_allow_html=True)
                    else:
                        st.warning("Select a page from the navigation to view its content, or the content file may be missing.")

            except FileNotFoundError:
                st.error("Wiki structure file found, but content files are missing. Please try re-generating the wiki.")
                st.session_state.wiki_generation_complete = False
            except Exception as e:
                st.error(f"An error occurred while displaying the wiki: {e}")