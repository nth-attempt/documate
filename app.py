# app.py

import os
import streamlit as st
import json
from dotenv import load_dotenv
from io import BytesIO

# Import all our modules
from src.documate.repo_manager import RepoManager
from src.documate.analytics_agent import AnalyticsAgent, ChromaStore
from src.documate.embeddings_factory import get_embedding_model
from src.documate.llm_factory import get_chat_model
from src.documate.qa_agent import QAAgent
from src.documate.callbacks.streamlit_callback import StreamlitCallbackHandler
from src.documate.wiki_agent.orchestrator import WikiOrchestrator
from src.documate.global_qa_agent import GlobalQAAgent

# --- 1. PAGE CONFIGURATION & SERVICE INITIALIZATION ---

st.set_page_config(page_title="CodeWiki", page_icon="🧠", layout="wide")
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
        st.error(f"Failed to initialize AI models: {e}. Check .env.")
        st.stop()

    services["repo_manager"] = RepoManager(base_clone_path=clone_path)
    services["analytics_agent"] = AnalyticsAgent(
        config_path="configs/file_filters.json",
        vector_store=ChromaStore(),
        embeddings=services["embeddings"]
    )
    services["qa_agent"] = QAAgent(
        chat_model=services["chat_model"], embeddings=services["embeddings"],
        vector_db_path=vector_db_path
    )
    services["wiki_orchestrator"] = WikiOrchestrator(
        llm=services["chat_model"],
        qa_agent=services["qa_agent"],
        analytics_agent=services["analytics_agent"]
    )
    services["global_qa_agent"] = GlobalQAAgent(
        chat_model=services["chat_model"],
        embeddings=services["embeddings"],
        vector_db_path=vector_db_path
    )
    return services

# Load all services once
services = get_services()
manager = services["repo_manager"]
wiki_orchestrator = services["wiki_orchestrator"]
qa_agent = services["qa_agent"]
global_qa_agent = services["global_qa_agent"]


# Initialize session state variables if they don't exist
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "gallery"
if "selected_repo" not in st.session_state:
    st.session_state.selected_repo = None

# --- 2. UI VIEW DEFINITIONS (Helper Functions) ---

def render_add_new_repo_view():
    st.header("➕ Add a New Repository")
    st.info("Provide a repository to be analyzed and documented. The full analysis and CodeWiki generation will be performed.")

    tab1, tab2 = st.tabs(["Clone from URL", "Upload ZIP File"])

    def handle_ingestion_and_generation(repo_path):
        repo_name = os.path.basename(repo_path)
        with st.spinner("Step 1/2: Analyzing and indexing source code..."):
            # Manually index the code first
            code_index_path = os.path.join(os.getenv("VECTOR_DB_PATH", "vector_stores"), repo_name, "code")
            services["analytics_agent"].index_codebase(repo_path, code_index_path)

        with st.spinner("Step 2/2: Generating and indexing CodeWiki... This may take several minutes."):
            # Now run the orchestrator which will generate and index the wiki
            wiki_orchestrator.generate(repo_name)
        
        st.success("Repository processed and CodeWiki generated successfully!")

    with tab1:
        repo_url = st.text_input("Repository URL", placeholder="https://github.com/your-org/your-repo.git")
        if st.button("Analyze & Generate CodeWiki from URL", use_container_width=True):
            if repo_url:
                with st.spinner(f"Cloning {repo_url}..."):
                    local_path = manager.clone_repo(repo_url)
                if local_path:
                    handle_ingestion_and_generation(local_path)
                else:
                    st.error("Failed to clone. Check URL and see terminal logs.")
    
    with tab2:
        uploaded_file = st.file_uploader("Choose a ZIP file", type="zip")
        if uploaded_file:
            if st.button("Analyze & Generate CodeWiki from ZIP", use_container_width=True):
                with st.spinner("Processing ZIP file..."):
                    local_path = manager.process_zip_file(BytesIO(uploaded_file.getvalue()), uploaded_file.name)
                if local_path:
                    handle_ingestion_and_generation(local_path)
                else:
                    st.error("Failed to process ZIP. See terminal logs.")
    
    if st.button("⬅️ Back to Gallery"):
        st.session_state.view_mode = "gallery"
        st.rerun()


def render_gallery_view():
    st.header("Available Repositories")
    
    clone_path = os.getenv("CLONE_PATH", "cloned_repos")
    if os.path.exists(clone_path):
        available_repos = [d for d in os.listdir(clone_path) if os.path.isdir(os.path.join(clone_path, d))]
    else:
        available_repos = []
    
    cols = st.columns(4)
    with cols[0]:
        with st.container(border=True):
            st.markdown("### ➕ Add New Repo")
            if st.button("Import & Analyze", key="add_new", use_container_width=True):
                st.session_state.view_mode = "add_new"
                st.session_state.selected_repo = None
                st.rerun()

    for i, repo_name in enumerate(available_repos):
        col_index = (i + 1) % 4
        with cols[col_index]:
            with st.container(border=True):
                st.markdown(f"#### {repo_name}")
                if st.button("Select", key=repo_name, use_container_width=True):
                    st.session_state.selected_repo = repo_name
                    st.rerun()


def render_selected_repo_view(repo_name):
    st.header(f"Inspecting: `{repo_name}`")
    if st.button("⬅️ Back to All Repositories"):
        st.session_state.selected_repo = None
        st.rerun()

    chat_tab, wiki_tab = st.tabs(["💬 Ask this Repo", "📚 View CodeWiki"])

    with chat_tab:
        st.subheader(f"Chat with `{repo_name}`")
        if "messages" not in st.session_state or st.session_state.get("current_chat_repo") != repo_name:
            st.session_state.messages = []
            st.session_state.current_chat_repo = repo_name

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input(f"Ask a question about {repo_name}..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                container = st.empty()
                st_callback = StreamlitCallbackHandler(container)
                response = qa_agent.get_answer(prompt, repo_name, [st_callback])
                final_answer = response.get("result", "Sorry, I couldn't find an answer.")
                container.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})

    with wiki_tab:
        st.subheader(f"CodeWiki for `{repo_name}`")
        wiki_output_path = os.path.join("wikis", repo_name)
        plan_path = os.path.join(wiki_output_path, "structure.json")
        if not os.path.exists(plan_path):
            st.warning("No CodeWiki has been generated for this repository yet.")
        else:
            display_wiki(wiki_output_path, plan_path, repo_name)


def display_wiki(wiki_output_path, plan_path, repo_name):
    with open(plan_path, 'r') as f:
        wiki_structure = json.load(f)

    session_key = f"wiki_page_{repo_name}"
    if session_key not in st.session_state or st.session_state.get("current_repo_wiki") != repo_name:
        st.session_state.current_repo_wiki = repo_name
        if wiki_structure.get("pages"):
            st.session_state[session_key] = wiki_structure["pages"][0]["file"]
        else:
            st.session_state[session_key] = None

    nav_col, content_col = st.columns([1, 3], gap="large")
    with nav_col:
        st.header(wiki_structure.get("title", "Wiki Navigation"))
        def render_nav_tree(pages):
            for page in pages:
                if page.get("pages"):
                    child_files = [p['file'] for p in page.get("pages", [])]
                    is_expanded = st.session_state.get(session_key) in child_files
                    with st.expander(page['title'], expanded=is_expanded):
                        render_nav_tree(page["pages"])
                else:
                    if st.button(page['title'], key=f"nav_{page['file']}", use_container_width=True):
                        st.session_state[session_key] = page['file']
                        st.rerun()
        render_nav_tree(wiki_structure["pages"])

        with content_col:
            page_file_to_display = st.session_state.get(session_key)
            
            if page_file_to_display:
                content_file_path = os.path.join(wiki_output_path, page_file_to_display)
                if os.path.exists(content_file_path):
                    with open(content_file_path, 'r', encoding='utf-8') as f:
                        page_content = f.read()

                    # --- NEW LOGIC TO RENDER MERMAID CHARTS ---
                    from streamlit_mermaid import st_mermaid # Import here

                    # Check for our placeholder
                    if "%%MERMAID_DIAGRAM%%" in page_content:
                        parts = page_content.split("%%MERMAID_DIAGRAM%%")
                        # The structure will be [text_before, diagram_code, text_after]
                        st.markdown(parts[0]) # Render text before the diagram
                        st_mermaid(parts[1])  # Render the live diagram
                        st.markdown(parts[2]) # Render text after the diagram
                    else:
                        # If no diagram, just render the whole markdown
                        st.markdown(page_content, unsafe_allow_html=True)
                else:
                    st.error(f"Content file not found: `{page_file_to_display}`. It might have been moved or deleted.")
                    st.info("You may need to re-generate the CodeWiki.")
            else:
                st.info("Select a page from the navigation tree to view its content.")

# --- 3. MAIN ROUTER ---

# --- HEADER WITH LOGO ---
# Use columns to center the image and control its size.
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Make sure the path to your logo is correct
    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", use_column_width=True)
    else:
        st.title("Documate - Effortless Documentation & Insightful Code Analysis") # Fallback to text title if logo not found

st.divider()

if "global_messages" not in st.session_state:
    st.session_state.global_messages = []

# The input field for the global search
prompt = st.chat_input("Search across all repositories...")
if prompt:
    st.session_state.global_messages.append({"role": "user", "content": prompt})
    with st.spinner("🤖 Searching across all knowledge bases..."):
        response = global_qa_agent.get_answer(prompt)
        st.session_state.global_messages.append({"role": "assistant", "content": response["result"]})

# --- Display Global Search Chat History ---
if st.session_state.global_messages:
    for message in st.session_state.global_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Add a button to clear the global search and return to the gallery
    if st.button("Clear Search & View Repositories"):
        st.session_state.global_messages = []
        st.rerun()
    st.divider()

# --- Main Content Router (Gallery or Selected Repo) ---
# Only show the gallery/repo views if there is no active global search
if not st.session_state.global_messages:
    if st.session_state.view_mode == "add_new":
        render_add_new_repo_view()
    elif st.session_state.selected_repo is not None:
        render_gallery_view()
        st.divider()
        render_selected_repo_view(st.session_state.selected_repo)
    else:
        render_gallery_view()