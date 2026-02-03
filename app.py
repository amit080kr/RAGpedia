import logging
import sys
from pathlib import Path
import os

import streamlit as st

from src.document_loader import ResearchPaperLoader
from src.vector_store import ResearchVectorStore
from src.retriever import ResearchPaperRetriever
from src.auth import AuthManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("research_paper_finder")

# --- Debug: Check imports immediately ---
try:
    import sentence_transformers
    logger.info("sentence_transformers successfully imported.")
except ImportError as e:
    logger.error(f"CRITICAL: Could not import sentence_transformers. Error: {e}")
except Exception as e:
    logger.error(f"CRITICAL: Unexpected error importing sentence_transformers. Error: {e}")
# ----------------------------------------

# Constants
VECTOR_STORE_DIR = "faiss_store"
# Make DATA_PATH empty by default — user can upload a CSV from the UI.
DATA_PATH = ""

# Initialize Streamlit state placeholders
status_placeholder = st.empty()
progress_bar = st.progress(0)

# Use session_state to persist uploaded data path, retriever and upload metadata between reruns
if 'data_path' not in st.session_state:
    st.session_state['data_path'] = DATA_PATH
if 'retriever' not in st.session_state:
    st.session_state['retriever'] = None
if 'auto_initialized' not in st.session_state:
    st.session_state['auto_initialized'] = False
if 'last_uploaded_name' not in st.session_state:
    st.session_state['last_uploaded_name'] = None


def log_error(e: Exception) -> str:
    """
    Log an error and return formatted error message.
    """
    logger.error(e, exc_info=True)
    return f"❌ Error: {str(e)}"


def get_documents():
    """
    Load research papers from the data file.
    """
    try:
        status_placeholder.info("📚 Loading research papers...")
        data_path = st.session_state.get('data_path', DATA_PATH)
        if not data_path:
            raise ValueError("No data path provided. Upload a CSV to proceed.")
        loader = ResearchPaperLoader(data_path)
        documents = loader.create_documents()
        if documents is None:
            raise ValueError("Failed to create documents from the provided CSV.")
        status_placeholder.success(f"✅ {len(documents)} research papers loaded successfully!")
        return documents
    except Exception as e:
        status_placeholder.error(log_error(e))
        return None


def create_new_vector_store():
    """
    Create a new FAISS vector store from scratch.
    """
    try:
        vector_store = ResearchVectorStore(store_path=VECTOR_STORE_DIR)

        status_placeholder.info("⚙️ Creating new vector store...")
        progress_bar.progress(40)

        # Create documents
        documents = get_documents()
        if not documents:
            return None
        progress_bar.progress(60)

        # Create embeddings and vector store
        status_placeholder.info("🔨 Generating embeddings and building FAISS index...")
        vector_store.create_vector_store(documents)
        progress_bar.progress(80)

        # Save vector store
        status_placeholder.info("💾 Saving vector store...")
        vector_store.save()
        progress_bar.progress(100)

        status_placeholder.success("✅ FAISS vector store created and saved successfully!")
        return vector_store

    except Exception as e:
        status_placeholder.error(log_error(e))
        return None


def load_existing_vector_store():
    """
    Load an existing FAISS vector store from disk.
    """
    try:
        status_placeholder.info("🔄 Loading existing FAISS vector store...")
        progress_bar.progress(30)
        vector_store = ResearchVectorStore.load(VECTOR_STORE_DIR)
        progress_bar.progress(100)
        status_placeholder.success("✅ FAISS vector store loaded successfully!")
        return vector_store
    except Exception as e:
        status_placeholder.error(log_error(e))
        return None


def initialize_retrieval_system(force_recreate: bool = False):
    """
    Initialize the retrieval system by loading or creating FAISS vector store.
    """
    try:
        # Decide which data path to use
        data_path = st.session_state.get('data_path', DATA_PATH)

        # Check if vector store directory exists and contains required files
        faiss_index_path = Path(VECTOR_STORE_DIR) / "faiss_index.bin"
        metadata_path = Path(VECTOR_STORE_DIR) / "metadata.pkl"
        documents_path = Path(VECTOR_STORE_DIR) / "documents.pkl"

        faiss_files_exist = all(p.exists() for p in [faiss_index_path, metadata_path, documents_path])

        vector_store = None
        if faiss_files_exist and not force_recreate:
            # Try to load existing vector store
            vector_store = load_existing_vector_store()
        else:
            # If we have a data path (uploaded CSV), create a new vector store
            if data_path and Path(data_path).exists():
                vector_store = create_new_vector_store()
            elif faiss_files_exist:
                # Fallback: try loading if files exist but previous load failed
                vector_store = load_existing_vector_store()
            else:
                # No FAISS store and no data path
                status_placeholder.info("No FAISS store found and no CSV uploaded. Upload a CSV to create one.")
                return None

        if not vector_store:
            return None

        # Initialize retriever
        status_placeholder.info("🤖 Initializing paper retriever...")
        retriever = ResearchPaperRetriever(vector_store)

        status_placeholder.empty()
        st.session_state['retriever'] = retriever
        return retriever

    except Exception as e:
        error_msg = log_error(e)
        status_placeholder.error(error_msg)
        return None


def render_search_results(query, nod, use_recency=False, similarity_threshold: float = 0.3):
    """
    Render search results for a query.
    """
    try:
        #Show spinner for paper retrieval
        with st.spinner("🔍 Searching for relevant papers..."):
            if use_recency:
                raw_results = st.session_state.retriever.retrieve_papers_with_recency(query, k=nod)
            else:
                raw_results = st.session_state.retriever.retrieve_papers(query, k=nod)

        # Filter by similarity threshold
        results = []
        for r in raw_results or []:
            score = r.get('similarity_score')
            try:
                if score is not None and float(score) >= similarity_threshold:
                    results.append(r)
            except Exception:
                # If score can't be converted, skip the result
                continue

        # Display results
        if not results:
            st.info(f"No matching papers found above similarity threshold {similarity_threshold}.")
            return

        st.subheader(f"Found {len(results)} relevant papers (threshold {similarity_threshold})")

        for i, paper in enumerate(results, 1):
            with st.expander(f"{i}. {paper.get('title', 'Untitled')} ({paper.get('year', 'N/A')})"):
                st.write(f"**Authors:** {paper.get('authors', 'N/A')}")
                st.write(f"**Published in:** {paper.get('venue', 'N/A')}")
                st.write(f"**Citations:** {paper.get('citations', paper.get('n_citation', 'N/A'))}")
                st.write(f"**Abstract:** {paper.get('abstract', 'N/A')}")
                st.write(f"**Similarity Score:** {float(paper.get('similarity_score')):.4f}")

    except ValueError as e:
        st.warning(str(e))
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")


def login_page():
    st.title("🔐 RAGpedia Login")
    
    auth_manager = AuthManager()
    
    tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Forgot Password"])
    
    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if auth_manager.login_user(username, password):
                st.session_state.authenticated = True
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid username or password")
                
    with tab2:
        new_user = st.text_input("Username", key="signup_user")
        new_pass = st.text_input("Password", type="password", key="signup_pass")
        question = st.selectbox("Security Question", [
            "What was the name of your first pet?",
            "What is your mother's maiden name?",
            "What city were you born in?",
            "What was your favorite book as a child?"
        ], key="signup_q")
        answer = st.text_input("Security Answer", key="signup_a", help="This will be used to recover your password.")
        
        if st.button("Sign Up"):
            if not new_user or not new_pass or not answer:
                st.error("Please fill in all fields")
            else:
                success, msg = auth_manager.register_user(new_user, new_pass, question, answer)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

    with tab3:
        reset_user = st.text_input("Username to Reset", key="reset_user")
        if reset_user:
            user_q = auth_manager.get_user_question(reset_user)
            if user_q:
                st.write(f"**Security Question:** {user_q}")
                reset_answer = st.text_input("Answer", key="reset_a")
                reset_pass = st.text_input("New Password", type="password", key="reset_p")
                if st.button("Reset Password"):
                    success, msg = auth_manager.reset_password(reset_user, reset_answer, reset_pass)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
            else:
                st.info("Enter a valid username to see your security question.")


def main():
    """Main application function."""
    # Clear progress indicators
    progress_bar.empty()

    # Set up the main page
    st.title("🔍 Academic Research Paper Finder (FAISS)")
    
    # Hide the "Limit 200MB per file" text using CSS
    st.markdown("""
        <style>
        [data-testid="stFileUploader"] section > input + div > small {
            display: none;
        }
        /* Fallback for different streamlit versions */
        [data-testid="stFileUploader"] small {
            display: none;
        }
        </style>
    """, unsafe_allow_html=True)

    st.write("""
    Find relevant academic papers using FAISS for efficient semantic search. Enter your research topic 
    or question to discover papers related to your area of interest.
    """)

    # --- File uploader ---
    #st.markdown("**Upload your CSV library (optional). If you don't upload, the app will look for an existing vector store in 'faiss_store'.**")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], help="Upload a CSV with columns like title, abstract, authors, year, venue, citations.")

    # small explanatory caption about the uploader UI
    #st.caption("Note: 'Limit 200MB per file • CSV' is Streamlit's uploader UI. It means you can drag a CSV onto the box; Streamlit limits each file to 200MB and only accepts CSV files because `type=['csv']`.")

    # Handle upload and auto-initialize only once per upload
    if uploaded_file is not None:
        # Only re-process if the uploaded file is new or hasn't been auto-initialized
        if st.session_state.get('last_uploaded_name') != uploaded_file.name or not st.session_state.get('auto_initialized'):
            upload_path = Path(os.getcwd()) / "uploaded_library.csv"
            with open(upload_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state['data_path'] = str(upload_path)
            st.session_state['last_uploaded_name'] = uploaded_file.name
            status_placeholder.success(f"Uploaded file saved to {st.session_state['data_path']}")

            # Auto-initialize (only once per upload)
            status_placeholder.info("Auto-initializing FAISS vector store from uploaded CSV...")
            retriever = initialize_retrieval_system(force_recreate=True)
            if retriever:
                st.success("Retriever initialized from uploaded CSV.")
                st.session_state['auto_initialized'] = True
            else:
                st.warning("Failed to initialize retriever from uploaded CSV. Check logs/details.")

    # If no retriever yet, try to load existing FAISS store on startup (but do NOT initialize on Search)
    if st.session_state.get('retriever') is None:
        faiss_exists = Path(VECTOR_STORE_DIR).exists() and any(Path(VECTOR_STORE_DIR).iterdir())
        if faiss_exists:
            initialize_retrieval_system()

    # Search interface — use keys so Streamlit preserves values across re-runs
    query = st.text_input(
        "Enter your research topic or question:",
        placeholder="e.g., 'Transformer models in NLP' or 'Quantum computing for cryptography'",
        key='query'
    )

    nod = st.text_input("Number of documents",
                        placeholder="e.g. 5",
                        key='nod')

    # Similarity threshold control (slider) — configurable from UI
    similarity_threshold = st.slider(
        "Minimum similarity threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.01,
        help="Only return papers with similarity score >= threshold.",
        key='similarity_threshold'
    )

    # Recency toggle with help text — give it a key so toggling doesn't clear other inputs
    use_recency = st.checkbox(
        "Prioritize recent papers", 
        value=False, 
        help="When checked, returned result based on a joint combined score and in sorted order of years.",
        key='use_recency'
    )

    search_button = st.button("Search")

    # Run search when button is clicked — do NOT auto-initialize here; require upload or existing FAISS store
    if search_button:
        # Ensure inputs persisted
        query_val = st.session_state.get('query', '').strip()
        nod_val = st.session_state.get('nod', '').strip()
        use_recency_val = st.session_state.get('use_recency', False)
        similarity_threshold_val = st.session_state.get('similarity_threshold', 0.3)

        if not query_val:
            st.warning("Please enter a query before searching.")
            return

        # If retriever is not initialized, do not attempt to initialize on Search — ask user to upload or place FAISS store
        if not st.session_state.get('retriever'):
            st.warning("Retriever is not initialized. Upload a CSV to create the vector store or ensure 'faiss_store' exists.")
            return

        # Validate number of documents
        try:
            num_of_documents = int(nod_val) if nod_val else 5
            if num_of_documents <= 0:
                st.warning("Please enter a positive number for 'Number of documents'.")
            else:
                # Perform search — pass user-controlled similarity threshold
                render_search_results(query_val, num_of_documents, use_recency_val, similarity_threshold=float(similarity_threshold_val))
        except ValueError:
            st.warning("Please enter a valid integer for 'Number of documents'.")


if __name__ == "__main__":
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if st.session_state.authenticated:
        with st.sidebar:
            st.write(f"Welcome!")
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.rerun()
        main()
    else:
        login_page()
