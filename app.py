"""
Streamlit Application for Academic RAG Chatbot.

A conversational interface for querying academic bibliography
in Probability and Statistics courses.

Run with: streamlit run app.py
"""

import sys
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta
import re
import time
import json

# Fix for Streamlit Cloud + ChromaDB (requires newer sqlite3)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import extra_streamlit_components as stx

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config, get_config
from src.utils.logger import setup_logger, get_logger
from src.retrieval.retriever import Retriever, create_retriever_from_config
from src.retrieval.confidence import ConfidenceCalculator, ConfidenceLevel
from src.generation.llm_client import GeminiClient, create_client_from_config
from src.generation.prompts import PromptTemplates
from src.retrieval.query_processor import QueryProcessor
from src.utils.text_processing import normalize_latex


# Page configuration
st.set_page_config(
    page_title="Academic RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better styling
CUSTOM_CSS = """
<style>
    /* Main container styling */
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 2rem;
    }
    
    /* Confidence badges */
    .confidence-high {
        background-color: #d4edda;
        color: #155724;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    
    .confidence-medium {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    
    .confidence-low {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    
    /* Source cards */
    .source-card {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0 0.25rem 0.25rem 0;
    }
    
    /* Chat messages */
    .user-message {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .assistant-message {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    /* Stat cards */
    .stat-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
    }
</style>
"""


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "config" not in st.session_state:
        st.session_state.config = get_config()
    
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    
    if "llm_client" not in st.session_state:
        st.session_state.llm_client = None
    
    if "confidence_calculator" not in st.session_state:
        st.session_state.confidence_calculator = ConfidenceCalculator()
    
    if "prompt_templates" not in st.session_state:
        st.session_state.prompt_templates = PromptTemplates()
        
    if "query_processor" not in st.session_state:
        st.session_state.query_processor = None
        
    if "rate_limit_checked" not in st.session_state:
        st.session_state.rate_limit_checked = False

    if "rate_limit_data" not in st.session_state:
        st.session_state.rate_limit_data = None


def check_rate_limit(cookie_manager, config) -> tuple[bool, str]:
    """
    Check if the user has exceeded the rate limit.
    
    Args:
        cookie_manager: CookieManager instance.
        config: Config instance.
        
    Returns:
        Tuple of (allowed, message).
    """
    if not config.enable_session_limit:
        return True, ""
        
    # Cookie key
    cookie_name = "rate_limit_data"
    
    # 1. Try to get data from session state first (fastest, most up-to-date)
    if st.session_state.rate_limit_data:
        data = st.session_state.rate_limit_data
    else:
        # 2. Fallback to cookie (first load)
        try:
            cookie_data = cookie_manager.get(cookie_name)
            if cookie_data:
                if isinstance(cookie_data, str):
                    data = json.loads(cookie_data)
                else:
                    data = cookie_data
            else:
                data = None
        except Exception:
            data = None
            
    current_time = time.time()
    window_seconds = config.session_window_minutes * 60
    
    if not data:
        # Initialize
        data = {
            "count": 0,
            "start_time": current_time
        }
    else:
        # Check if window expired
        start_time = data.get("start_time", current_time)
        if current_time - start_time > window_seconds:
            # Reset window
            data = {
                "count": 0,
                "start_time": current_time
            }
    
    # Update session state with the guaranteed data
    st.session_state.rate_limit_data = data
            
    # Check limit
    if data["count"] >= config.session_limit:
        time_left_mins = int((window_seconds - (current_time - data["start_time"])) / 60) + 1
        return False, f"⚠️ You have reached the limit of {config.session_limit} questions per {config.session_window_minutes} minutes. Please wait {time_left_mins} minutes."
        
    return True, ""


def update_rate_limit(cookie_manager, config):
    """
    Increment the rate limit counter in the cookie.
    
    Args:
        cookie_manager: CookieManager instance.
        config: Config instance.
    """
    if not config.enable_session_limit:
        return
        
    cookie_name = "rate_limit_data"
    current_time = time.time()
    window_seconds = config.session_window_minutes * 60
    
    # 1. Get from session state (should be populated by check_rate_limit)
    data = st.session_state.rate_limit_data
    
    if not data:
        # Should not happen usually
         data = {
            "count": 0,
            "start_time": current_time
        }
    
    # Check reset again just in case (though check_rate_limit handles this)
    start_time = data.get("start_time", current_time)
    if current_time - start_time > window_seconds:
        data = {
            "count": 1,
            "start_time": current_time
        }
    else:
        data["count"] += 1
            
    # 2. Update session state IMMEDIATELY
    st.session_state.rate_limit_data = data
            
    # 3. Update cookie (for persistence next session)
    try:
        cookie_manager.set(cookie_name, json.dumps(data), expires_at=datetime.now() + timedelta(days=1))
    except Exception as e:
        print(f"Failed to set cookie: {e}")


def load_components() -> tuple[Optional[Retriever], Optional[GeminiClient]]:
    """
    Load retriever and LLM client components.
    
    Returns:
        Tuple of (retriever, llm_client) or (None, None) on error.
    """
    config = st.session_state.config
    
    # Load retriever
    if st.session_state.retriever is None:
        try:
            st.session_state.retriever = create_retriever_from_config(config)
        except FileNotFoundError as e:
            st.error(
                "📚 **Vector database not found!**\n\n"
                "Please run the ingestion script first:\n"
                "```bash\npython ingest.py --data-dir data/raw\n```"
            )
            return None, None
        except Exception as e:
            st.error(f"Failed to initialize retriever: {e}")
            return None, None
    
    # Load LLM client
    if st.session_state.llm_client is None:
        if not config.google_api_key:
            st.error(
                "🔑 **Google API key not configured!**\n\n"
                "Please add your API key to the `.env` file:\n"
                "```\nGOOGLE_API_KEY=your_api_key_here\n```"
            )
            return st.session_state.retriever, None
        
        # Check for placeholder key
        if "YOUR_" in config.google_api_key or len(config.google_api_key) < 20:
            st.error(
                "⚠️ **Invalid Google API Key detected!**\n\n"
                "It looks like you are using a placeholder key. "
                "Please update your secrets with a valid key from Google AI Studio."
            )
            return st.session_state.retriever, None
        
        try:
            st.session_state.llm_client = create_client_from_config(config)
        except Exception as e:
            st.error(f"Failed to initialize LLM client: {e}")
            return st.session_state.retriever, None
            
    # Always refresh QueryProcessor to pick up prompt changes
    if config.google_api_key:
         st.session_state.query_processor = QueryProcessor(api_key=config.google_api_key)
    
    return st.session_state.retriever, st.session_state.llm_client


def render_sidebar() -> dict:
    """
    Render the sidebar with settings and filters.
    
    Returns:
        Dictionary with current settings.
    """
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # --- DEBUG SECTION START ---
        with st.expander("🛠️ Deployment Debugger", expanded=False):
            st.write(f"**API Key Present:** {'✅ Yes' if config.google_api_key else '❌ No'}")
            st.write(f"**Key Length:** {len(config.google_api_key)}")
            st.write(f"**Prefix:** {config.google_api_key[:4]}..." if config.google_api_key else "N/A")
            st.code(f"ENABLE_SESSION_LIMIT: {os.getenv('ENABLE_SESSION_LIMIT')}")
            st.code(f"SESSION_LIMIT: {os.getenv('SESSION_LIMIT')}")
        # --- DEBUG SECTION END ---

        # Reset session button (hidden for clarity)ion
        st.subheader("🤖 Model Info")
        config = st.session_state.config
        st.info(f"""
        **LLM:** {config.llm_model}  
        **Embeddings:** sentence-transformers  
        **Model:** {config.embedding_model.split('/')[-1]}
        """)
        
        st.divider()
        
        # Filters
        st.subheader("🔍 Search Filters")
        
        # Source type filter
        source_type = st.selectbox(
            "Source Type",
            options=["All", "Books", "Lecture Notes", "Exercises"],
            index=0,
            help="Filter search to specific source types"
        )
        
        # Author filter (populated from database)
        authors = ["All"]
        if st.session_state.retriever:
            try:
                db_authors = st.session_state.retriever.get_unique_values("author")
                authors.extend([a for a in db_authors if a and a != "unknown"])
            except Exception:
                pass
        
        author_filter = st.selectbox(
            "Author",
            options=authors,
            index=0,
            help="Filter search to specific author"
        )
        
        st.divider()
        
        # Retrieval settings
        st.subheader("🎛️ Retrieval Settings")
        
        top_k = st.slider(
            "Number of sources",
            min_value=3,
            max_value=30,
            value=10,
            help="Number of document chunks to retrieve"
        )
        
        temperature = st.slider(
            "Response creativity",
            min_value=0.0,
            max_value=1.0,
            value=config.llm_temperature,
            step=0.1,
            help="Lower = more factual, Higher = more creative"
        )
        
        # Context expansion toggle
        expand_context = st.checkbox(
            "🔍 Expand context (slower, more thorough)",
            value=True,
            help="Fetch adjacent pages from matched sources for more complete answers"
        )
        
        expand_pages = 1
        if expand_context:
            expand_pages = st.slider(
                "Pages to expand",
                min_value=1,
                max_value=3,
                value=1,
                help="Number of pages before/after to include (max 3 pages)"
            )
            
        # Query Expansion toggle
        enable_expansion = st.checkbox(
            "✨ Enable Query Expansion",
            value=True,
            help="Rewrites your query to be more precise and academic before searching"
        )
        
        st.divider()
        
        # Database statistics
        st.subheader("📊 Database Statistics")
        
        if st.session_state.retriever:
            try:
                stats = st.session_state.retriever.get_collection_stats()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Chunks", f"{stats.get('total_documents', 0):,}")
                with col2:
                    # Count unique files
                    unique_files = st.session_state.retriever.get_unique_values("source_file")
                    st.metric("PDFs", len(unique_files))
            except Exception:
                st.warning("Unable to load statistics")
        else:
            st.warning("Database not connected")
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        # Map source type to database value
        source_type_map = {
            "All": None,
            "Books": "book",
            "Lecture Notes": "lecture_notes",
            "Exercises": "exercises"
        }
        
        return {
            "source_type": source_type_map.get(source_type),
            "author": author_filter if author_filter != "All" else None,
            "top_k": top_k,
            "temperature": temperature,
            "expand_context": expand_context,
            "expand_pages": expand_pages,
            "enable_expansion": enable_expansion
        }


def render_confidence_badge(confidence_result) -> None:
    """Render a confidence badge."""
    level = confidence_result.level
    score = confidence_result.score
    
    emoji = confidence_result.emoji
    color = confidence_result.color
    
    st.markdown(
        f"""<span style="background-color: {color}; color: white; 
        padding: 0.25rem 0.75rem; border-radius: 1rem; font-size: 0.875rem;">
        {emoji} {level.value.upper()} ({score:.0%})
        </span>""",
        unsafe_allow_html=True
    )


def render_sources(documents: list) -> None:
    """Render the sources panel with highlighting."""
    if not documents:
        return
    
    with st.expander("📚 Sources Consulted", expanded=False):
        for i, doc in enumerate(documents, 1):
            # Format source type icon
            type_icons = {
                "book": "📖",
                "lecture_notes": "📝",
                "exercises": "✏️"
            }
            icon = type_icons.get(doc.source_type, "📄")
            
            # Build citation
            citation_parts = [doc.author] if doc.author else []
            if doc.chapter:
                citation_parts.append(f"Chapter {doc.chapter}")
            citation_parts.append(f"page {doc.page}")
            citation = ", ".join(citation_parts)
            
            st.markdown(f"**{i}. {icon} {citation}**")
            
            # Source Highlighting Logic
            # We want to show the parent content with the child chunk highlighted
            content_to_show = doc.text # Fallback
            
            if hasattr(doc, 'parent_content') and doc.parent_content:
                parent = doc.parent_content
                child = doc.text
                
                # Try to find the child text in the parent
                # We normalize whitespace to improve matching chances
                
                # Escape the child text for regex
                escaped_child = re.escape(child.strip())
                # Allow flexible whitespace
                pattern = escaped_child.replace(r'\ ', r'\s+')
                
                match = re.search(pattern, parent, re.IGNORECASE)
                
                if match:
                    # Highlight the match
                    start, end = match.span()
                    
                    # Extract context around the match (e.g., 200 chars before and after)
                    context_start = max(0, start - 200)
                    context_end = min(len(parent), end + 200)
                    
                    prefix = "..." if context_start > 0 else ""
                    suffix = "..." if context_end < len(parent) else ""
                    
                    pre_text = parent[context_start:start]
                    highlighted_text = parent[start:end]
                    post_text = parent[end:context_end]
                    
                    # Streamlit markdown highlighting (using background color or bold/color)
                    # :orange[...] or **...** 
                    # Let's use a yellow background if possible, or bold orange
                    content_to_show = f"{prefix}{pre_text} :orange[**{highlighted_text}**] {post_text}{suffix}"
                else:
                    # Fallback if exact match fails (e.g. text cleaning differences)
                    content_to_show = f"... {doc.text} ..."
            
            # Text Cleaning for UI
            # Remove PDF artifacts like [** ... **]
            content_to_show = re.sub(r'\[\*\*|\*\*\]', '', content_to_show)
            # Remove page headers/footers if any left (e.g. "CHAPTER 3...")
            # We can be aggressive here for UI
            content_to_show = re.sub(r'CHAPTER \d+\..*?(?=\n)', '', content_to_show)
            # Clean up excessive whitespace
            content_to_show = re.sub(r'\s+', ' ', content_to_show).strip()
            
            st.markdown(f"> {content_to_show}")
            st.markdown(f"<small>Similarity: {doc.similarity:.0%}</small>", 
                       unsafe_allow_html=True)
            st.divider()


def process_query(
    query: str,
    retriever: Retriever,
    llm_client: GeminiClient,
    settings: dict
) -> dict:
    """
    Process a user query through the RAG pipeline.
    
    Args:
        query: User's question.
        retriever: Retriever instance.
        llm_client: LLM client instance.
        settings: Current settings from sidebar.
    
    Returns:
        Dictionary with response data.
    """
    # Check if context expansion is enabled
    expand_context = settings.get("expand_context", True)
    expand_pages = settings.get("expand_pages", 2)
    top_k = settings.get("top_k", 5)
    enable_expansion = settings.get("enable_expansion", True)
    
    # 1. Query Expansion
    optimized_query = query
    if enable_expansion and st.session_state.query_processor:
        with st.status("🧠 Optimizing query...", expanded=False) as status:
            try:
                # Pass recent history for context (exclude current user message which is 'query')
                # st.session_state.chat_history contains the current message if called after append?
                # In main(), we append user message BEFORE calling process_query.
                # So chat_history[-1] is the current message. We want steps before that.
                
                history_for_context = st.session_state.chat_history[:-1]
                
                optimized_query = st.session_state.query_processor.generate_optimized_query(
                    original_query=query,
                    chat_history=history_for_context
                )
                if optimized_query != query:
                    status.update(
                        label=f"Query optimized: {optimized_query}", 
                        state="complete"
                    )
                else:
                    status.update(
                        label="Query optimization complete (using original query)", 
                        state="complete"
                    )
            except Exception as e:
                status.update(label="Optimization failed", state="error")
                optimized_query = query
    
    # Retrieve context with or without expansion using OPTIMIZED query
    # Build filters
    filters = {}
    
    # Source Type Filter (already mapped to backend values by render_sidebar)
    selected_source_type = settings.get("source_type")
    if selected_source_type:
        filters["source_type"] = selected_source_type

    # Author Filter (already filtered to non-"All" by render_sidebar)
    selected_author = settings.get("author")
    if selected_author:
        filters["author"] = selected_author
        
    documents = []
    context = ""

    # Retrieve context with or without expansion using OPTIMIZED query
    if expand_context:
        # Dynamic Limit: allow more chunks if user requested more top_k or expansion
        # Rough calc: top_k * (pages_before + pages_after + 1)
        # But we don't want to explode. Let's start with a generous cap.
        # If user asks for 30 sources + 3 pages expansion, that's huge.
        # We'll set max_total_docs to allow at least what the user explicitly asked for in top_k, 
        # plus expansion room. 
        max_total_limit = top_k * 5 # Allow 5 chunks per source on average
        
        context, documents = retriever.retrieve_context_expanded(
            query=optimized_query,
            top_k=top_k,
            expand_pages=expand_pages,
            filters=filters if filters else None,
            max_total_docs=max_total_limit
        )
    else:
        context, documents = retriever.retrieve_context(
            query=optimized_query,
            top_k=top_k,
            filters=filters if filters else None
        )
    
    # Calculate confidence
    similarities = retriever.get_similarities(documents)
    confidence_calc = st.session_state.confidence_calculator
    confidence_result = confidence_calc.calculate(similarities)
    
    # NEW: Smart Relevance Filtering
    # Instead of failing on the first False Positive, we check the top N results.
    # We filter out the irrelevant ones (e.g. "proof omitted") and keep the rest.
    # If valid documents remain, we recalculate confidence.
    if documents and st.session_state.query_processor and not filters:
        
        with st.status("🕵️ Verifying result relevance...", expanded=False) as status:
            valid_documents = []
            checked_count = 0
            max_checks = 5 # Check up to top 5 docs
            
            for doc in documents[:max_checks]:
                checked_count += 1
                # Prefer context_text if available
                text_to_check = getattr(doc, 'context_text', doc.text)
                
                is_relevant = st.session_state.query_processor.verify_content_relevance(
                    query=optimized_query,
                    content=text_to_check
                )
                
                if is_relevant:
                    valid_documents.append(doc)
                else:
                    # Log or update status for feedback
                    # status.write(f"⚠️ Filtered out irrelevant document (Source: {doc.get_citation()})")
                    pass
            
            # Add back any remaining unchecked docs (we only verified the top 5)
            # If the top 5 were all bad, we might want to discard the rest too? 
            # Strategy: If we found ANY valid docs in top 5, we keep them + the rest.
            # If we found NONE in top 5, likely the rest are bad too or irrelevant.
            # Let's keep it simple: valid_documents contains ONLY verified good docs from top 5 
            # PLUS the unverified rest? No, unsafe. 
            # Let's trust the top 5 verification. If we filter top 5, we likely should filter 
            # the list to ONLY valid ones found.
            
            if len(valid_documents) < len(documents[:max_checks]):
                status.write(f"⚠️ Filtered {checked_count - len(valid_documents)} irrelevant result(s).")
            
            # Update the documents list to use ONLY the valid ones (plus maybe the tail if we want only strict top-k checking)
            # Strict approach: The final list is valid_documents + documents[max_checks:] 
            # (assuming tail is less likely to be "proof omitted" false positives? No, probably similar).
            # SAFE APPROACH: If we filtered the top, we should probably stick to the verified ones 
            # or continue checking (too slow).
            # Let's use: valid_documents + documents[max_checks:]
            # But if top 5 were all "proof omitted", usually the rest are worse matches.
            
            # Updated Strategy:
            # - If we found VALID docs in top 5, great. We prioritize them.
            # - If we found 0 valid docs in top 5, we consider the query failed for this retrieval batch.
            
            if valid_documents:
                # We have some good results! Use them.
                # Append the rest of the unverified tail (optional, but good for context)
                # valid_documents.extend(documents[max_checks:]) 
                # actually, let's stick to valid ones to increase quality
                
                documents = valid_documents + documents[max_checks:]
                status.update(label=f"✅ Verified {len(valid_documents)} relevant source(s)", state="complete")
                
                # RECALCULATE CONFIDENCE with the filtered list
                # This is crucial. If we removed the top result (sim=0.9) and left with #2 (sim=0.8),
                # confidence might drop or stay high depending on density.
                similarities = retriever.get_similarities(documents)
                confidence_result = confidence_calc.calculate(similarities)
                
            else:
                # Top 5 were all IRRELEVANT.
                status.update(
                    label="⚠️ Top results were False Positives (e.g. proof omitted). Forcing fallback search...", 
                    state="error"
                )
                # Effective list is empty or just the tail (which is likely bad)
                # Force low confidence to trigger deep search
                confidence_result.level = ConfidenceLevel.LOW
                confidence_result.score = 0.0
                confidence_result.should_respond = True
                confidence_result.message = "Initial results deemed irrelevant. Trying deep search..."
                # We can empty the documents list to avoid showing garbage
                documents = []

    # CRITICAL: Iterative Deep Search (Source-by-Source verification)
    # If confidence is still LOW/INSUFFICIENT, we force a deep search.
    # We iterate through EVERY author in the DB, retrieve their best chunks,
    # and VERIFY them one by one. We stop as soon as we find a VALID answer.
    if (confidence_result.level == ConfidenceLevel.LOW or 
        confidence_result.level == ConfidenceLevel.INSUFFICIENT) and not filters:
        
        with st.status("🔍 Low confidence — starting Deep Search...", expanded=True) as status:
            try:
                # Check scope first
                query_proc = st.session_state.query_processor
                if query_proc and query_proc.classify_scope(optimized_query):
                    
                    # Get all authors to iterate over
                    status.write("📚 Fetching all available sources...")
                    all_authors = retriever.get_unique_values("author")
                    # Move "Ferrari" and "Walpole" to front if present (heuristics for likely hits)
                    priority_authors = ["Ferrari", "Walpole", "Devore"]
                    all_authors = sorted(all_authors, key=lambda x: 0 if any(p in x for p in priority_authors) else 1)
                    
                    status.write(f"🕵️ Deep Search: Will check up to {len(all_authors)} sources...")
                    
                    found_verified_docs = []
                    
                    # Iterate through authors
                    for idx, author in enumerate(all_authors):
                        status.update(label=f"🔍 Checking source {idx+1}/{len(all_authors)}: {author}...", state="running")
                        
                        # Rate limit protection (sleep to avoid 429s)
                        time.sleep(1.0) # 1 second delay between authors
                        
                        # DUAL RETRIEVAL WITH CONTEXT EXPANSION
                        # We use context expansion to ensuring we catch the "Proof" or "Definition" 
                        # even if the keyword match was just on the Section Header.
                        author_docs = []
                        seen_ids = set()
                        
                        # 1. Search with ORIGINAL query (best for Spanish books like Ferrari)
                        try:
                            # Use top_k=3 and expand_pages=2 to ensure we get the full proof (e.g. if it spills to next page)
                            docs_original = retriever.retrieve_with_context_expansion(
                                query=query, 
                                top_k=3, 
                                expand_pages=2,
                                max_total_docs=30,
                                author=author
                            )
                            for d in docs_original:
                                if d.chunk_id not in seen_ids:
                                    author_docs.append(d)
                                    seen_ids.add(d.chunk_id)
                        except Exception:
                            pass
                            
                        # 2. Search with OPTIMIZED query (best for English books)
                        try:
                            # Use top_k=3 and expand_pages=2
                            docs_optimized = retriever.retrieve_with_context_expansion(
                                query=optimized_query,
                                top_k=3, 
                                expand_pages=2,
                                max_total_docs=30,
                                author=author
                            )
                            for d in docs_optimized:
                                if d.chunk_id not in seen_ids:
                                    author_docs.append(d)
                                    seen_ids.add(d.chunk_id)
                        except Exception:
                            pass
                        
                        if not author_docs:
                            continue
                            
                        # Batch Verification Logic:
                        # If ANY document in this expanded batch is relevant (e.g. the Theorem Statement), 
                        # we imply the neighbors (Proof, Examples) are also relevant context.
                        # We return the WHOLE batch to the LLM to ensure nothing is lost.
                        
                        batch_is_relevant = False
                        # Check ALL docs (sorted by similarity) to find an anchor
                        # We removed the [:5] limit to ensure we don't miss the relevant chunk
                        # if it's ranked slightly lower.
                        docs_to_check = sorted(author_docs, key=lambda x: x.similarity, reverse=True)
                        
                        for doc in docs_to_check:
                            # Use parent_content if available
                            text_to_check = getattr(doc, 'parent_content', None) or getattr(doc, 'context_text', None) or doc.text
                            
                            is_relevant = query_proc.verify_content_relevance(query, text_to_check)
                            
                            if is_relevant:
                                batch_is_relevant = True
                                break # Found a solid anchor!
                        
                        if batch_is_relevant:
                            # Found the answer! Return the whole context.
                            found_verified_docs.extend(author_docs)
                            status.update(label=f"✅ Deep Search found answer in {author}!", state="complete")
                            status.write(f"✅ ACCEPTED: {author} - Content verified relevant.")
                            break # Stop iterating authors
                        else:
                            status.write(f"❌ REJECTED: {author} - Content verified IRRELEVANT (e.g. proof omitted).")

                        # If we found at least one verified doc for this author, we consider if it's enough.
                        if found_verified_docs:
                            # We stop looking at other authors to save time/cost, 
                            # assuming this author provides the answer.
                            break
                    
                    if found_verified_docs:
                        # Success!
                        # Update context and documents
                        documents = found_verified_docs
                        
                        # Recalculate confidence based on these verified docs
                        # We artificially boost their similarity because they passed the LLM verification
                        # This ensures the UI shows High/Medium confidence
                        for d in documents:
                            d.similarity = max(d.similarity, 0.90) # Boost high to confirm
                            
                        sims = [d.similarity for d in documents]
                        confidence_result = confidence_calc.calculate(sims)
                        
                        # Format context
                        context_parts = []
                        for i, doc in enumerate(documents, 1):
                            citation = doc.get_citation()
                            context_parts.append(f"[Source {i} - {citation}]\n{doc.context_text}")
                        context = "\n\n---\n\n".join(context_parts)
                        
                        status.update(
                            label=f"✅ Deep Search found answer in {documents[0].author}!", 
                            state="complete"
                        )
                    else:
                        status.update(
                            label="❌ Deep Search exhausted all sources without finding a definitive answer.", 
                            state="error"
                        )
                        
                else:
                    status.update(label="Query determined to be out of scope.", state="complete")
                    
            except Exception as e:
                st.error(f"Deep search error: {e}")
                status.update(label="Deep Search failed", state="error")
    
    # Check if we should respond
    if not confidence_result.should_respond:
        return {
            "answer": confidence_result.message,
            "sources": [],
            "documents": documents,
            "confidence": confidence_result,
            "should_warn": True
        }
    
    # Prepare sources for LLM
    sources = [
        {
            "author": doc.author,
            "page": doc.page,
            "chapter": doc.chapter,
            "source_type": doc.source_type,
            "snippet": doc.get_snippet(200)
        }
        for doc in documents
    ]
    
    # Get chat history for context
    chat_history = []
    for msg in st.session_state.chat_history[-6:]:  # Last 3 turns
        chat_history.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    # Generate response
    # We use the ORIGINAL query so the LLM knows the target language (Spanish),
    # but we search using the OPTIMIZED (English) query.
    # To help the LLM connect the dots, we append the optimized query to the input.
    generation_query = query
    if optimized_query != query:
        generation_query = f"{query}\n[Context: Search performed for '{optimized_query}']"
        
    # Update temperature from slider
    llm_client.temperature = settings.get("temperature", 0.2)
        
    result = llm_client.generate(
        query=generation_query, 
        context=context,
        chat_history=chat_history,
        sources=sources
    )
    
    # Normalize LaTeX in the answer
    final_answer = result.answer if result.success else result.error
    
    if result.success:
        try:
            final_answer = normalize_latex(final_answer)
        except Exception as e:
            print(f"Latex normalization failed: {e}")
            # Fallback to raw answer if normalization crashes
            # final_answer = result.answer 
            pass
        
    return {
        "answer": final_answer,
        "sources": sources,
        "documents": documents,
        "confidence": confidence_result,
        "should_warn": confidence_result.level == ConfidenceLevel.LOW,
        "success": result.success,
        "optimized_query": optimized_query if optimized_query != query else None
    }


def render_chat_message(message: dict) -> None:
    """Render a single chat message."""
    with st.chat_message(message["role"]):
        # Render confidence if present
        if message.get("confidence"):
            render_confidence_badge(message["confidence"])
            st.write("")  # Spacing
        
        # Render warning if needed
        if message.get("should_warn"):
            st.warning(
                "⚠️ The bibliography has limited information on this topic. "
            )
            
        # Render optimized query if present
        if message.get("optimized_query"):
            st.caption(f"✨ Interpreted as: *{message['optimized_query']}*")
        
        # Render message content
        st.markdown(message["content"])
        
        # Render sources if present
        if message.get("documents"):
            render_sources(message["documents"])


def main() -> None:
    """Main application entry point."""
    # Apply custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Setup logging
    logger = get_logger("app")
    
    # Header
    st.title("📚 Academic RAG Chatbot")
    st.markdown(
        "*Your study assistant for Probability & Statistics bibliography*"
    )
    st.divider()
    
    # Load components
    retriever, llm_client = load_components()
    
    # Render sidebar and get settings
    settings = render_sidebar()
    
    # Cookie Manager for Rate Limiting
    # We initialize it here to ensure it's available
    cookie_manager = stx.CookieManager()
    
    # Check if components are loaded
    if retriever is None:
        st.warning(
            "Please set up the database first by running the ingestion script."
        )
        st.code("python ingest.py --data-dir data/raw", language="bash")
        return
    
    if llm_client is None:
        st.warning(
            "Please configure your Google API key in the `.env` file to enable responses."
        )
        st.info(
            "You can still search the bibliography, but responses won't be generated."
        )
    
    # Display chat history
    for message in st.session_state.chat_history:
        render_chat_message(message)
    
    # Chat input
    # First check rate limit to see if we should disable input or show warning
    is_allowed, limit_message = check_rate_limit(cookie_manager, config=st.session_state.config)
    
    prompt = st.chat_input(
        "Ask a question about Probability & Statistics..." if is_allowed else "Rate limit reached.",
        disabled=not is_allowed
    )
    
    if not is_allowed:
        st.warning(limit_message)
        
    if prompt and is_allowed:
        # Add user message to history
        user_message = {"role": "user", "content": prompt}
        st.session_state.chat_history.append(user_message)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process query
        with st.chat_message("assistant"):
            if llm_client:
                with st.spinner("🔍 Searching bibliography..."):
                    try:
                        result = process_query(
                            query=prompt,
                            retriever=retriever,
                            llm_client=llm_client,
                            settings=settings
                        )
                        
                        # Render confidence
                        render_confidence_badge(result["confidence"])
                        st.write("")
                        
                        # Show warning if needed
                        if result.get("should_warn"):
                            st.warning(
                                "⚠️ The bibliography has limited information on this topic."
                            )
                        
                        # Render answer
                        st.markdown(result["answer"])
                        
                        # Render sources
                        render_sources(result["documents"])
                        
                        # Add to history
                        assistant_message = {
                            "role": "assistant",
                            "content": result["answer"],
                            "documents": result["documents"],
                            "confidence": result["confidence"],
                            "should_warn": result.get("should_warn", False),
                            "optimized_query": result.get("optimized_query")
                        }
                        st.session_state.chat_history.append(assistant_message)
                        
                        # Increment Limit Counter (only on success)
                        update_rate_limit(cookie_manager, st.session_state.config)
                        
                    except Exception as e:
                        error_msg = f"An error occurred: {str(e)}"
                        st.error(error_msg)
                        logger.error(f"Query processing error: {e}")
            else:
                # Retrieval only mode
                with st.spinner("🔍 Searching bibliography..."):
                    try:
                        context, documents = retriever.retrieve_context(
                            query=prompt,
                            top_k=settings.get("top_k", 5)
                        )
                        
                        # Calculate confidence
                        similarities = retriever.get_similarities(documents)
                        confidence = st.session_state.confidence_calculator.calculate(
                            similarities
                        )
                        
                        render_confidence_badge(confidence)
                        st.write("")
                        
                        st.info(
                            "📚 **Retrieved Sources** (LLM not configured)\n\n"
                            "Configure your API key to get AI-generated responses."
                        )
                        
                        render_sources(documents)
                        
                        # Add to history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": "Sources retrieved (configure API key for responses)",
                            "documents": documents,
                            "confidence": confidence
                        })
                        
                        # Increment Limit Counter (only on success)
                        update_rate_limit(cookie_manager, st.session_state.config)
                        
                    except Exception as e:
                        st.error(f"Search error: {str(e)}")
                        logger.error(f"Retrieval error: {e}")


if __name__ == "__main__":
    main()
