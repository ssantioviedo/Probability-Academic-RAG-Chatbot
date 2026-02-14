import sys
import os
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_config
from src.retrieval.retriever import create_retriever_from_config
from src.retrieval.query_processor import QueryProcessor
from src.retrieval.confidence import ConfidenceCalculator
from src.generation.llm_client import create_client_from_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_retrieval_full_pipeline():
    logger.info("Starting FULL PIPELINE debug...")
    
    # Load config
    config = get_config()
    
    # FORCE INVALID BM25 PATH to simulate the user's issue
    logger.info(f"Original BM25 path: {config.sparse_index_path}")
    config.sparse_index_path = Path("data/processed/non_existent_bm25_index.pkl")
    logger.info(f"Modified config to use invalid BM25 index: {config.sparse_index_path}")

    # Initialize components
    try:
        retriever = create_retriever_from_config(config)
        logger.info("Retriever initialized.")
        
        query_processor = QueryProcessor(api_key=config.google_api_key)
        logger.info("QueryProcessor initialized.")
        
        confidence_calculator = ConfidenceCalculator()
        logger.info("ConfidenceCalculator initialized.")
        
        llm_client = create_client_from_config(config)
        logger.info("LLM Client initialized.")
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return

    # User Query
    query = "cual es el teorema central del límite?"
    logger.info(f"Processing query: '{query}'")

    # 1. Optimize Query
    optimized_query = query_processor.generate_optimized_query(query)
    logger.info(f"Optimized query: '{optimized_query}'")

    # 2. Retrieve Documents
    logger.info("Retrieving documents...")
    # Simulate app.py logic: context expanded
    context, documents = retriever.retrieve_context_expanded(
        query=optimized_query,
        top_k=5,
        expand_pages=1, # app.py default from slider interaction
        max_total_docs=15 
    )
    logger.info(f"Retrieved {len(documents)} documents.")
    
    if not documents:
        logger.error("No documents retrieved!")
        # proceed to check failure mode
    
    # 3. Verify Relevance (Simulating app.py loop)
    logger.info("Verifying relevance (top 5)...")
    valid_documents = []
    for doc in documents[:5]:
        content_preview = getattr(doc, 'context_text', doc.text)[:100].replace('\n', ' ')
        try:
            is_relevant = query_processor.verify_content_relevance(optimized_query, getattr(doc, 'context_text', doc.text))
            status = "RELEVANT" if is_relevant else "IRRELEVANT"
            logger.info(f"Doc {doc.chunk_id} ({content_preview}...): {status}")
            if is_relevant:
                valid_documents.append(doc)
        except Exception as e:
            logger.error(f"Error verifying doc {doc.chunk_id}: {e}")
            valid_documents.append(doc) # Fallback

    if valid_documents:
        # Use filtered list + tail
        final_documents = valid_documents + documents[5:]
        logger.info(f"Kept {len(final_documents)} documents after verification.")
        
        # 4. Calculate Confidence
        similarities = retriever.get_similarities(final_documents)
        confidence_result = confidence_calculator.calculate(similarities)
        logger.info(f"Confidence Level: {confidence_result.level}")
        logger.info(f"Confidence Score: {confidence_result.score}")
        logger.info(f"Confidence Message: {confidence_result.message}")
        
        # 5. Generate Response
        logger.info("Generating response from LLM...")
        
        # Re-construct context from final_documents
        context_parts = []
        current_source = None
        for doc in final_documents:
            source = doc.source_file
            if source != current_source:
                current_source = source
                context_parts.append(f"\n=== From: {doc.author} - {source} ===\n")
            citation = doc.get_citation()
            context_parts.append(f"[{citation}]\n{doc.context_text}\n")
        final_context = "\n".join(context_parts)

        result = llm_client.generate(
            query=query, # Original Spanish query
            context=final_context,
            chat_history=[]
        )
        
        logger.info("=== GENERATION RESULT ===")
        print(result.answer.encode('utf-8', errors='replace').decode('utf-8')) # Handle printing safe
        logger.info("=========================")

    else:
        logger.warning("All top 5 documents were rejected as IRRELEVANT!")
        # Simulate app.py fallback behavior
        logger.info("Simulating fallback to Deep Search (not implemented in this script, just noting failure state).")

if __name__ == "__main__":
    debug_retrieval_full_pipeline()
