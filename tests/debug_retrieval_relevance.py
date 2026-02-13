import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.config import get_config
from src.retrieval.retriever import create_retriever_from_config
from src.retrieval.query_processor import QueryProcessor

# Load env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY not found in .env")
    sys.exit(1)

def debug_retrieval():
    print("Initializing components...")
    config = get_config()
    retriever = create_retriever_from_config(config)
    processor = QueryProcessor(api_key=api_key)
    
    query = "Central Limit Theorem definition"
    print(f"\nSearching for: '{query}'")
    
    # Get top result
    # We simulate what app.py does: retrieve_context_expanded -> documents[0]
    # But for debugging transparency, we just get the docs.
    context, documents = retriever.retrieve_context_expanded(query, top_k=5, expand_pages=1)
    
    if not documents:
        print("No documents found!")
        return

    top_doc = documents[0]
    text_to_check = getattr(top_doc, 'context_text', top_doc.text)
    
    print("\n" + "="*50)
    print("TOP DOCUMENT CONTENT TO VERIFY:")
    print("="*50)
    print(text_to_check[:1000] + "..." if len(text_to_check) > 1000 else text_to_check)
    print("="*50)
    
    print("\nRunning Verification...")
    is_relevant = processor.verify_content_relevance(query, text_to_check)
    
    print(f"\nVERDICT: {'✅ RELEVANT' if is_relevant else '❌ IRRELEVANT (False Positive?)'}")

if __name__ == "__main__":
    debug_retrieval()
