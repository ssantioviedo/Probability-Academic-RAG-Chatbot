"""
Retriever module for querying the vector database.

Provides semantic search capabilities over the indexed
bibliography with metadata filtering support.
"""

from pathlib import Path
from typing import Optional
from src.utils.logger import get_logger, LoggerMixin
from src.utils.config import Config
from src.retrieval.sparse import BM25Retriever
from src.retrieval.reranker import Reranker
from src.retrieval.models import RetrievedDocument
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


# Default retrieval configuration
DEFAULT_TOP_K = 5
DEFAULT_COLLECTION_NAME = "probability_bibliography"


class Retriever(LoggerMixin):
    """
    Semantic search retriever for the indexed bibliography.
    
    Provides query functionality with metadata filtering,
    similarity scoring, and result ranking.
    
    Attributes:
        persist_directory: Path to ChromaDB storage.
        collection_name: Name of the ChromaDB collection.
        embedding_model: Sentence-transformers model name.
        default_top_k: Default number of results to return.
    
    Example:
        >>> retriever = Retriever(Path("./chroma_db"))
        >>> results = retriever.retrieve("What is the law of large numbers?")
        >>> for doc in results:
        ...     print(f"{doc.author}: {doc.get_snippet()}")
    """
    
    def __init__(
        self,
        persist_directory: Path,
        embedding_model: str = "paraphrase-multilingual-mpnet-base-v2",
        collection_name: str = DEFAULT_COLLECTION_NAME,
        default_top_k: int = DEFAULT_TOP_K,
        config: Optional[Config] = None
    ) -> None:
        """
        Initialize the retriever.
        
        Args:
            persist_directory: Directory for ChromaDB persistence.
            embedding_model: Sentence-transformers model name.
            collection_name: Name for the ChromaDB collection.
            default_top_k: Default number of results to return.
            config: App config for advanced settings.
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.default_top_k = default_top_k
        self.config = config  # Store config for RRF parameters
        
        self.logger.info(
            f"Initializing Retriever: model={embedding_model}, "
            f"collection={collection_name}"
        )
        
        # Initialize sparse retriever
        self.sparse_retriever = BM25Retriever(config.sparse_index_path)
        self.sparse_retriever.load()
        
        # Initialize reranker
        self.reranker = Reranker(config.reranker_model_name)
        
        # Check if persist directory exists
        if not self.persist_directory.exists():
            raise FileNotFoundError(
                f"ChromaDB directory not found: {self.persist_directory}. "
                "Please run the ingestion pipeline first."
            )
        
        # Initialize embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model
        )
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=False
            )
        )
        
        # Get existing collection
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            self.logger.info(
                f"Retriever connected: {self.collection.count()} documents available"
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not access collection '{collection_name}': {e}. "
                "Please run the ingestion pipeline first."
            ) from e
    
    def _distance_to_similarity(self, distance: float) -> float:
        """
        Convert distance metric to similarity score.
        
        ChromaDB uses L2 distance by default. For cosine similarity,
        we use the formula: similarity = 1 - (distance / 2).
        
        Args:
            distance: Distance from query.
        
        Returns:
            Similarity score (0-1).
        """
        # For cosine distance, convert to similarity
        # Cosine distance is in [0, 2], similarity in [0, 1]
        similarity = 1 - (distance / 2)
        return max(0.0, min(1.0, similarity))
    
    def _build_where_filter(
        self,
        source_type: Optional[str] = None,
        author: Optional[str] = None,
        language: Optional[str] = None,
        filters: Optional[dict] = None
    ) -> Optional[dict]:
        """
        Build a ChromaDB where filter from parameters.
        
        Args:
            source_type: Filter by source type.
            author: Filter by author.
            language: Filter by language.
            filters: Additional custom filters.
        
        Returns:
            Where filter dictionary or None.
        """
        conditions = []
        
        if source_type and source_type.lower() != "all":
            conditions.append({"source_type": source_type})
        
        if author:
            conditions.append({"author": author})
        
        if language:
            conditions.append({"language": language})
        
        if filters:
            for key, value in filters.items():
                if value is not None:
                    conditions.append({key: value})
        
        if not conditions:
            return None
        
        if len(conditions) == 1:
            return conditions[0]
        
        return {"$and": conditions}
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        source_type: Optional[str] = None,
        author: Optional[str] = None,
        language: Optional[str] = None,
        filters: Optional[dict] = None,
        include_distances: bool = True
    ) -> list[RetrievedDocument]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query string.
            top_k: Number of results to return.
            source_type: Filter by source type (book, lecture_notes, exercises).
            author: Filter by author name.
            language: Filter by language (en, es).
            filters: Additional metadata filters.
            include_distances: Whether to include distance scores.
        
        Returns:
            List of RetrievedDocument objects ranked by relevance.
        """
        if not query.strip():
            self.logger.warning("Empty query provided")
            return []
        
        top_k = top_k or self.default_top_k
        
        # Build where filter
        where_filter = self._build_where_filter(
            source_type=source_type,
            author=author,
            language=language,
            filters=filters
        )
        
        self.logger.debug(
            f"Retrieving for query: '{query[:50]}...' "
            f"(top_k={top_k}, filters={where_filter})"
        )
        
        try:
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            # Parse results
            documents: list[RetrievedDocument] = []
            
            if results and results.get("documents") and results["documents"][0]:
                for i, text in enumerate(results["documents"][0]):
                    metadata = {}
                    if results.get("metadatas") and results["metadatas"][0]:
                        metadata = results["metadatas"][0][i] or {}
                    
                    distance = 0.0
                    if results.get("distances") and results["distances"][0]:
                        distance = results["distances"][0][i]
                    
                    similarity = self._distance_to_similarity(distance)
                    
                    chunk_id = ""
                    if results.get("ids") and results["ids"][0]:
                        chunk_id = results["ids"][0][i]
                    
                    # Extract parent content if available
                    parent_content = metadata.get("parent_content")
                    
                    doc = RetrievedDocument(
                        text=text,
                        metadata=metadata,
                        similarity=similarity,
                        distance=distance,
                        chunk_id=chunk_id,
                        parent_content=parent_content
                    )
                    documents.append(doc)
            
            self.logger.info(
                f"Retrieved {len(documents)} documents for query"
            )
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Retrieval error: {e}")
            raise RuntimeError(f"Failed to retrieve documents: {e}") from e
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: list[RetrievedDocument],
        sparse_results: list[RetrievedDocument],
        top_k: int,
        k_param: int = 60
    ) -> list[RetrievedDocument]:
        """
        Merge dense and sparse results using Reciprocal Rank Fusion.
        
        RRF formula: score(doc) = sum(1 / (k + rank_i)) for all rankings
        
        Args:
            dense_results: Results from dense (vector) retrieval.
            sparse_results: Results from sparse (BM25) retrieval.
            top_k: Number of results to return.
            k_param: RRF constant (default 60).
            
        Returns:
            Merged and ranked list of documents.
        """
        scores = {}
        doc_map = {}  # chunk_id -> RetrievedDocument
        
        # Score dense results by rank
        for rank, doc in enumerate(dense_results):
            chunk_id = doc.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k_param + rank + 1)
            if chunk_id not in doc_map:
                doc_map[chunk_id] = doc
        
        # Score sparse results by rank
        for rank, doc in enumerate(sparse_results):
            chunk_id = doc.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k_param + rank + 1)
            if chunk_id not in doc_map:
                doc_map[chunk_id] = doc
        
        # Sort by RRF score (descending)
        ranked_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top-k documents with updated similarity scores
        results = []
        for chunk_id, rrf_score in ranked_ids[:top_k]:
            doc = doc_map[chunk_id]
            # Update similarity to reflect RRF score (normalize to 0-1)
            doc.similarity = min(1.0, rrf_score)
            results.append(doc)
        
        return results
    
    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,  # Not used in RRF but kept for API compatibility
        rerank: bool = True,
        expand_query: bool = True  # NEW: Enable query expansion
    ) -> list[RetrievedDocument]:
        """
        Perform hybrid retrieval (Dense + Sparse) with Reciprocal Rank Fusion.
        
        Args:
            query: User query.
            top_k: Number of final results.
            alpha: Weighting factor (unused in RRF, kept for compatibility).
            rerank: Whether to apply cross-encoder reranking.
            expand_query: Whether to expand query with mathematical terms.
            
        Returns:
            List of top_k RetrievedDocument objects.
        """
        # Expand query for sparse retrieval if enabled
        sparse_query = query
        use_expansion = expand_query if expand_query is not None else self.config.enable_query_expansion
        if use_expansion:
            try:
                from src.retrieval.query_expansion import expand_query_for_hybrid
                sparse_query = expand_query_for_hybrid(query)
                self.logger.debug(f"Expanded query: '{query}' -> '{sparse_query}'")
            except Exception as e:
                self.logger.warning(f"Query expansion failed: {e}, using original query")
                sparse_query = query
        
        # 1. Dense Retrieval (Vector) - use original query for semantic matching
        dense_multiplier = self.config.dense_fetch_multiplier
        dense_results = self.retrieve(query, top_k=top_k * dense_multiplier)
        
        # 2. Sparse Retrieval (BM25) - use expanded query for keyword matching
        sparse_multiplier = self.config.sparse_fetch_multiplier
        sparse_chunks = self.sparse_retriever.search(sparse_query, top_k=top_k * sparse_multiplier)
        sparse_results = []
        for chunk in sparse_chunks:
            doc = RetrievedDocument(
                text=chunk.text,
                metadata=chunk.metadata,
                similarity=0.5,  # Placeholder, will be updated by RRF
                distance=0.0,
                chunk_id=chunk.chunk_id,
                parent_content=chunk.metadata.get("parent_content")
            )
            sparse_results.append(doc)
        
        # 3. Reciprocal Rank Fusion with configurable k_param
        fused_results = self._reciprocal_rank_fusion(
            dense_results, 
            sparse_results, 
            top_k=top_k * 2 if rerank else top_k,
            k_param=self.config.rrf_k_param  # Use config value
        )
        
        # 4. Optional Reranking
        if rerank and fused_results:
            reranked_docs = self.reranker.rerank(query, fused_results, top_k=top_k)
            return reranked_docs
        
        return fused_results[:top_k]
    
    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None
    ) -> tuple[str, list[RetrievedDocument]]:
        """
        Retrieve documents and format as context string.
        
        Uses hybrid retrieval (dense + sparse + rerank) when no filters
        are specified, falling back to dense-only when filters are present.
        
        Args:
            query: Search query string.
            top_k: Number of results to return.
            filters: Metadata filters.
        
        Returns:
            Tuple of (context_string, list_of_documents).
        """
        # Use hybrid search when no metadata filters are active
        if not filters:
            documents = self.retrieve_hybrid(query=query, top_k=top_k)
        else:
            documents = self.retrieve(
                query=query,
                top_k=top_k,
                filters=filters
            )
        
        # Format context
        context_parts = []
        for i, doc in enumerate(documents, 1):
            citation = doc.get_citation()
            context_parts.append(
                f"[Source {i} - {citation}]\n{doc.context_text}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        
        return context, documents
    
    def get_similarities(
        self,
        documents: list[RetrievedDocument]
    ) -> list[float]:
        """
        Extract similarity scores from documents.
        
        Args:
            documents: List of retrieved documents.
        
        Returns:
            List of similarity scores.
        """
        return [doc.similarity for doc in documents]
    
    def get_collection_stats(self) -> dict:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics.
        """
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model,
            }
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}
    
    def get_unique_values(self, field: str) -> list[str]:
        """
        Get unique values for a metadata field.
        
        Args:
            field: Metadata field name.
        
        Returns:
            List of unique values.
        """
        try:
            results = self.collection.get(include=["metadatas"])
            
            values = set()
            if results and results.get("metadatas"):
                for metadata in results["metadatas"]:
                    if metadata and field in metadata:
                        value = metadata[field]
                        if value:
                            values.add(str(value))
            
            return sorted(list(values))
            
        except Exception as e:
            self.logger.error(f"Error getting unique values: {e}")
            return []
    
    def retrieve_with_context_expansion(
        self,
        query: str,
        top_k: int = 5,
        expand_pages: int = 1,
        max_total_docs: int = 25,
        source_type: Optional[str] = None,
        author: Optional[str] = None,
        filters: Optional[dict] = None,
        strategy: str = "hybrid" # default to hybrid now
    ) -> list[RetrievedDocument]:
        """
        Retrieve documents with context expansion.
        
        First retrieves top_k most similar chunks, then expands
        by fetching adjacent chunks from the same source files
        to provide more complete context.
        
        Args:
            query: Search query string.
            top_k: Number of initial results to retrieve.
            expand_pages: Number of pages before/after to include.
            max_total_docs: Maximum total documents to return (to avoid token limits).
            source_type: Filter by source type.
            author: Filter by author.
            filters: Additional metadata filters.
        
        Returns:
            List of RetrievedDocument objects with expanded context.
        """
        
        # Step 1: Get Initial Results
        if strategy == "hybrid":
             # Note: Filters (author/source) are hard to apply to BM25 without pre-filtering the index.
             # For now, Hybrid ignores metadata filters for the sparse part or applies them post-hoc.
             # Given the implementation, we'll stick to Dense if filters are present, or apply filters after.
             # Simplest: If filters are present, use Dense only to ensure compliance.
             if source_type or author or filters:
                 initial_docs = self.retrieve(
                    query=query,
                    top_k=top_k,
                    source_type=source_type,
                    author=author,
                    filters=filters
                )
             else:
                 initial_docs = self.retrieve_hybrid(query, top_k=top_k)
        else:
             initial_docs = self.retrieve(
                query=query,
                top_k=top_k,
                source_type=source_type,
                author=author,
                filters=filters
            )
        
        if not initial_docs:
            return []
        
        # Collect source files and pages to expand
        seen_chunks = {doc.chunk_id for doc in initial_docs}
        expanded_docs = list(initial_docs)
        
        # Limit expansion to top sources only (avoid expanding too many)
        top_sources = {}
        for doc in initial_docs[:5]:  # Only expand from top 5 matches
            source_file = doc.source_file
            page = doc.page
            if source_file and page:
                if source_file not in top_sources:
                    top_sources[source_file] = set()
                # Add pages to expand (before and after)
                for p in range(max(1, page - expand_pages), page + expand_pages + 1):
                    top_sources[source_file].add(p)
        
        # Fetch adjacent chunks
        for source_file, pages in top_sources.items():
            if len(expanded_docs) >= max_total_docs:
                break
                
            for page in sorted(pages):
                if len(expanded_docs) >= max_total_docs:
                    break
                    
                try:
                    # Query for chunks from specific file and page
                    results = self.collection.get(
                        where={
                            "$and": [
                                {"source_file": source_file},
                                {"page": page}
                            ]
                        },
                        include=["documents", "metadatas"]
                    )
                    
                    if results and results.get("ids"):
                        for i, chunk_id in enumerate(results["ids"]):
                            if len(expanded_docs) >= max_total_docs:
                                break
                            if chunk_id not in seen_chunks:
                                seen_chunks.add(chunk_id)
                                text = results["documents"][i] if results.get("documents") else ""
                                metadata = results["metadatas"][i] if results.get("metadatas") else {}
                                
                                # Calculate expansion similarity as slightly below
                                # the lowest initial match to rank them correctly
                                expansion_similarity = max(0.3, min(
                                    doc.similarity for doc in initial_docs
                                ) - 0.05)
                                
                                doc = RetrievedDocument(
                                    text=text,
                                    metadata=metadata,
                                    similarity=expansion_similarity,
                                    distance=0.0,
                                    chunk_id=chunk_id,
                                    parent_content=metadata.get("parent_content")
                                )
                                expanded_docs.append(doc)
                                
                except Exception as e:
                    self.logger.debug(f"Error expanding context for {source_file} p{page}: {e}")
        
        # Sort by source file and page for coherent reading
        def sort_key(doc):
            return (doc.source_file, doc.page, -doc.similarity)
        
        expanded_docs.sort(key=sort_key)
        
        self.logger.info(
            f"Context expansion: {len(initial_docs)} -> {len(expanded_docs)} documents"
        )
        
        return expanded_docs
    
    def retrieve_context_expanded(
        self,
        query: str,
        top_k: int = 5,
        expand_pages: int = 2,
        filters: Optional[dict] = None
    ) -> tuple[str, list[RetrievedDocument]]:
        """
        Retrieve documents with expansion and format as context string.
        
        Args:
            query: Search query string.
            top_k: Number of initial results.
            expand_pages: Pages to expand around matches.
            filters: Metadata filters.
        
        Returns:
            Tuple of (context_string, list_of_documents).
        """
        documents = self.retrieve_with_context_expansion(
            query=query,
            top_k=top_k,
            expand_pages=expand_pages,
            filters=filters
        )
        
        # Format context - group by source for better readability
        context_parts = []
        current_source = None
        
        # Keep documents sorted by Source/Page for the LLM Context
        for doc in documents:
            source = doc.source_file
            if source != current_source:
                current_source = source
                context_parts.append(f"\n=== From: {doc.author} - {source} ===\n")
            
            citation = doc.get_citation()
            context_parts.append(f"[{citation}]\n{doc.context_text}\n")
        
        context = "\n".join(context_parts)
        
        # Sort documents by similarity (descending) for the UI/User
        # The user wants to see best matches first, even if context expansion added lower checks
        documents.sort(key=lambda d: d.similarity, reverse=True)
        
        return context, documents




def create_retriever_from_config(config) -> Retriever:
    """
    Create a Retriever from application config.
    
    Args:
        config: Config object with retrieval settings.
    
    Returns:
        Configured Retriever instance.
    """
    return Retriever(
        persist_directory=Path(config.chroma_db_path),
        embedding_model=config.embedding_model,
        collection_name=DEFAULT_COLLECTION_NAME,
        default_top_k=config.default_top_k,
        config=config # Pass config to initialize sparse/reranker
    )
