"""
Text Chunker module for intelligent text splitting.

Uses LangChain's RecursiveCharacterTextSplitter to create
semantically meaningful chunks while preserving context.
"""

import re
import hashlib
from typing import Optional
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.logger import get_logger, LoggerMixin
from src.ingestion.pdf_extractor import ExtractedPage, PDFDocument


# Default chunking configuration
DEFAULT_CHUNK_SIZE = 2500  # characters (optimized for theorem+proof pairs)
DEFAULT_CHUNK_OVERLAP = 500  # characters overlap (20%)
CHARS_PER_TOKEN = 4  # Rough estimate for English text

# Separators in priority order
SEPARATORS = [
    "\n\n",      # Paragraph breaks
    "\n",        # Line breaks
    ". ",        # Sentence endings
    "? ",        # Question endings
    "! ",        # Exclamation endings
    "; ",        # Semicolon breaks
    ", ",        # Comma breaks
    " ",         # Word breaks
    ""           # Character level (last resort)
]

# Pattern to detect theorem/definition boundaries (NOT proofs - they belong to theorems)
THEOREM_PATTERN = re.compile(
    r'^(Theorem|Teorema|Lemma|Lema|Proposition|Proposición|'
    r'Definition|Definición|Corollary|Corolario|'
    r'Example|Ejemplo|Exercise|Ejercicio)\s*[\d.]*',
    re.IGNORECASE | re.MULTILINE
)

# Pattern to detect proof end markers
PROOF_END_PATTERN = re.compile(
    r'(□|∎|Q\.E\.D\.|QED|Fin de la demostración|Fin de demostración)',
    re.MULTILINE | re.IGNORECASE
)


@dataclass
class TextChunk:
    """
    Represents a chunk of text with associated metadata.
    
    Attributes:
        text: The chunk text content.
        chunk_id: Unique identifier for this chunk.
        metadata: Dictionary of metadata from source.
        start_char: Starting character position in original text.
        end_char: Ending character position in original text.
    """
    text: str
    chunk_id: str
    metadata: dict = field(default_factory=dict)
    start_char: int = 0
    end_char: int = 0
    
    @property
    def word_count(self) -> int:
        """Get approximate word count."""
        return len(self.text.split())
    
    @property
    def char_count(self) -> int:
        """Get character count."""
        return len(self.text)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "metadata": self.metadata,
            "word_count": self.word_count,
            "char_count": self.char_count,
        }


class TextChunker(LoggerMixin):
    """
    Intelligent text chunker using LangChain's text splitter.
    
    Creates semantically meaningful chunks while preserving
    mathematical content, theorem boundaries, and context.
    
    Attributes:
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between chunks in characters.
        splitter: LangChain text splitter instance.
    
    Example:
        >>> chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        >>> chunks = chunker.chunk_document(pdf_document)
        >>> print(f"Created {len(chunks)} chunks")
    """
    
    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        use_token_count: bool = True,
        parent_chunk_size: Optional[int] = None,
        child_chunk_size: Optional[int] = None,
    ) -> None:
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Target chunk size (in tokens if use_token_count, else chars).
            chunk_overlap: Overlap between chunks.
            use_token_count: If True, interpret sizes as tokens and convert to chars.
            parent_chunk_size: Size of parent chunks (context).
            child_chunk_size: Size of child chunks (indexing).
        """
        # Convert token counts to character counts
        if use_token_count:
            self.chunk_size = chunk_size * CHARS_PER_TOKEN
            self.chunk_overlap = chunk_overlap * CHARS_PER_TOKEN
            
            if parent_chunk_size:
                self.parent_chunk_size = parent_chunk_size * CHARS_PER_TOKEN
            else:
                self.parent_chunk_size = self.chunk_size * 2  # Default fallback
                
            if child_chunk_size:
                self.child_chunk_size = child_chunk_size * CHARS_PER_TOKEN
            else:
                self.child_chunk_size = self.chunk_size
        else:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.parent_chunk_size = parent_chunk_size or (chunk_size * 2)
            self.child_chunk_size = child_chunk_size or chunk_size
        
        # Recursive splitter for fallbacks (when an atomic block is too big)
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=SEPARATORS,
            length_function=len,
            is_separator_regex=False,
        )
        
        self.logger.info(
            f"TextChunker initialized: chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}, "
            f"Atomic Block Strategy enabled"
        )
    
    def _generate_chunk_id(
        self,
        source_file: str,
        page: int,
        chunk_index: int,
        text: str
    ) -> str:
        """
        Generate a unique chunk ID.
        
        Args:
            source_file: Source filename.
            page: Page number.
            chunk_index: Index of chunk within page.
            text: Chunk text (used for hash).
        
        Returns:
            Unique chunk ID string.
        """
        # Create base ID
        base_name = source_file.replace(".pdf", "").replace(" ", "_").replace("__", "_")[:30]
        base_id = f"{base_name}_p{page}_seq{chunk_index}"
        
        # Add short hash for uniqueness
        text_hash = hashlib.md5(text.encode()).hexdigest()[:6]
        
        return f"{base_id}_{text_hash}"
    
    def _preserve_latex_blocks(self, text: str) -> str:
        """
        Ensure LaTeX blocks are not split inappropriately.
        
        Adds newlines around [LATEX] tags to encourage splits there.
        
        Args:
            text: Text with [LATEX] tags.
        
        Returns:
            Text with improved split points.
        """
        # Add paragraph breaks before [LATEX] tags to encourage splits there
        text = re.sub(r'(?<!\n)\[LATEX\]', '\n\n[LATEX]', text)
        text = re.sub(r'\[/LATEX\](?!\n)', '[/LATEX]\n\n', text)
        return text
    
    def _detect_block_type(self, text: str) -> str:
        """
        Classify the type of mathematical content in a chunk.
        
        Args:
            text: Chunk text to classify.
            
        Returns:
            Block type: 'theorem', 'proof', 'definition', 'example', 'lemma', or 'general'.
        """
        text_start = text[:300].lower()
        
        # Check for specific mathematical structures
        if any(kw in text_start for kw in ['theorem', 'teorema']):
            return 'theorem'
        elif any(kw in text_start for kw in ['proof', 'demostración', 'dem.', 'prueba']):
            return 'proof'
        elif any(kw in text_start for kw in ['definition', 'definición']):
            return 'definition'
        elif any(kw in text_start for kw in ['example', 'ejemplo']):
            return 'example'
        elif any(kw in text_start for kw in ['lemma', 'lema']):
            return 'lemma'
        elif any(kw in text_start for kw in ['proposition', 'proposición']):
            return 'proposition'
        elif any(kw in text_start for kw in ['corollary', 'corolario']):
            return 'corollary'
        elif any(kw in text_start for kw in ['exercise', 'ejercicio']):
            return 'exercise'
        
        return 'general'


    def _split_into_atomic_blocks(self, text: str) -> list[str]:
        """
        Split text into atomic blocks (Theorems with proofs, paragraphs, etc.).
        
        Strategy:
        1. Split by double newlines to get paragraphs
        2. Detect theorem/definition starts
        3. Keep accumulating paragraphs until:
           - We find a proof end marker (□, QED, etc.), OR
           - We find the next theorem/definition start, OR
           - The block becomes too large
        
        Args:
            text: Input text.
            
        Returns:
            List of text blocks.
        """
        # Split by double newlines to get raw paragraphs
        raw_paragraphs = re.split(r'\n\s*\n', text)
        clean_paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]
        
        if not clean_paragraphs:
            return []
        
        blocks = []
        current_block = ""
        in_theorem_block = False
        
        for i, para in enumerate(clean_paragraphs):
            # Check if this paragraph starts a new theorem/definition
            is_theorem_start = bool(THEOREM_PATTERN.match(para))
            
            # Check if this paragraph contains a proof end marker
            has_proof_end = bool(PROOF_END_PATTERN.search(para))
            
            if is_theorem_start:
                # Save previous block if exists
                if current_block:
                    blocks.append(current_block)
                    current_block = ""
                
                # Start new theorem block
                current_block = para
                in_theorem_block = True
                
            elif in_theorem_block:
                # We're inside a theorem block, keep accumulating
                current_block += "\n\n" + para
                
                # Check if this paragraph ends the proof
                if has_proof_end:
                    # Proof ended, save the block
                    blocks.append(current_block)
                    current_block = ""
                    in_theorem_block = False
                    
            else:
                # Regular paragraph (not part of theorem)
                if current_block:
                    # Check if adding this would make block too large
                    if len(current_block) + len(para) + 2 > self.chunk_size * 1.5:
                        blocks.append(current_block)
                        current_block = para
                    else:
                        current_block += "\n\n" + para
                else:
                    current_block = para
        
        # Don't forget the last block
        if current_block:
            blocks.append(current_block)
        
        return blocks

    def chunk_page(
        self,
        page: ExtractedPage,
        chunk_start_index: int = 0
    ) -> list[TextChunk]:
        """
        Chunk a single extracted page using atomic blocks strategy.
        
        Args:
            page: ExtractedPage to chunk.
            chunk_start_index: Starting index for chunk numbering.
        
        Returns:
            List of TextChunk objects.
        """
        # Prepare text
        text = page.text.strip()
        if not text:
            return []
            
        if page.has_latex:
            text = self._preserve_latex_blocks(text)
            
        chunks: list[TextChunk] = []
        
        # 1. Get Atomic Blocks (Paragraphs with proper headers attached)
        atomic_blocks = self._split_into_atomic_blocks(text)
        
        current_chunk_text = ""
        current_chunk_blocks = []
        
        def commit_chunk(text_content: str):
            """Helper to create and append a chunk"""
            if not text_content.strip():
                return
                
            nonlocal chunk_start_index
            idx = chunk_start_index + len(chunks)
            chunk_id = self._generate_chunk_id(page.source_file, page.page_number, idx, text_content)
            
            # Calculate metadata
            metadata = page.to_metadata()
            metadata["chunk_id"] = chunk_id
            metadata["chunk_index"] = idx
            
            # Add semantic metadata
            metadata["block_type"] = self._detect_block_type(text_content)
            metadata["has_theorem"] = bool(re.search(r'\b(theorem|teorema)\b', text_content, re.I))
            metadata["has_proof"] = bool(re.search(r'\b(proof|demostración|dem\.|prueba|□|∎)\b', text_content, re.I))
            metadata["has_definition"] = bool(re.search(r'\b(definition|definición)\b', text_content, re.I))
            
            # Calculate positions (approximate)
            start_char = page.text.find(text_content[:50]) # heuristic
            if start_char == -1: start_char = 0
            end_char = start_char + len(text_content)
            
            chunks.append(TextChunk(
                text=text_content.strip(),
                chunk_id=chunk_id,
                metadata=metadata,
                start_char=start_char,
                end_char=end_char
            ))

        for block in atomic_blocks:
            # Check if adding this block exceeds chunk size
            new_len = len(current_chunk_text) + len(block) + 2 # +2 for \n\n
            
            if new_len > self.chunk_size:
                # If current chunk has content, commit it first
                if current_chunk_text:
                    commit_chunk(current_chunk_text)
                    current_chunk_text = ""
                    current_chunk_blocks = []
                
                # Now handle the current block
                if len(block) > self.chunk_size:
                    # Block is HUGE (bigger than chunk size alone). Split it recursively.
                    sub_chunks = self.recursive_splitter.split_text(block)
                    for sub in sub_chunks:
                        commit_chunk(sub)
                else:
                    # Block fits in a new chunk
                    current_chunk_text = block
            else:
                # Add to current chunk
                if current_chunk_text:
                    current_chunk_text += "\n\n" + block
                else:
                    current_chunk_text = block
                    
        # Commit remaining
        if current_chunk_text:
            commit_chunk(current_chunk_text)
        
        return chunks

    def create_parent_chunks(self, child_chunks: list[TextChunk]) -> list[TextChunk]:
        """
        Create parent chunks from child chunks for hierarchical retrieval.
        
        Parent chunks are larger (parent_chunk_size) and contain multiple child chunks.
        Each child chunk stores a reference to its parent for context expansion.
        
        Args:
            child_chunks: List of child chunks to group into parents.
            
        Returns:
            List of parent chunks.
        """
        if not child_chunks:
            return []
        
        parents = []
        parent_id_counter = 0
        
        current_parent_text = ""
        current_children_ids = []
        first_child_metadata = None
        
        for child in child_chunks:
            # Check if adding this child exceeds parent size
            potential_length = len(current_parent_text) + len(child.text) + 2  # +2 for \n\n
            
            if potential_length > self.parent_chunk_size and current_parent_text:
                # Save current parent
                parent_id = f"parent_{first_child_metadata.get('source_file', 'unknown')}_{parent_id_counter}"
                parent_chunk = TextChunk(
                    text=current_parent_text,
                    chunk_id=parent_id,
                    metadata={
                        **first_child_metadata,
                        "is_parent": True,
                        "child_count": len(current_children_ids),
                        "child_ids": ",".join(current_children_ids)
                    }
                )
                parents.append(parent_chunk)
                
                # Link children to parent (update metadata in original child_chunks)
                for child_id in current_children_ids:
                    for c in child_chunks:
                        if c.chunk_id == child_id:
                            c.metadata["parent_id"] = parent_id
                            c.metadata["parent_content"] = current_parent_text
                            break
                
                parent_id_counter += 1
                current_parent_text = ""
                current_children_ids = []
                first_child_metadata = None
            
            # Add child to current parent
            if not first_child_metadata:
                first_child_metadata = child.metadata.copy()
            
            if current_parent_text:
                current_parent_text += "\n\n" + child.text
            else:
                current_parent_text = child.text
            current_children_ids.append(child.chunk_id)
        
        # Don't forget last parent
        if current_parent_text and first_child_metadata:
            parent_id = f"parent_{first_child_metadata.get('source_file', 'unknown')}_{parent_id_counter}"
            parent_chunk = TextChunk(
                text=current_parent_text,
                chunk_id=parent_id,
                metadata={
                    **first_child_metadata,
                    "is_parent": True,
                    "child_count": len(current_children_ids),
                    "child_ids": ",".join(current_children_ids)
                }
            )
            parents.append(parent_chunk)
            
            # Link children to parent
            for child_id in current_children_ids:
                for c in child_chunks:
                    if c.chunk_id == child_id:
                        c.metadata["parent_id"] = parent_id
                        c.metadata["parent_content"] = current_parent_text
                        break
        
        self.logger.info(
            f"Created {len(parents)} parent chunks from {len(child_chunks)} child chunks"
        )
        
        return parents
    

    def chunk_document(self, document: PDFDocument, create_parents: bool = True) -> list[TextChunk]:
        """
        Chunk an entire PDF document.
        
        Args:
            document: PDFDocument to chunk.
            create_parents: Whether to create parent chunks for hierarchical retrieval.
        
        Returns:
            List of all TextChunk objects from the document (children + parents if enabled).
        """
        all_chunks: list[TextChunk] = []
        
        self.logger.debug(f"Chunking document: {document.filename}")
        
        for page in document.pages:
            page_chunks = self.chunk_page(page, chunk_start_index=len(all_chunks))
            all_chunks.extend(page_chunks)
        
        # Create parent chunks if enabled
        if create_parents and self.parent_chunk_size > self.chunk_size:
            parent_chunks = self.create_parent_chunks(all_chunks)
            # Return both children and parents
            all_chunks.extend(parent_chunks)
            self.logger.info(
                f"Created {len(all_chunks)} total chunks ({len(all_chunks) - len(parent_chunks)} children + "
                f"{len(parent_chunks)} parents) from {document.filename} ({document.total_pages} pages)"
            )
        else:
            self.logger.info(
                f"Created {len(all_chunks)} chunks from {document.filename} "
                f"({document.total_pages} pages)"
            )
        
        return all_chunks
    
    def chunk_documents(
        self,
        documents: list[PDFDocument]
    ) -> tuple[list[TextChunk], dict]:
        """
        Chunk multiple documents and return statistics.
        
        Args:
            documents: List of PDFDocuments to chunk.
        
        Returns:
            Tuple of (all_chunks, statistics_dict).
        """
        all_chunks: list[TextChunk] = []
        stats = {
            "total_documents": len(documents),
            "total_pages": 0,
            "total_chunks": 0,
            "avg_chunk_size": 0,
            "by_source_type": {},
        }
        
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
            
            stats["total_pages"] += doc.total_pages
            
            # Track by source type
            if doc.pages:
                source_type = doc.pages[0].source_type
                if source_type not in stats["by_source_type"]:
                    stats["by_source_type"][source_type] = {
                        "documents": 0,
                        "chunks": 0
                    }
                stats["by_source_type"][source_type]["documents"] += 1
                stats["by_source_type"][source_type]["chunks"] += len(chunks)
        
        stats["total_chunks"] = len(all_chunks)
        
        if all_chunks:
            total_chars = sum(chunk.char_count for chunk in all_chunks)
            stats["avg_chunk_size"] = total_chars / len(all_chunks)
        
        self.logger.info(
            f"Chunking complete: {stats['total_chunks']} chunks from "
            f"{stats['total_documents']} documents"
        )
        
        return all_chunks, stats


def create_chunker_from_config(config) -> TextChunker:
    """
    Create a TextChunker from application config.
    
    Args:
        config: Config object with chunk_size and chunk_overlap.
    
    Returns:
        Configured TextChunker instance.
    """
    return TextChunker(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        parent_chunk_size=config.parent_chunk_size,
        child_chunk_size=config.child_chunk_size,
        use_token_count=True
    )
