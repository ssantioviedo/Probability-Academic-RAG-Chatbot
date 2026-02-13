"""
PDF Extractor module for extracting text content from PDF files.

Uses PyMuPDF (fitz) to extract text while preserving structure
and handling mathematical content appropriately.
"""

import re
from pathlib import Path
from typing import Optional, Generator
from dataclasses import dataclass, field

import fitz  # PyMuPDF

from src.utils.logger import get_logger, LoggerMixin


# Source type mapping based on directory
SOURCE_TYPE_MAP = {
    "books": "book",
    "lecture_notes": "lecture_notes",
    "exercises": "exercises",
}

# Regex patterns for content detection
LATEX_PATTERN = re.compile(r'\$[^$]+\$|\\\[.*?\\\]|\\\(.*?\\\)', re.DOTALL)
CHAPTER_PATTERN = re.compile(
    r'^(?:Chapter|Capítulo|CHAPTER|CAPÍTULO)\s*(\d+)',
    re.IGNORECASE | re.MULTILINE
)



from collections import Counter

@dataclass
class LayoutStats:
    """
    Statistics about the layout of a PDF document.
    
    Attributes:
        header_height: Height of the header region (from top).
        footer_height: Height of the footer region (from bottom).
        recurring_texts: List of recurring text strings found in margins.
    """
    header_height: float = 0.0
    footer_height: float = 0.0
    recurring_texts: set[str] = field(default_factory=set)


class StatisticalLayoutAnalyzer:
    """
    Analyzes PDF layout to statistically identify headers and footers.
    
    Samples pages to find recurring text blocks at the top and bottom
    of pages to dynamically determine exclusion zones.
    """
    
    def __init__(
        self, 
        sample_size: int = 10,
        min_frequency_ratio: float = 0.4, # Lowered to catch alternating headers (left/right pages)
        margin_percent: float = 0.15
    ):
        """
        Initialize the analyzer.
        
        Args:
            sample_size: Number of pages to sample.
            min_frequency_ratio: Ratio of pages a block must appear in to be considered rigid.
            margin_percent: Percentage of page height to scan for headers/footers.
        """
        self.sample_size = sample_size
        self.min_frequency_ratio = min_frequency_ratio
        self.margin_percent = margin_percent
    
    def analyze(self, doc: fitz.Document) -> LayoutStats:
        """
        Analyze the document to find layout statistics.
        
        Args:
            doc: PyMuPDF Document object.
            
        Returns:
            LayoutStats object with detected zones.
        """
        if len(doc) == 0:
            return LayoutStats()
            
        # Select sample pages from the middle to avoid title/preface quirks
        start_idx = max(0, len(doc) // 4)
        end_idx = min(len(doc), start_idx + len(doc) // 2)
        step = max(1, (end_idx - start_idx) // self.sample_size)
        sample_indices = list(range(start_idx, end_idx, step))[:self.sample_size]
        
        if not sample_indices:
            # Fallback for small docs
            sample_indices = list(range(len(doc)))[:self.sample_size]
            
        header_candidates = []
        footer_candidates = []
        
        page_height = 0.0
        
        for idx in sample_indices:
            page = doc[idx]
            page_height = page.rect.height
            margin_px = page_height * self.margin_percent
            
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:
                    continue
                    
                bbox = block["bbox"] # (x0, y0, x1, y1)
                y0, y1 = bbox[1], bbox[3]
                text = " ".join([span["text"] for line in block["lines"] for span in line["spans"]]).strip()
                
                if not text:
                    continue
                # Note: We include digits here to detect recurring page numbers if they are static (rare)
                # But mostly we want to detect text headers.
                
                # Check for Header Candidate (Top margin)
                if y1 < margin_px:
                    header_candidates.append((round(y1, 1), text))
                    
                # Check for Footer Candidate (Bottom margin)
                if y0 > (page_height - margin_px):
                    footer_candidates.append((round(y0, 1), text))

        # Analyze Headers
        header_height = 50.0 # Default safe minimal margin
        recurring = set()
        
        if header_candidates:
            # Group by approximate Y position
            y_counts = Counter(y for y, t in header_candidates)
            # If a Y position is common, it's likely the header separator
            common_ys = [y for y, count in y_counts.items() if count >= len(sample_indices) * 0.4]
            if common_ys:
                header_height = max(common_ys) + 5 # Add buffer
            
            # Find recurring text
            text_counts = Counter(t for y, t in header_candidates)
            for text, count in text_counts.items():
                if count >= len(sample_indices) * self.min_frequency_ratio:
                    recurring.add(text)

        # Analyze Footers
        footer_height = 50.0 # Default safe minimal margin
        
        if footer_candidates:
             # Group by approximate Y position (from bottom this time essentially)
            y_counts = Counter(y for y, t in footer_candidates)
            # For footers, the "start" (y0) is what matters. We exclude anything below the "highest" common footer start.
            common_ys = [y for y, count in y_counts.items() if count >= len(sample_indices) * 0.4]
            if common_ys:
                # The footer zone starts at the lowest Y (highest on page) of the common footers
                footer_start_y = min(common_ys)
                footer_height = page_height - footer_start_y + 5
            
            # Find recurring text
            text_counts = Counter(t for y, t in footer_candidates)
            for text, count in text_counts.items():
                if count >= len(sample_indices) * self.min_frequency_ratio:
                    recurring.add(text)
                    
        return LayoutStats(
            header_height=header_height,
            footer_height=footer_height,
            recurring_texts=recurring
        )




@dataclass
class ExtractedPage:
    """
    Represents extracted content from a single PDF page.
    
    Attributes:
        text: Extracted text content.
        page_number: 1-indexed page number.
        source_file: Name of the source PDF file.
        source_path: Full path to the source PDF.
        source_type: Type of source (book, lecture_notes, exercises).
        author: Extracted or inferred author name.
        chapter: Detected chapter number (if any).
        language: Detected language code (en, es, mixed).
        has_latex: Whether the page contains LaTeX-style math.
    """
    text: str
    page_number: int
    source_file: str
    source_path: str
    source_type: str
    author: str = ""
    chapter: Optional[int] = None
    language: str = "en"
    has_latex: bool = False
    
    def to_metadata(self) -> dict:
        """Convert to metadata dictionary for ChromaDB."""
        return {
            "source_file": self.source_file,
            "source_path": self.source_path,
            "source_type": self.source_type,
            "author": self.author,
            "page": self.page_number,
            "chapter": self.chapter if self.chapter else -1,
            "language": self.language,
            "has_latex": self.has_latex,
        }


@dataclass
class PDFDocument:
    """
    Represents a processed PDF document with all its pages.
    
    Attributes:
        file_path: Path to the PDF file.
        pages: List of extracted pages.
        total_pages: Total number of pages in the document.
        metadata: Document-level metadata.
    """
    file_path: Path
    pages: list[ExtractedPage] = field(default_factory=list)
    total_pages: int = 0
    metadata: dict = field(default_factory=dict)
    
    @property
    def filename(self) -> str:
        """Get the filename without extension."""
        return self.file_path.stem
    
    @property
    def full_text(self) -> str:
        """Get concatenated text from all pages."""
        return "\n\n".join(page.text for page in self.pages)


class PDFExtractor(LoggerMixin):
    """
    Extracts text content from PDF files using PyMuPDF.
    
    Handles text extraction while preserving structure, detecting
    mathematical content, and extracting relevant metadata.
    
    Attributes:
        data_dir: Root directory containing PDF files.
    
    Example:
        >>> extractor = PDFExtractor(Path("./data/raw"))
        >>> for doc in extractor.extract_all():
        ...     print(f"Extracted {doc.total_pages} pages from {doc.filename}")
    """
    
    def __init__(self, data_dir: Path) -> None:
        """
        Initialize the PDF extractor.
        
        Args:
            data_dir: Root directory containing PDF subdirectories.
        """
        self.data_dir = Path(data_dir)
        self.logger.info(f"PDFExtractor initialized with data_dir: {self.data_dir}")
        self.layout_analyzer = StatisticalLayoutAnalyzer()
    
    def find_pdfs(self) -> list[Path]:
        """
        Find all PDF files in the data directory.
        
        Recursively searches for .pdf files in all subdirectories.
        
        Returns:
            List of Path objects for found PDF files.
        """
        pdfs = list(self.data_dir.rglob("*.pdf"))
        self.logger.info(f"Found {len(pdfs)} PDF files in {self.data_dir}")
        return pdfs
    
    def _detect_source_type(self, file_path: Path) -> str:
        """
        Detect the source type based on file location.
        
        Args:
            file_path: Path to the PDF file.
        
        Returns:
            Source type string (book, lecture_notes, exercises).
        """
        # Check parent directories for type indicators
        parts = file_path.parts
        for part in parts:
            part_lower = part.lower()
            if part_lower in SOURCE_TYPE_MAP:
                return SOURCE_TYPE_MAP[part_lower]
        
        # Default to book if not in a recognized directory
        return "book"
    
    def _extract_author(self, filename: str, text_sample: str) -> str:
        """
        Extract or infer author from filename or content.
        
        Args:
            filename: Name of the PDF file.
            text_sample: Sample of text from the first pages.
        
        Returns:
            Author name or empty string if not detected.
        """
        # Common textbook authors in probability/statistics
        known_authors = [
            "Durrett", "Feller", "Ross", "Billingsley", "Ferrari",
            "Casella", "Berger", "Grimmett", "Stirzaker", "Williams"
        ]
        
        # Check filename first
        for author in known_authors:
            if author.lower() in filename.lower():
                return author
        
        # Check first page content
        for author in known_authors:
            if author in text_sample:
                return author
        
        # Extract from filename pattern like "author_title.pdf"
        name_parts = filename.replace("_", " ").replace("-", " ").split()
        if name_parts:
            # Capitalize first word as potential author
            return name_parts[0].capitalize()
        
        return ""
    
    def _detect_language(self, text: str) -> str:
        """
        Detect the language of the text.
        
        Uses simple heuristics based on common words.
        
        Args:
            text: Text to analyze.
        
        Returns:
            Language code: 'en', 'es', or 'mixed'.
        """
        try:
            from langdetect import detect, LangDetectException
            
            # Use a sample to avoid processing huge texts
            sample = text[:2000] if len(text) > 2000 else text
            
            try:
                lang = detect(sample)
                if lang in ["en", "es"]:
                    return lang
                return "en"  # Default to English for other languages
            except LangDetectException:
                return "en"
        except ImportError:
            # Fallback if langdetect not available
            spanish_indicators = [
                "el ", "la ", "los ", "las ", "de ", "que ", "en ", "y ",
                "sea ", "para ", "por ", "como ", "con ", "se "
            ]
            english_indicators = [
                "the ", "and ", "is ", "are ", "for ", "that ", "with ",
                "this ", "from ", "have ", "has "
            ]
            
            text_lower = text.lower()
            spanish_count = sum(1 for w in spanish_indicators if w in text_lower)
            english_count = sum(1 for w in english_indicators if w in text_lower)
            
            if spanish_count > english_count * 1.5:
                return "es"
            elif english_count > spanish_count * 1.5:
                return "en"
            else:
                return "mixed"
    
    def _detect_chapter(self, text: str) -> Optional[int]:
        """
        Detect chapter number from text content.
        
        Args:
            text: Text to analyze.
        
        Returns:
            Chapter number or None if not detected.
        """
        match = CHAPTER_PATTERN.search(text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None
    
    def _has_latex_content(self, text: str) -> bool:
        """
        Check if text contains LaTeX-style mathematical notation.
        
        Args:
            text: Text to analyze.
        
        Returns:
            True if LaTeX content detected.
        """
        return bool(LATEX_PATTERN.search(text))
    
    def _preserve_latex(self, text: str) -> str:
        """
        Mark LaTeX content with special tags for preservation.
        
        Args:
            text: Original text with LaTeX.
        
        Returns:
            Text with LaTeX wrapped in [LATEX]...[/LATEX] tags.
        """
        def wrap_latex(match):
            return f"[LATEX]{match.group(0)}[/LATEX]"
        
        return LATEX_PATTERN.sub(wrap_latex, text)
    
    def _clean_text(self, text: str, recurring_texts: set[str] = None) -> str:
        """
        Clean extracted text while preserving structure.
        
        Args:
            text: Raw extracted text.
            recurring_texts: Set of exact text strings to always remove (headers/footers).
        
        Returns:
            Cleaned text.
        """
        if recurring_texts:
            for bad_text in recurring_texts:
                text = text.replace(bad_text, "")
        
        # Merge hyphenated words at line breaks (e.g., "prob-\nability" -> "probability")
        # Pattern: Word, hyphen, newline (and spaces), Word
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove page headers/footers identifiers (isolated numbers)
        # Be careful not to remove "1." in lists.
        # Match line that is JUST a number
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_pdf(self, file_path: Path) -> PDFDocument:
        """
        Extract text content from a single PDF file.
        
        Args:
            file_path: Path to the PDF file.
        
        Returns:
            PDFDocument with extracted pages.
        
        Raises:
            FileNotFoundError: If PDF file doesn't exist.
            RuntimeError: If PDF extraction fails.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        self.logger.debug(f"Extracting PDF: {file_path}")
        
        doc = PDFDocument(file_path=file_path)
        source_type = self._detect_source_type(file_path)
        
        try:
            pdf = fitz.open(file_path)
            doc.total_pages = len(pdf)
            
            # 1. Statistical Layout Analysis
            self.logger.debug(f"Running layout analysis on {file_path.name}")
            layout_stats = self.layout_analyzer.analyze(pdf)
            self.logger.info(f"Analysis for {file_path.name}: Header < {layout_stats.header_height}px, Footer > page_h - {layout_stats.footer_height}px")
            
            # Get author from first page
            first_page_text = pdf[0].get_text() if pdf else ""
            author = self._extract_author(file_path.stem, first_page_text)
            
            current_chapter: Optional[int] = None
            
            for page_num, page in enumerate(pdf, start=1):
                page_height = page.rect.height
                
                # Use "dict" extraction to filter blocks by position
                blocks = page.get_text("dict")["blocks"]
                valid_text_parts = []
                
                for block in blocks:
                    if "lines" not in block:
                        continue
                        
                    bbox = block["bbox"]
                    y0, y1 = bbox[1], bbox[3]
                    
                    text_content = " ".join([span["text"] for line in block["lines"] for span in line["spans"]]).strip()
                    if not text_content: continue

                    # ROBUST FILTERING LOGIC
                    
                    # 1. Always remove explicitly recurring text (Running Headers/Footers)
                    if text_content in layout_stats.recurring_texts:
                        continue
                        
                    # 2. Check Exclusion Zones
                    in_header_zone = y1 < layout_stats.header_height
                    in_footer_zone = y0 > (page_height - layout_stats.footer_height)
                    
                    if in_header_zone or in_footer_zone:
                        # If it's in the zone, ONLY remove if it looks like a page number
                        # or if we are very confident it's junk.
                        # We already checked occurrences.
                        
                        # Heuristic: If it's just a number, remove it (Page Number)
                        if text_content.replace(" ", "").isdigit():
                            continue
                            
                        # Heuristic: If it's mixed text but NOT recurring, it might be a Section Title.
                        # KEEP IT.
                        pass
                        
                    # Extract text from this block (re-extracting line by line to preserve newlines if needed)
                    block_text = ""
                    for line in block["lines"]:
                        line_text = " ".join(span["text"] for span in line["spans"])
                        block_text += line_text + "\n"
                    
                    valid_text_parts.append(block_text)
                
                text = "\n".join(valid_text_parts)
                
                if not text.strip():
                    self.logger.debug(f"Page {page_num} is empty after filtering, skipping")
                    continue
                
                # Clean and process text
                cleaned_text = self._clean_text(text, recurring_texts=layout_stats.recurring_texts)
                
                # Detect chapter if present on this page
                detected_chapter = self._detect_chapter(cleaned_text)
                if detected_chapter:
                    current_chapter = detected_chapter
                
                # Check for LaTeX and preserve it
                has_latex = self._has_latex_content(cleaned_text)
                if has_latex:
                    cleaned_text = self._preserve_latex(cleaned_text)
                
                # Detect language (use sample from this page)
                language = self._detect_language(cleaned_text)
                
                # Create extracted page
                extracted_page = ExtractedPage(
                    text=cleaned_text,
                    page_number=page_num,
                    source_file=file_path.name,
                    source_path=str(file_path),
                    source_type=source_type,
                    author=author,
                    chapter=current_chapter,
                    language=language,
                    has_latex=has_latex,
                )
                
                doc.pages.append(extracted_page)
            
            pdf.close()
            
            self.logger.info(
                f"Extracted {len(doc.pages)} pages from {file_path.name} "
                f"(type: {source_type}, author: {author or 'unknown'})"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to extract PDF {file_path}: {e}")
            raise RuntimeError(f"PDF extraction failed: {e}") from e
        
        return doc
    
    def extract_all(self) -> Generator[PDFDocument, None, None]:
        """
        Extract all PDFs from the data directory.
        
        Yields:
            PDFDocument for each successfully extracted PDF.
        """
        pdfs = self.find_pdfs()
        
        for pdf_path in pdfs:
            try:
                yield self.extract_pdf(pdf_path)
            except Exception as e:
                self.logger.error(f"Skipping {pdf_path.name}: {e}")
                continue
    
    def get_statistics(self) -> dict:
        """
        Get statistics about PDFs in the data directory.
        
        Returns:
            Dictionary with PDF statistics.
        """
        pdfs = self.find_pdfs()
        
        stats = {
            "total_pdfs": len(pdfs),
            "by_type": {"book": 0, "lecture_notes": 0, "exercises": 0},
            "files": []
        }
        
        for pdf in pdfs:
            source_type = self._detect_source_type(pdf)
            stats["by_type"][source_type] = stats["by_type"].get(source_type, 0) + 1
            stats["files"].append({
                "name": pdf.name,
                "type": source_type,
                "path": str(pdf)
            })
        
        return stats
