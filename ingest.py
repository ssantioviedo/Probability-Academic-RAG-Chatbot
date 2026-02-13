#!/usr/bin/env python3
"""
PDF Ingestion Script for Academic RAG Chatbot.

This script processes PDF files from the data directory, extracts text,
creates chunks, generates embeddings, and stores them in ChromaDB.

Usage:
    python ingest.py --data-dir data/raw           # Incremental indexing
    python ingest.py --data-dir data/raw --reset   # Reset and re-index all
    python ingest.py --stats                       # Show index statistics
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config, get_config
from src.utils.logger import setup_logger, get_logger, log_processing_stats
from src.ingestion.pdf_extractor import PDFExtractor
from src.ingestion.chunker import TextChunker, create_chunker_from_config
from src.ingestion.indexer import ChromaIndexer, create_indexer_from_config


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into the RAG chatbot's vector database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py --data-dir data/raw           # Index new PDFs only
  python ingest.py --data-dir data/raw --reset   # Reset DB and re-index all
  python ingest.py --stats                       # Show current statistics
        """
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing PDF files to ingest (default: from config)"
    )
    
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the database and re-index all PDFs"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics about the current index and exit"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually indexing"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging"
    )
    
    return parser.parse_args()


def show_statistics(config: Config) -> None:
    """
    Display statistics about the current index.
    
    Args:
        config: Application configuration.
    """
    print("\n" + "=" * 60)
    print("📊 INDEX STATISTICS")
    print("=" * 60)
    
    try:
        indexer = create_indexer_from_config(config)
        stats = indexer.get_statistics()
        
        print(f"\n📁 Collection: {stats.get('collection_name', 'N/A')}")
        print(f"🧠 Embedding Model: {stats.get('embedding_model', 'N/A')}")
        print(f"📂 Storage Path: {stats.get('persist_directory', 'N/A')}")
        
        print(f"\n📈 Total Chunks: {stats.get('total_chunks', 0):,}")
        print(f"📚 Total Files: {stats.get('total_files', 0)}")
        
        if stats.get("by_source_type"):
            print("\n📖 By Source Type:")
            for source_type, count in stats["by_source_type"].items():
                print(f"   • {source_type}: {count:,} chunks")
        
        if stats.get("by_author"):
            print("\n✍️  By Author:")
            for author, count in sorted(
                stats["by_author"].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]:  # Top 10 authors
                print(f"   • {author}: {count:,} chunks")
        
        if stats.get("languages"):
            print("\n🌐 By Language:")
            for lang, count in stats["languages"].items():
                print(f"   • {lang}: {count:,} chunks")
        
        if stats.get("unique_files"):
            print("\n📄 Indexed Files:")
            for filename in sorted(stats["unique_files"]):
                print(f"   • {filename}")
        
    except Exception as e:
        print(f"\n❌ Error accessing index: {e}")
        print("   The index may not exist yet. Run ingestion first.")
    
    print("\n" + "=" * 60 + "\n")


def run_ingestion(
    config: Config,
    data_dir: Optional[Path],
    reset: bool,
    dry_run: bool,
    logger
) -> None:
    """
    Run the complete ingestion pipeline.
    
    Args:
        config: Application configuration.
        data_dir: Directory with PDF files.
        reset: Whether to reset the database.
        dry_run: If True, show what would be processed without indexing.
        logger: Logger instance.
    """
    start_time = time.time()
    
    # Determine data directory
    if data_dir:
        data_path = Path(data_dir)
    else:
        data_path = Path(config.data_dir)
    
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_path}")
        print(f"\n❌ Error: Data directory does not exist: {data_path}")
        print("   Please create the directory and add PDF files.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("📚 ACADEMIC RAG CHATBOT - PDF INGESTION")
    print("=" * 60)
    print(f"\n📂 Data Directory: {data_path.absolute()}")
    print(f"💾 ChromaDB Path: {config.chroma_db_path}")
    print(f"🧠 Embedding Model: {config.embedding_model}")
    print(f"📏 Chunk Size: {config.chunk_size} tokens")
    print(f"🔄 Reset Mode: {'Yes' if reset else 'No'}")
    print(f"🧪 Dry Run: {'Yes' if dry_run else 'No'}")
    
    # Step 1: Initialize PDF Extractor
    print("\n" + "-" * 40)
    print("Step 1: Scanning for PDFs...")
    print("-" * 40)
    
    extractor = PDFExtractor(data_path)
    pdf_files = extractor.find_pdfs()
    
    if not pdf_files:
        print("\n⚠️  No PDF files found in the data directory!")
        print("   Please add PDF files to:")
        print(f"   • {data_path / 'books'} (textbooks)")
        print(f"   • {data_path / 'lecture_notes'} (lecture notes)")
        print(f"   • {data_path / 'exercises'} (exercise guides)")
        return
    
    pdf_stats = extractor.get_statistics()
    print(f"\n📄 Found {len(pdf_files)} PDF files:")
    for ptype, count in pdf_stats["by_type"].items():
        if count > 0:
            print(f"   • {ptype}: {count}")
    
    # Step 2: Initialize Indexer
    print("\n" + "-" * 40)
    print("Step 2: Initializing ChromaDB...")
    print("-" * 40)
    
    indexer = create_indexer_from_config(config)
    
    if reset:
        print("⚠️  Resetting database...")
        indexer.reset_collection()
        existing_files = set()
    else:
        existing_files = indexer.get_existing_files()
        if existing_files:
            print(f"📋 Already indexed: {len(existing_files)} files")
    
    # Filter out already indexed files
    new_pdfs = [
        p for p in pdf_files 
        if p.name not in existing_files
    ]
    
    if not new_pdfs:
        print("\n✅ All PDFs are already indexed!")
        total_time = time.time() - start_time
        print(f"\n⏱️  Total time: {total_time:.2f} seconds")
        return
    
    print(f"\n📥 New PDFs to process: {len(new_pdfs)}")
    for pdf in new_pdfs:
        print(f"   • {pdf.name}")
    
    if dry_run:
        print("\n🧪 Dry run complete. No changes made.")
        return
    
    # Step 3: Extract and Chunk
    print("\n" + "-" * 40)
    print("Step 3: Extracting and chunking PDFs...")
    print("-" * 40)
    
    chunker = create_chunker_from_config(config)
    all_chunks = []
    extraction_errors = []
    
    for pdf_path in tqdm(new_pdfs, desc="Processing PDFs", unit="file"):
        try:
            # Extract PDF
            document = extractor.extract_pdf(pdf_path)
            
            # Chunk document
            chunks = chunker.chunk_document(document)
            all_chunks.extend(chunks)
            
            logger.info(
                f"Processed {pdf_path.name}: "
                f"{document.total_pages} pages, {len(chunks)} chunks"
            )
            
        except Exception as e:
            extraction_errors.append((pdf_path.name, str(e)))
            logger.error(f"Failed to process {pdf_path.name}: {e}")
    
    if extraction_errors:
        print(f"\n⚠️  {len(extraction_errors)} files had errors:")
        for filename, error in extraction_errors:
            print(f"   • {filename}: {error[:50]}...")
    
    print(f"\n📦 Total chunks created: {len(all_chunks):,}")
    
    if not all_chunks:
        print("\n⚠️  No chunks to index!")
        return
    
    # Step 4: Index chunks
    print("\n" + "-" * 40)
    print("Step 4: Indexing chunks in ChromaDB...")
    print("-" * 40)
    
    index_stats = indexer.add_chunks(
        all_chunks, 
        show_progress=True,
        build_sparse_index=True,
        sparse_index_path=config.sparse_index_path
    )
    
    print(f"\n✅ Indexing complete:")
    print(f"   • Added: {index_stats['added']:,} chunks")
    print(f"   • Skipped (duplicates): {index_stats['skipped']:,}")
    print(f"   • Errors: {index_stats['errors']:,}")
    
    # Final statistics
    total_time = time.time() - start_time
    final_stats = indexer.get_statistics()
    
    print("\n" + "=" * 60)
    print("📊 FINAL STATISTICS")
    print("=" * 60)
    print(f"\n📈 Total chunks in database: {final_stats['total_chunks']:,}")
    print(f"📚 Total files indexed: {final_stats['total_files']}")
    print(f"⏱️  Total processing time: {total_time:.2f} seconds")
    
    log_processing_stats(
        logger,
        "PDF ingestion",
        len(new_pdfs),
        total_time,
        {"chunks": len(all_chunks), "errors": len(extraction_errors)}
    )
    
    print("\n" + "=" * 60)
    print("✅ INGESTION COMPLETE")
    print("=" * 60 + "\n")


def main() -> None:
    """Main entry point for the ingestion script."""
    args = parse_arguments()
    
    # Load configuration
    config = get_config()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else config.log_level
    logger = setup_logger(log_level=log_level)
    logger = get_logger("ingest")
    
    logger.info("Starting ingestion script")
    logger.debug(f"Arguments: {args}")
    logger.debug(f"Config: {config}")
    
    # Show stats if requested
    if args.stats:
        show_statistics(config)
        return
    
    # Run ingestion
    try:
        run_ingestion(
            config=config,
            data_dir=Path(args.data_dir) if args.data_dir else None,
            reset=args.reset,
            dry_run=args.dry_run,
            logger=logger
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Ingestion interrupted by user.")
        logger.warning("Ingestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logger.exception(f"Fatal error during ingestion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
