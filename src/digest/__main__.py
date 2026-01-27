"""
CLI Entry Point for Daily Digest Generation.

Usage:
    python -m src.digest --date 2026-01-27
    python -m src.digest --feed-path outputs/cache/filtered_events_cache.json
    python -m src.digest --skip-llm  # Use fallback summaries
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Daily Intelligence Digest from Global Intelligence Feed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.digest
    Generate digest for today using default feed
    
  python -m src.digest --date 2026-01-27
    Generate digest for specific date
    
  python -m src.digest --feed-path outputs/cache/filtered_events_cache.json
    Use specific feed file
    
  python -m src.digest --items 5 --threshold 0.3
    Select 5 items per category with 0.3 impact threshold
    
  python -m src.digest --skip-llm
    Skip LLM summarization, use fallback summaries
    
  python -m src.digest --no-pdf
    Skip PDF generation
        """
    )
    
    parser.add_argument(
        '--date', '-d',
        type=str,
        default=None,
        help='Target date (YYYY-MM-DD), defaults to today'
    )
    
    parser.add_argument(
        '--feed-path', '-f',
        type=str,
        default=None,
        help='Path to feed file (JSON or Parquet)'
    )
    
    parser.add_argument(
        '--outdir', '-o',
        type=str,
        default='outputs/digest',
        help='Output directory (default: outputs/digest)'
    )
    
    parser.add_argument(
        '--items', '-n',
        type=int,
        default=3,
        help='Number of items per category (default: 3)'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.25,
        help='Minimum impact score threshold (default: 0.25)'
    )
    
    parser.add_argument(
        '--skip-llm',
        action='store_true',
        help='Skip LLM summarization, use fallback summaries'
    )
    
    parser.add_argument(
        '--no-pdf',
        action='store_true',
        help='Skip PDF generation'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Import here to avoid circular imports
    from .digest_engine import generate_daily_digest
    
    print("=" * 60)
    print("📊 Daily Intelligence Digest Generator")
    print("=" * 60)
    print()
    
    result = generate_daily_digest(
        date=args.date,
        feed_path=args.feed_path,
        outdir=args.outdir,
        items_per_category=args.items,
        impact_threshold=args.threshold,
        include_pdf=not args.no_pdf,
        skip_llm=args.skip_llm,
    )
    
    print()
    print("=" * 60)
    
    if result['success']:
        print("✅ Digest generation SUCCESSFUL")
        print()
        print(f"📄 HTML: {result['html_path']}")
        print(f"📕 PDF:  {result['pdf_path'] or 'Not generated'}")
        print(f"📋 JSON: {result['json_path']}")
        print()
        if result['metadata']:
            meta = result['metadata']
            print(f"📊 Stats:")
            print(f"   Date: {meta['date']}")
            print(f"   Items processed: {meta['items_processed']}")
            print(f"   Items selected: {meta['items_selected']}")
            print(f"   Categories: {meta['categories']}")
            print(f"   Generation time: {meta['generation_time']:.1f}s")
    else:
        print("❌ Digest generation FAILED")
        print(f"   Error: {result['error']}")
        sys.exit(1)
    
    print("=" * 60)


if __name__ == '__main__':
    main()
