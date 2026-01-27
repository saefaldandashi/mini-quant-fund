"""
Digest Engine - Main orchestrator for Daily Digest generation.

Orchestrates:
1. Feed loading from intel feed output
2. Item selection per category
3. LLM summarization
4. HTML rendering
5. PDF export
6. JSON output

Usage:
    from src.digest import generate_daily_digest
    
    result = generate_daily_digest(
        date="2026-01-27",
        feed_path="outputs/cache/filtered_events_cache.json",
        outdir="outputs/digest"
    )
"""

import json
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import pytz

from .schema import (
    FeedItem,
    DigestCategory,
    DigestSection,
    DigestOutput,
    DigestMetadata,
    ExecutiveBrief,
    MarketSnapshot,
    CATEGORY_DISPLAY_NAMES,
    CATEGORY_ICONS,
)
from .selector import ItemSelector, SelectionConfig, SelectionResult
from .llm_summarizer import LLMSummarizer
from .renderer import DigestRenderer
from .pdf_export import PDFExporter, check_pdf_support

logger = logging.getLogger(__name__)


class FeedAdapter:
    """
    Adapter to load feed items from various sources.
    
    Supports:
    - JSON files (filtered_events_cache.json)
    - Parquet files
    - Direct list of dictionaries
    """
    
    def __init__(self, feed_path: Optional[str] = None):
        self.feed_path = feed_path
        
    def load(
        self, 
        path: Optional[str] = None,
        target_date: Optional[str] = None,
    ) -> List[FeedItem]:
        """
        Load feed items from file.
        
        Args:
            path: Path to feed file (overrides constructor path)
            target_date: Optional date filter (YYYY-MM-DD)
            
        Returns:
            List of FeedItem objects
        """
        path = path or self.feed_path
        if not path:
            # Default to filtered events cache
            path = "outputs/cache/filtered_events_cache.json"
            
        path = Path(path)
        
        if not path.exists():
            logger.warning(f"Feed file not found: {path}")
            return []
        
        # Load based on file type
        if path.suffix == '.json':
            return self._load_json(path, target_date)
        elif path.suffix == '.parquet':
            return self._load_parquet(path, target_date)
        else:
            logger.warning(f"Unknown file type: {path.suffix}")
            return []
    
    def _load_json(
        self, 
        path: Path, 
        target_date: Optional[str]
    ) -> List[FeedItem]:
        """Load from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        events = data.get('events', [])
        if not events and isinstance(data, list):
            events = data
            
        items = []
        for event in events:
            try:
                item = FeedItem(**event)
                
                # Filter by date if specified
                if target_date:
                    item_date = item.timestamp.strftime('%Y-%m-%d')
                    if item_date != target_date:
                        continue
                        
                items.append(item)
            except Exception as e:
                logger.debug(f"Could not parse event: {e}")
                continue
                
        logger.info(f"Loaded {len(items)} items from {path}")
        return items
    
    def _load_parquet(
        self, 
        path: Path, 
        target_date: Optional[str]
    ) -> List[FeedItem]:
        """Load from Parquet file."""
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            
            items = []
            for _, row in df.iterrows():
                try:
                    item = FeedItem(**row.to_dict())
                    
                    if target_date:
                        item_date = item.timestamp.strftime('%Y-%m-%d')
                        if item_date != target_date:
                            continue
                            
                    items.append(item)
                except Exception:
                    continue
                    
            logger.info(f"Loaded {len(items)} items from {path}")
            return items
            
        except ImportError:
            logger.error("pandas required for parquet support")
            return []
    
    def load_from_list(self, events: List[Dict]) -> List[FeedItem]:
        """Load from list of dictionaries."""
        items = []
        for event in events:
            try:
                items.append(FeedItem(**event))
            except Exception as e:
                logger.debug(f"Could not parse event: {e}")
                continue
        return items


class DigestEngine:
    """
    Main engine for generating daily digests.
    
    Orchestrates the full pipeline from feed to output files.
    """
    
    def __init__(
        self,
        feed_adapter: Optional[FeedAdapter] = None,
        selector: Optional[ItemSelector] = None,
        summarizer: Optional[LLMSummarizer] = None,
        renderer: Optional[DigestRenderer] = None,
        pdf_exporter: Optional[PDFExporter] = None,
        output_dir: str = "outputs/digest",
    ):
        self.feed_adapter = feed_adapter or FeedAdapter()
        self.selector = selector or ItemSelector()
        self.summarizer = summarizer or LLMSummarizer()
        self.renderer = renderer or DigestRenderer()
        self.pdf_exporter = pdf_exporter or PDFExporter()
        self.output_dir = Path(output_dir)
    
    def generate(
        self,
        date: Optional[str] = None,
        feed_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        include_pdf: bool = True,
        items_per_category: int = 3,
        impact_threshold: float = 0.25,
        skip_llm: bool = False,
    ) -> DigestOutput:
        """
        Generate a complete daily digest.
        
        Args:
            date: Target date (YYYY-MM-DD), defaults to today
            feed_path: Path to feed file
            output_dir: Override output directory
            include_pdf: Generate PDF output
            items_per_category: Number of items per category
            impact_threshold: Minimum impact score
            skip_llm: Skip LLM summarization (use fallbacks)
            
        Returns:
            DigestOutput with all generated content and file paths
        """
        start_time = time.time()
        
        # Set defaults
        if date is None:
            date = datetime.now(pytz.UTC).strftime('%Y-%m-%d')
        
        outdir = Path(output_dir or self.output_dir) / date
        outdir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating digest for {date}")
        
        # Step 1: Load feed items
        logger.info("Step 1: Loading feed items...")
        items = self.feed_adapter.load(feed_path)
        
        if not items:
            raise ValueError(f"No items found in feed: {feed_path}")
        
        logger.info(f"  Loaded {len(items)} items")
        
        # Step 2: Select top items per category
        logger.info("Step 2: Selecting top items per category...")
        self.selector.config.items_per_category = items_per_category
        self.selector.config.impact_threshold = impact_threshold
        
        selection_results = self.selector.select(items)
        
        total_selected = sum(len(r.selected) for r in selection_results.values())
        logger.info(f"  Selected {total_selected} items across {len(selection_results)} categories")
        
        if total_selected == 0:
            raise ValueError("No items passed selection thresholds")
        
        # Step 3: Generate summaries per category
        logger.info("Step 3: Generating category summaries...")
        sections = []
        category_summaries = {}
        
        for category, result in selection_results.items():
            if not result.selected:
                continue
                
            logger.info(f"  Processing {category.value} ({len(result.selected)} items)")
            
            # Create section
            section = DigestSection(
                category=category,
                display_name=CATEGORY_DISPLAY_NAMES.get(category, str(category)),
                icon=CATEGORY_ICONS.get(category, "📋"),
                items=result.selected,
                avg_impact_score=result.avg_impact,
                max_impact_score=result.max_impact,
                total_items_considered=result.total_considered,
            )
            
            # Generate summary
            if not skip_llm:
                summary, error = self.summarizer.summarize_category(
                    category, result.selected
                )
                if summary:
                    section.summary = summary
                    category_summaries[category] = summary
                if error:
                    section.llm_failed = True
                    section.llm_error = error
            else:
                # Use fallback
                summary, _ = self.summarizer._fallback_summary(result.selected), None
                section.summary = summary
                category_summaries[category] = summary
            
            sections.append(section)
        
        # Step 4: Generate executive brief
        logger.info("Step 4: Generating executive brief...")
        if not skip_llm:
            exec_brief, error = self.summarizer.generate_executive_brief(
                date, category_summaries
            )
        else:
            exec_brief = self.summarizer._fallback_executive_brief(category_summaries)
            
        # Step 5: Get market snapshot (if available)
        logger.info("Step 5: Getting market snapshot...")
        market_snapshot = self._get_market_snapshot()
        
        # Step 6: Build output
        logger.info("Step 6: Building digest output...")
        generation_time = time.time() - start_time
        
        metadata = DigestMetadata(
            date=date,
            generated_at=datetime.now(pytz.UTC),
            version="1.0",
            feed_path=str(feed_path or "default"),
            total_items_processed=len(items),
            total_items_selected=total_selected,
            categories_with_content=len(sections),
            llm_model=getattr(self.summarizer.llm_service, 'model', None) if self.summarizer.llm_service else None,
            generation_time_seconds=generation_time,
        )
        
        digest = DigestOutput(
            metadata=metadata,
            executive_brief=exec_brief,
            market_snapshot=market_snapshot,
            sections=sorted(sections, key=lambda s: s.max_impact_score, reverse=True),
        )
        
        # Step 7: Render HTML
        logger.info("Step 7: Rendering HTML...")
        html_path = outdir / "daily_digest.html"
        self.renderer.render(digest, str(html_path))
        digest.html_path = str(html_path)
        
        # Step 8: Export PDF
        if include_pdf:
            logger.info("Step 8: Exporting PDF...")
            pdf_path = outdir / "daily_digest.pdf"
            pdf_support = check_pdf_support()
            
            if pdf_support['any_available']:
                success = self.pdf_exporter.export(
                    str(html_path),
                    str(pdf_path),
                    title="Daily Intelligence Digest",
                    date=date,
                )
                if success:
                    digest.pdf_path = str(pdf_path)
            else:
                logger.warning("PDF export skipped - no engine available")
        
        # Step 9: Save JSON
        logger.info("Step 9: Saving JSON...")
        json_path = outdir / "daily_digest.json"
        self._save_json(digest, json_path)
        digest.json_path = str(json_path)
        
        logger.info(f"✅ Digest generated in {generation_time:.1f}s")
        logger.info(f"   HTML: {digest.html_path}")
        logger.info(f"   PDF: {digest.pdf_path or 'N/A'}")
        logger.info(f"   JSON: {digest.json_path}")
        
        return digest
    
    def _get_market_snapshot(self) -> Optional[MarketSnapshot]:
        """Get current market data for snapshot."""
        try:
            # Try to get from existing market data loader
            from src.data.market_data import MarketDataLoader
            
            loader = MarketDataLoader()
            
            # Get key indices
            symbols = ["SPY", "QQQ", "TLT", "GLD", "USO", "DXY"]
            prices = loader.get_current_prices(symbols)
            
            if not prices:
                return None
                
            return MarketSnapshot(
                spx=prices.get("SPY", 0) * 10,  # Approximate SPX from SPY
                brent=prices.get("USO", 0) * 1.05 if prices.get("USO") else None,
                gold=prices.get("GLD", 0) * 17 if prices.get("GLD") else None,  # Approximate
                timestamp=datetime.now(pytz.UTC),
            )
            
        except Exception as e:
            logger.debug(f"Could not get market snapshot: {e}")
            return None
    
    def _save_json(self, digest: DigestOutput, path: Path):
        """Save digest to JSON file."""
        # Convert to dict, handling datetime serialization
        data = digest.dict()
        
        def serialize(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, 'value'):  # Enum
                return obj.value
            return obj
        
        def deep_serialize(d):
            if isinstance(d, dict):
                return {k: deep_serialize(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [deep_serialize(i) for i in d]
            else:
                return serialize(d)
        
        data = deep_serialize(data)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved JSON to {path}")


def generate_daily_digest(
    date: Optional[str] = None,
    feed_path: Optional[str] = None,
    outdir: str = "outputs/digest",
    items_per_category: int = 20,  # Comprehensive coverage
    impact_threshold: float = 0.30,  # All notable events
    include_pdf: bool = True,
    skip_llm: bool = False,
) -> Dict[str, Any]:
    """
    Main entry point for digest generation.
    
    Args:
        date: Target date (YYYY-MM-DD), defaults to today
        feed_path: Path to feed file
        outdir: Output directory
        items_per_category: Number of items per category
        impact_threshold: Minimum impact score for selection
        include_pdf: Generate PDF output
        skip_llm: Skip LLM summarization
        
    Returns:
        Dict with paths to generated files:
        {
            'html_path': '...',
            'pdf_path': '...',  # None if not generated
            'json_path': '...',
            'success': True/False,
            'error': None or error message,
            'metadata': {...}
        }
    """
    try:
        engine = DigestEngine(output_dir=outdir)
        
        digest = engine.generate(
            date=date,
            feed_path=feed_path,
            include_pdf=include_pdf,
            items_per_category=items_per_category,
            impact_threshold=impact_threshold,
            skip_llm=skip_llm,
        )
        
        return {
            'html_path': digest.html_path,
            'pdf_path': digest.pdf_path,
            'json_path': digest.json_path,
            'success': True,
            'error': None,
            'metadata': {
                'date': digest.metadata.date,
                'items_processed': digest.metadata.total_items_processed,
                'items_selected': digest.metadata.total_items_selected,
                'categories': digest.metadata.categories_with_content,
                'generation_time': digest.metadata.generation_time_seconds,
            }
        }
        
    except Exception as e:
        logger.error(f"Digest generation failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'html_path': None,
            'pdf_path': None,
            'json_path': None,
            'success': False,
            'error': str(e),
            'metadata': None,
        }
