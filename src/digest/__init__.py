"""
Daily Digest Module

Generates investor-grade daily digests from the Global Intelligence Feed.
Produces clean HTML + PDF reports with structured LLM summaries for
US and GCC market impacts.

Components:
- schema: Pydantic models for feed items and digest output
- selector: Selects top N items per category with scoring
- llm_summarizer: LLM wrapper with strict prompts
- renderer: Jinja2 HTML rendering
- pdf_export: HTML to PDF conversion
- digest_engine: Main orchestrator

Usage:
    from src.digest import generate_daily_digest
    
    result = generate_daily_digest(
        date="2026-01-27",
        feed_path="outputs/cache/filtered_events_cache.json",
        outdir="outputs/digest"
    )
"""

from .digest_engine import DigestEngine, generate_daily_digest
from .schema import FeedItem, DigestSection, DigestOutput, CategoryMapping

__all__ = [
    'DigestEngine',
    'generate_daily_digest',
    'FeedItem',
    'DigestSection', 
    'DigestOutput',
    'CategoryMapping',
]
