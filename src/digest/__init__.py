"""
Daily Digest Module

Generates investor-grade daily digests from the Global Intelligence Feed.
Produces clean HTML + PDF reports with structured LLM summaries for
US and GCC market impacts.

Components:
- schema: Pydantic models for feed items and digest output
- selector: Selects top N items per category with scoring
- llm_summarizer: LLM wrapper with strict prompts and market analysis
- renderer: Jinja2 HTML rendering
- pdf_export: HTML to PDF conversion
- digest_engine: Main orchestrator
- ml_integration: Feeds signals to learning system for smarter strategies

NEW FEATURES:
- Detailed GCC market analysis (Tadawul, DFM, ADX, QSE)
- Market Outlook with actionable guidance
- Noise vs Signal classification
- Key Levels to Watch
- ML-ready strategy signals (bias, conviction, volatility expectation)
- Automatic signal saving for learning system integration

Usage:
    from src.digest import generate_daily_digest, get_digest_signals
    
    # Generate digest
    result = generate_daily_digest(
        date="2026-01-27",
        feed_path="outputs/cache/filtered_events_cache.json",
        outdir="outputs/digest"
    )
    
    # Get signals for strategies
    signals = get_digest_signals()
    print(f"Bias: {signals['overall_bias']}, Conviction: {signals['conviction_score']}")
"""

from .digest_engine import DigestEngine, generate_daily_digest
from .schema import FeedItem, DigestSection, DigestOutput
from .ml_integration import get_digest_ml_integration, DigestMLIntegration


def get_digest_signals():
    """
    Get current digest signals for strategy use.
    
    Returns dict with:
    - overall_bias: BULLISH, BEARISH, NEUTRAL
    - conviction_score: 1-10
    - volatility_expectation: LOW, MEDIUM, HIGH, SPIKE
    - risk_tone: Risk-On, Risk-Off, Neutral
    - sector_tilts: dict of sector -> overweight/underweight
    - act_on: list of actionable news
    - ignore: list of noise to ignore
    - wait_for: list of things needing more clarity
    """
    ml = get_digest_ml_integration()
    return ml.get_current_signals()


def get_bias_adjustment():
    """Get numeric bias adjustment (-1 to +1) for strategies."""
    ml = get_digest_ml_integration()
    return ml.get_bias_adjustment()


def should_reduce_exposure():
    """Check if digest signals suggest reducing exposure."""
    ml = get_digest_ml_integration()
    return ml.should_reduce_exposure()


def list_digests(outdir: str = "outputs/digest"):
    """List available digest dates."""
    from pathlib import Path
    digest_dir = Path(outdir)
    if not digest_dir.exists():
        return []
    
    digests = []
    for date_dir in sorted(digest_dir.iterdir(), reverse=True):
        if date_dir.is_dir() and (date_dir / "daily_digest.html").exists():
            digests.append({
                "date": date_dir.name,
                "html_path": str(date_dir / "daily_digest.html"),
                "pdf_path": str(date_dir / "daily_digest.pdf") if (date_dir / "daily_digest.pdf").exists() else None,
                "json_path": str(date_dir / "daily_digest.json") if (date_dir / "daily_digest.json").exists() else None,
            })
    return digests


def get_digest_json(date: str, outdir: str = "outputs/digest"):
    """Get digest JSON for a specific date."""
    from pathlib import Path
    import json
    
    json_path = Path(outdir) / date / "daily_digest.json"
    if not json_path.exists():
        return None
    
    with open(json_path, 'r') as f:
        return json.load(f)


def get_latest_digest_json(outdir: str = "outputs/digest"):
    """Get the most recent digest JSON."""
    digests = list_digests(outdir)
    if not digests:
        return None
    return get_digest_json(digests[0]["date"], outdir)


__all__ = [
    # Core
    'DigestEngine',
    'generate_daily_digest',
    'FeedItem',
    'DigestSection', 
    'DigestOutput',
    
    # ML Integration
    'get_digest_ml_integration',
    'DigestMLIntegration',
    'get_digest_signals',
    'get_bias_adjustment',
    'should_reduce_exposure',
    
    # Utilities
    'list_digests',
    'get_digest_json',
    'get_latest_digest_json',
]
