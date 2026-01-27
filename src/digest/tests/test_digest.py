"""
Tests for Daily Digest module.

Run with: pytest src/digest/tests/test_digest.py -v
"""

import pytest
import json
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import pytz

from src.digest.schema import (
    FeedItem,
    DigestCategory,
    DigestSection,
    DigestOutput,
    CATEGORY_MAPPING,
)
from src.digest.selector import ItemSelector, SelectionConfig, select_items
from src.digest.llm_summarizer import SummaryParser
from src.digest.renderer import DigestRenderer
from src.digest.digest_engine import FeedAdapter, DigestEngine, generate_daily_digest


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def sample_feed_items():
    """Create sample feed items for testing."""
    now = datetime.now(pytz.UTC)
    
    return [
        FeedItem(
            event_id="test1",
            timestamp=now - timedelta(hours=1),
            headline="Fed signals potential rate cut amid inflation concerns",
            summary="The Federal Reserve indicated...",
            source="reuters",
            url="https://example.com/1",
            category="central_bank",
            tags=["central_bank", "rates"],
            matched_keywords=["fed", "rate cut"],
            relevance_score=0.8,
            impact_score=0.75,
            credibility_score=0.9,
            novelty_score=1.0,
            final_score=0.8,
            direction="rates_down",
            direction_confidence=0.7,
            affected_assets=["TLT", "SPY"],
            affected_regions=["americas"],
            rationale="High impact Fed news",
        ),
        FeedItem(
            event_id="test2",
            timestamp=now - timedelta(hours=2),
            headline="Oil prices surge after OPEC cuts",
            summary="OPEC announced production cuts...",
            source="bloomberg",
            url="https://example.com/2",
            category="energy_commodity",
            tags=["energy", "opec"],
            matched_keywords=["opec", "oil"],
            relevance_score=0.7,
            impact_score=0.65,
            credibility_score=0.9,
            novelty_score=1.0,
            final_score=0.75,
            direction="oil_up",
            direction_confidence=0.8,
            affected_assets=["USO", "XLE"],
            affected_regions=["middle_east"],
            rationale="Energy supply impact",
        ),
        FeedItem(
            event_id="test3",
            timestamp=now - timedelta(hours=3),
            headline="Geopolitical tensions rise in region",
            summary="Military activity reported...",
            source="guardian_world",
            url="https://example.com/3",
            category="geopolitical",
            tags=["geopolitical", "conflict"],
            matched_keywords=["military", "tensions"],
            relevance_score=0.6,
            impact_score=0.55,
            credibility_score=0.7,
            novelty_score=1.0,
            final_score=0.6,
            direction="risk_off",
            direction_confidence=0.5,
            affected_assets=["GLD", "VIX"],
            affected_regions=["middle_east", "europe"],
            rationale="Geopolitical risk",
        ),
        FeedItem(
            event_id="test4",
            timestamp=now - timedelta(hours=4),
            headline="Low impact local news",
            summary="Local event happened...",
            source="local_news",
            url="https://example.com/4",
            category="irrelevant",
            tags=[],
            matched_keywords=[],
            relevance_score=0.1,
            impact_score=0.1,
            credibility_score=0.3,
            novelty_score=1.0,
            final_score=0.15,
            direction="neutral",
            direction_confidence=0.0,
            affected_assets=[],
            affected_regions=[],
            rationale="Low relevance",
        ),
    ]


@pytest.fixture
def sample_feed_json(tmp_path, sample_feed_items):
    """Create a temporary feed JSON file."""
    feed_path = tmp_path / "test_feed.json"
    
    data = {
        "events": [item.dict() for item in sample_feed_items],
        "last_update": datetime.now(pytz.UTC).isoformat(),
    }
    
    # Handle datetime serialization
    def serialize(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return obj
    
    with open(feed_path, 'w') as f:
        json.dump(data, f, default=serialize)
    
    return str(feed_path)


# ============================================================
# SCHEMA TESTS
# ============================================================

class TestSchema:
    
    def test_feed_item_creation(self, sample_feed_items):
        """Test FeedItem can be created."""
        item = sample_feed_items[0]
        assert item.event_id == "test1"
        assert item.impact_score == 0.75
        assert item.category == "central_bank"
    
    def test_category_mapping(self):
        """Test feed category maps to digest category."""
        assert CATEGORY_MAPPING["central_bank"] == DigestCategory.CENTRAL_BANKS
        assert CATEGORY_MAPPING["energy_commodity"] == DigestCategory.ENERGY_COMMODITIES
        assert CATEGORY_MAPPING["geopolitical"] == DigestCategory.GEOPOLITICS
        assert CATEGORY_MAPPING["irrelevant"] is None
    
    def test_feed_item_digest_category(self, sample_feed_items):
        """Test FeedItem.get_digest_category()."""
        item = sample_feed_items[0]
        assert item.get_digest_category() == DigestCategory.CENTRAL_BANKS
    
    def test_feed_item_source_tier(self, sample_feed_items):
        """Test FeedItem.get_source_tier()."""
        item = sample_feed_items[0]  # reuters
        assert item.get_source_tier() == 1
        
        item = sample_feed_items[2]  # guardian_world
        assert item.get_source_tier() == 3


# ============================================================
# SELECTOR TESTS
# ============================================================

class TestSelector:
    
    def test_select_basic(self, sample_feed_items):
        """Test basic selection."""
        results = select_items(sample_feed_items, items_per_category=2)
        
        assert DigestCategory.CENTRAL_BANKS in results
        assert DigestCategory.ENERGY_COMMODITIES in results
        assert DigestCategory.GEOPOLITICS in results
    
    def test_select_respects_threshold(self, sample_feed_items):
        """Test selection respects impact threshold."""
        # High threshold should exclude low-impact items
        results = select_items(
            sample_feed_items, 
            items_per_category=5, 
            impact_threshold=0.7
        )
        
        # Only the Fed item should pass
        central_bank = results.get(DigestCategory.CENTRAL_BANKS)
        assert central_bank is not None
        assert len(central_bank.selected) == 1
        assert central_bank.selected[0].impact_score >= 0.7
    
    def test_select_filters_irrelevant(self, sample_feed_items):
        """Test irrelevant category is not selected."""
        results = select_items(sample_feed_items)
        
        # There should be no "irrelevant" category in results
        for category in results.keys():
            assert category is not None
    
    def test_select_ranks_by_impact(self, sample_feed_items):
        """Test items are ranked by impact score."""
        results = select_items(sample_feed_items, items_per_category=5)
        
        for result in results.values():
            if len(result.selected) > 1:
                scores = [item.impact_score for item in result.selected]
                assert scores == sorted(scores, reverse=True)


# ============================================================
# PARSER TESTS
# ============================================================

class TestParser:
    
    def test_parse_category_summary_valid(self):
        """Test parsing valid LLM output."""
        llm_output = """
WHAT_HAPPENED:
- Fed signaled potential rate cut [Reuters]
- Markets rallied on dovish comments [Bloomberg]

WHY_IT_MATTERS:
- Rate cut could boost equities [Reuters]

MARKET_IMPACT_US:
- UST 10Y yields may fall 10-15 bps [Reuters]
- SPX could rally 1-2% [Bloomberg]

MARKET_IMPACT_GCC:
- Lower US rates may ease GCC funding costs via USD peg [Reuters]
- Brent could stabilize [Bloomberg]

CONFIDENCE: High

WATCHLIST:
- Monitor upcoming CPI data [Reuters]
- Watch Fed speakers this week [Bloomberg]
"""
        
        summary, error = SummaryParser.parse_category_summary(llm_output)
        
        assert error is None
        assert summary is not None
        assert len(summary.what_happened) >= 1
        assert len(summary.market_impact_us.bullets) >= 1
        assert len(summary.market_impact_gcc.bullets) >= 1
        assert summary.confidence == "High"
    
    def test_parse_category_summary_invalid(self):
        """Test parsing invalid LLM output returns error."""
        llm_output = "This is not a valid summary format."
        
        summary, error = SummaryParser.parse_category_summary(llm_output)
        
        assert summary is None
        assert error is not None
        assert "Missing" in error
    
    def test_parse_executive_brief_valid(self):
        """Test parsing valid executive brief."""
        llm_output = """
TOP_TAKEAWAYS:
- Fed signals dovish pivot [Reuters]
- Oil prices surge on OPEC cuts [Bloomberg]
- Geopolitical risks remain elevated [Guardian]

TODAYS_THEMES:
- Monetary Policy
- Energy Supply
- Geopolitics

RISK_TONE: Risk-Off
"""
        
        brief, error = SummaryParser.parse_executive_brief(llm_output)
        
        assert error is None
        assert brief is not None
        assert len(brief.top_takeaways) >= 3
        assert len(brief.todays_themes) >= 3
        assert brief.risk_tone == "Risk-Off"


# ============================================================
# RENDERER TESTS
# ============================================================

class TestRenderer:
    
    def test_render_produces_html(self, sample_feed_items):
        """Test renderer produces valid HTML."""
        from src.digest.schema import (
            DigestOutput, DigestMetadata, ExecutiveBrief, DigestSection
        )
        
        digest = DigestOutput(
            metadata=DigestMetadata(
                date="2026-01-27",
                total_items_processed=10,
                total_items_selected=3,
                categories_with_content=2,
            ),
            executive_brief=ExecutiveBrief(
                top_takeaways=["Test takeaway [Source]"],
                todays_themes=["Test Theme"],
                risk_tone="Neutral",
            ),
            sections=[],
        )
        
        renderer = DigestRenderer()
        html = renderer.render(digest)
        
        assert html is not None
        assert "<!DOCTYPE html>" in html
        assert "Daily Intelligence Digest" in html
        assert "2026-01-27" in html


# ============================================================
# FEED ADAPTER TESTS
# ============================================================

class TestFeedAdapter:
    
    def test_load_json_feed(self, sample_feed_json):
        """Test loading JSON feed file."""
        adapter = FeedAdapter(sample_feed_json)
        items = adapter.load()
        
        assert len(items) >= 1
        assert items[0].event_id == "test1"
    
    def test_load_missing_file(self):
        """Test loading missing file returns empty list."""
        adapter = FeedAdapter("nonexistent.json")
        items = adapter.load()
        
        assert items == []


# ============================================================
# ENGINE TESTS
# ============================================================

class TestEngine:
    
    def test_generate_digest_basic(self, sample_feed_json, tmp_path):
        """Test basic digest generation."""
        result = generate_daily_digest(
            feed_path=sample_feed_json,
            outdir=str(tmp_path / "digest"),
            skip_llm=True,  # Use fallbacks
            include_pdf=False,  # Skip PDF for faster test
        )
        
        assert result['success'] is True
        assert result['html_path'] is not None
        assert Path(result['html_path']).exists()
        assert result['json_path'] is not None
        assert Path(result['json_path']).exists()
    
    def test_generate_digest_empty_feed_fails(self, tmp_path):
        """Test empty feed raises error."""
        # Create empty feed
        empty_feed = tmp_path / "empty.json"
        with open(empty_feed, 'w') as f:
            json.dump({"events": []}, f)
        
        result = generate_daily_digest(
            feed_path=str(empty_feed),
            outdir=str(tmp_path / "digest"),
            skip_llm=True,
        )
        
        assert result['success'] is False
        assert "No items" in result['error']


# ============================================================
# INTEGRATION TEST
# ============================================================

class TestIntegration:
    
    def test_full_pipeline(self, sample_feed_json, tmp_path):
        """Test full digest generation pipeline."""
        outdir = tmp_path / "full_test"
        
        result = generate_daily_digest(
            date="2026-01-27",
            feed_path=sample_feed_json,
            outdir=str(outdir),
            items_per_category=2,
            impact_threshold=0.3,
            skip_llm=True,
            include_pdf=False,
        )
        
        assert result['success'] is True
        
        # Check files exist
        assert Path(result['html_path']).exists()
        assert Path(result['json_path']).exists()
        
        # Verify JSON content
        with open(result['json_path']) as f:
            data = json.load(f)
        
        assert 'metadata' in data
        assert 'executive_brief' in data
        assert 'sections' in data
        assert data['metadata']['date'] == "2026-01-27"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
