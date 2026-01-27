"""
ML Integration for Daily Digest.

Feeds digest signals into the learning system to make strategies smarter.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class DigestMLIntegration:
    """
    Integrates Daily Digest signals into the learning system.
    
    This class:
    1. Extracts actionable signals from digest
    2. Saves them for learning engine consumption
    3. Provides signals to strategies for smarter decisions
    """
    
    def __init__(self, outputs_dir: str = "outputs"):
        self.outputs_dir = Path(outputs_dir)
        self.signals_file = self.outputs_dir / "digest_signals.json"
        self.history_file = self.outputs_dir / "digest_history.json"
        self.signals_cache: Dict[str, Any] = {}
        self._load_signals()
    
    def _load_signals(self):
        """Load cached signals from disk."""
        try:
            if self.signals_file.exists():
                with open(self.signals_file, 'r') as f:
                    self.signals_cache = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load digest signals: {e}")
            self.signals_cache = {}
    
    def _save_signals(self):
        """Save signals to disk."""
        try:
            self.outputs_dir.mkdir(parents=True, exist_ok=True)
            with open(self.signals_file, 'w') as f:
                json.dump(self.signals_cache, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not save digest signals: {e}")
    
    def extract_signals_from_digest(self, digest_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract ML-ready signals from a digest output.
        
        Args:
            digest_output: The full digest output dict
            
        Returns:
            Dict of signals for the learning system
        """
        signals = {
            "timestamp": datetime.now().isoformat(),
            "date": digest_output.get("metadata", {}).get("date"),
            
            # From executive brief
            "risk_tone": None,
            "overall_bias": "NEUTRAL",
            "conviction_score": 5,
            "volatility_expectation": "MEDIUM",
            
            # Posture recommendations
            "equity_posture": "Neutral",
            "cash_posture": "Neutral",
            "risk_posture": "Neutral",
            
            # Category-level signals
            "category_signals": {},
            
            # What to watch
            "watchlist": [],
            "risk_events": [],
            
            # Noise vs signal
            "act_on": [],
            "ignore": [],
            "wait_for": [],
            
            # Key levels
            "key_levels": [],
            
            # Sector tilts
            "sector_tilts": {},
        }
        
        # Extract from executive brief
        exec_brief = digest_output.get("executive_brief", {})
        if exec_brief:
            signals["risk_tone"] = exec_brief.get("risk_tone")
            
            # Strategy signals
            strategy_signals = exec_brief.get("strategy_signals", {})
            if strategy_signals:
                signals["overall_bias"] = strategy_signals.get("overall_bias", "NEUTRAL")
                signals["conviction_score"] = strategy_signals.get("conviction_score", 5)
                signals["volatility_expectation"] = strategy_signals.get("volatility_expectation", "MEDIUM")
                signals["sector_tilts"] = strategy_signals.get("sector_tilts", {})
                signals["risk_events"] = strategy_signals.get("risk_events_next_24h", [])
            
            # Recommended posture
            posture = exec_brief.get("recommended_posture", {})
            if posture:
                signals["equity_posture"] = posture.get("equity_exposure", "Neutral")
                signals["cash_posture"] = posture.get("cash_position", "Neutral")
                signals["risk_posture"] = posture.get("risk_assets", "Neutral")
            
            # Noise vs signal
            nvs = exec_brief.get("noise_vs_signal", {})
            if nvs:
                signals["act_on"] = nvs.get("signal_act_on", [])
                signals["ignore"] = nvs.get("noise_ignore", [])
                signals["wait_for"] = nvs.get("watch_wait_for", [])
            
            # Key levels
            signals["key_levels"] = exec_brief.get("key_levels", [])
        
        # Extract category-level signals
        for section in digest_output.get("sections", []):
            category = section.get("category")
            summary = section.get("summary", {})
            if summary:
                trading_signal = summary.get("trading_signal", {})
                signals["category_signals"][category] = {
                    "confidence": summary.get("confidence", "Medium"),
                    "direction": trading_signal.get("direction") if trading_signal else None,
                    "conviction": trading_signal.get("conviction") if trading_signal else None,
                    "affected_assets": trading_signal.get("affected_assets", []) if trading_signal else [],
                    "watchlist": summary.get("watchlist", []),
                }
                
                # Aggregate watchlist
                signals["watchlist"].extend(summary.get("watchlist", []))
        
        return signals
    
    def save_digest_signals(self, digest_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and save signals from a digest for the learning system.
        
        Args:
            digest_output: The full digest output dict
            
        Returns:
            The extracted signals
        """
        signals = self.extract_signals_from_digest(digest_output)
        
        # Update cache with latest signals
        self.signals_cache = {
            "current": signals,
            "last_updated": datetime.now().isoformat(),
        }
        self._save_signals()
        
        # Also append to history
        self._append_to_history(signals)
        
        logger.info(f"Saved digest signals: bias={signals['overall_bias']}, "
                    f"conviction={signals['conviction_score']}, "
                    f"risk_tone={signals['risk_tone']}")
        
        return signals
    
    def _append_to_history(self, signals: Dict[str, Any]):
        """Append signals to history for learning."""
        try:
            history = []
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
            
            # Keep last 30 days
            history.append(signals)
            if len(history) > 30:
                history = history[-30:]
            
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2, default=str)
                
        except Exception as e:
            logger.warning(f"Could not append to digest history: {e}")
    
    def get_current_signals(self) -> Optional[Dict[str, Any]]:
        """Get the most recent digest signals for strategy use."""
        return self.signals_cache.get("current")
    
    def get_bias_adjustment(self) -> float:
        """
        Get a bias adjustment factor for strategies.
        
        Returns:
            Float between -1 (very bearish) and +1 (very bullish)
        """
        signals = self.get_current_signals()
        if not signals:
            return 0.0
        
        # Map bias to numeric
        bias_map = {
            "BULLISH": 1.0,
            "BEARISH": -1.0,
            "NEUTRAL": 0.0,
        }
        base_bias = bias_map.get(signals.get("overall_bias", "NEUTRAL"), 0.0)
        
        # Scale by conviction (1-10 -> 0.1-1.0)
        conviction = signals.get("conviction_score", 5) / 10.0
        
        return base_bias * conviction
    
    def get_volatility_adjustment(self) -> float:
        """
        Get a volatility adjustment for position sizing.
        
        Returns:
            Float multiplier (0.5 for low vol, 1.0 for medium, 1.5 for high, 2.0 for spike)
        """
        signals = self.get_current_signals()
        if not signals:
            return 1.0
        
        vol_map = {
            "LOW": 0.5,
            "MEDIUM": 1.0,
            "HIGH": 1.5,
            "SPIKE": 2.0,
        }
        return vol_map.get(signals.get("volatility_expectation", "MEDIUM"), 1.0)
    
    def should_reduce_exposure(self) -> bool:
        """Check if digest signals suggest reducing exposure."""
        signals = self.get_current_signals()
        if not signals:
            return False
        
        # Reduce if bearish with high conviction or expecting volatility spike
        if signals.get("overall_bias") == "BEARISH" and signals.get("conviction_score", 5) >= 7:
            return True
        if signals.get("volatility_expectation") == "SPIKE":
            return True
        if signals.get("risk_tone") == "Risk-Off":
            return True
            
        return False
    
    def get_sector_tilts(self) -> Dict[str, str]:
        """Get recommended sector tilts (overweight/underweight)."""
        signals = self.get_current_signals()
        if not signals:
            return {}
        return signals.get("sector_tilts", {})
    
    def get_watchlist_items(self) -> List[str]:
        """Get items to monitor from the digest."""
        signals = self.get_current_signals()
        if not signals:
            return []
        return signals.get("watchlist", [])


# Singleton instance
_ml_integration: Optional[DigestMLIntegration] = None


def get_digest_ml_integration() -> DigestMLIntegration:
    """Get or create the ML integration singleton."""
    global _ml_integration
    if _ml_integration is None:
        _ml_integration = DigestMLIntegration()
    return _ml_integration
