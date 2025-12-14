"""Unit tests for Edge-level Fuzzy MCDA."""

import unittest
from datetime import datetime, timedelta

from firenet.edge.fuzzy_mcda import (
    EdgeFuzzyMCDA,
    EdgeMCDAConfig,
    NodeReport,
    WeatherContext,
    FuzzyLevel
)


class TestEdgeFuzzyMCDA(unittest.TestCase):
    """Test cases for EdgeFuzzyMCDA."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mcda = EdgeFuzzyMCDA(edge_id="test_edge")
    
    def test_initialization(self):
        """Test EdgeFuzzyMCDA initializes correctly."""
        self.assertIsNotNone(self.mcda)
        self.assertEqual(self.mcda.edge_id, "test_edge")
    
    def test_insufficient_nodes(self):
        """Test handling of insufficient nodes."""
        result = self.mcda.evaluate_event([])
        
        self.assertEqual(result["fire_confidence"], 0.0)
        self.assertFalse(result["is_validated"])
        self.assertEqual(result["reason"], "insufficient_nodes")
    
    def test_low_risk_evaluation(self):
        """Test evaluation with low-risk node reports."""
        reports = [
            NodeReport(
                node_id="node_1",
                risk_score=0.2,
                confidence=0.8,
                timestamp=datetime.now()
            ),
            NodeReport(
                node_id="node_2",
                risk_score=0.15,
                confidence=0.85,
                timestamp=datetime.now()
            )
        ]
        
        result = self.mcda.evaluate_event(reports)
        
        self.assertLess(result["fire_confidence"], 0.5)
        self.assertEqual(result["classification"], "low")
    
    def test_high_risk_evaluation(self):
        """Test evaluation with high-risk node reports."""
        now = datetime.now()
        reports = [
            NodeReport(
                node_id="node_1",
                risk_score=0.85,
                confidence=0.9,
                timestamp=now
            ),
            NodeReport(
                node_id="node_2",
                risk_score=0.9,
                confidence=0.92,
                timestamp=now - timedelta(seconds=60)
            ),
            NodeReport(
                node_id="node_3",
                risk_score=0.88,
                confidence=0.88,
                timestamp=now - timedelta(seconds=30)
            )
        ]
        
        weather = WeatherContext(wind_speed=25.0)
        result = self.mcda.evaluate_event(reports, weather)
        
        self.assertGreater(result["fire_confidence"], 0.6)
        self.assertTrue(result["is_validated"])
    
    def test_fuzzy_risk_membership(self):
        """Test fuzzy membership calculation for risk."""
        # Test low risk
        low_memberships = self.mcda._fuzzify_risk(0.1)
        self.assertGreater(low_memberships[FuzzyLevel.LOW], 0.5)
        
        # Test high risk
        high_memberships = self.mcda._fuzzify_risk(0.9)
        self.assertGreater(high_memberships[FuzzyLevel.HIGH], 0.5)
    
    def test_fuzzy_agreement_membership(self):
        """Test fuzzy membership calculation for agreement."""
        # Low agreement
        low_agreement = self.mcda._fuzzify_agreement(0.1)
        self.assertGreater(low_agreement[FuzzyLevel.LOW], 0.5)
        
        # High agreement
        high_agreement = self.mcda._fuzzify_agreement(0.9)
        self.assertGreater(high_agreement[FuzzyLevel.HIGH], 0.5)
    
    def test_fuzzy_wind_membership(self):
        """Test fuzzy membership calculation for wind."""
        # Low wind
        low_wind = self.mcda._fuzzify_wind(2.0)
        self.assertGreater(low_wind[FuzzyLevel.LOW], 0.5)
        
        # High wind
        high_wind = self.mcda._fuzzify_wind(25.0)
        self.assertGreater(high_wind[FuzzyLevel.HIGH], 0.5)
    
    def test_fuzzy_rules_very_high(self):
        """Test fuzzy rule for very high confidence."""
        risk_fuzzy = {FuzzyLevel.LOW: 0.0, FuzzyLevel.MEDIUM: 0.0, FuzzyLevel.HIGH: 1.0}
        agreement_fuzzy = {FuzzyLevel.LOW: 0.0, FuzzyLevel.MEDIUM: 0.0, FuzzyLevel.HIGH: 1.0}
        wind_fuzzy = {FuzzyLevel.LOW: 0.0, FuzzyLevel.MEDIUM: 0.0, FuzzyLevel.HIGH: 1.0}
        
        output = self.mcda._apply_fuzzy_rules(
            risk_fuzzy, agreement_fuzzy, wind_fuzzy
        )
        
        self.assertEqual(output[FuzzyLevel.VERY_HIGH], 1.0)
    
    def test_weather_context_integration(self):
        """Test that weather context affects evaluation."""
        reports = [
            NodeReport(
                node_id="node_1",
                risk_score=0.6,
                confidence=0.8,
                timestamp=datetime.now()
            )
        ]
        
        # Low wind
        low_wind_weather = WeatherContext(wind_speed=2.0)
        low_wind_result = self.mcda.evaluate_event(reports, low_wind_weather)
        
        # High wind
        high_wind_weather = WeatherContext(wind_speed=25.0)
        high_wind_result = self.mcda.evaluate_event(reports, high_wind_weather)
        
        # Results may differ based on wind
        self.assertIn("weather_context", low_wind_result)
        self.assertIn("weather_context", high_wind_result)
    
    def test_temporal_persistence(self):
        """Test temporal persistence checking."""
        now = datetime.now()
        
        # Reports spanning sufficient time
        persistent_reports = [
            NodeReport("n1", 0.8, 0.9, now),
            NodeReport("n2", 0.85, 0.88, now - timedelta(seconds=60))
        ]
        
        result = self.mcda.evaluate_event(persistent_reports)
        self.assertTrue(result["temporal_persistence"])
    
    def test_output_structure(self):
        """Test that output has all required fields."""
        reports = [
            NodeReport("node_1", 0.5, 0.8, datetime.now())
        ]
        
        result = self.mcda.evaluate_event(reports)
        
        required_fields = [
            "fire_confidence", "classification", "is_validated",
            "node_agreement_ratio", "average_risk_score",
            "temporal_persistence", "fuzzy_inputs", "fuzzy_output",
            "timestamp", "edge_id", "node_count"
        ]
        for field in required_fields:
            self.assertIn(field, result)


class TestEdgeMCDAConfig(unittest.TestCase):
    """Test cases for EdgeMCDAConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EdgeMCDAConfig()
        
        self.assertEqual(config.risk_low_threshold, 0.3)
        self.assertEqual(config.risk_medium_threshold, 0.5)
        self.assertEqual(config.risk_high_threshold, 0.7)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = EdgeMCDAConfig(
            risk_high_threshold=0.8,
            min_nodes_for_assessment=3
        )
        
        mcda = EdgeFuzzyMCDA(config=config)
        self.assertEqual(mcda.config.risk_high_threshold, 0.8)
        self.assertEqual(mcda.config.min_nodes_for_assessment, 3)


if __name__ == "__main__":
    unittest.main()
