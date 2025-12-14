"""Unit tests for Cloud-level Policy MCDA."""

import unittest
from datetime import datetime

from firenet.cloud.policy_mcda import (
    CloudMCDA,
    CloudMCDAConfig,
    EdgeEvent,
    WeatherForecast,
    SeasonalRisk,
    SatelliteAlert,
    AHPWeightCalculator,
    SeverityLevel,
    ResponseUrgency
)


class TestCloudMCDA(unittest.TestCase):
    """Test cases for CloudMCDA."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mcda = CloudMCDA()
    
    def test_initialization(self):
        """Test CloudMCDA initializes correctly."""
        self.assertIsNotNone(self.mcda)
    
    def test_weight_validation(self):
        """Test that invalid weights raise error."""
        invalid_config = CloudMCDAConfig(
            weight_edge_confidence=0.5,
            weight_weather_severity=0.5,
            weight_seasonal_risk=0.5,
            weight_satellite_confirmation=0.5
        )
        with self.assertRaises(ValueError):
            CloudMCDA(config=invalid_config)
    
    def test_low_priority_event(self):
        """Test evaluation of low priority event."""
        event = EdgeEvent(
            edge_id="edge_1",
            fire_confidence=0.2,
            classification="low",
            location=(40.0, -74.0),
            timestamp=datetime.now(),
            node_count=2,
            average_risk_score=0.15
        )
        
        result = self.mcda.evaluate_event(event)
        
        self.assertLess(result["priority_score"], 0.4)
        self.assertIn(result["severity"], ["minimal", "low"])
        self.assertIn(result["urgency"], ["monitor", "standby"])
    
    def test_high_priority_event(self):
        """Test evaluation of high priority event."""
        event = EdgeEvent(
            edge_id="edge_1",
            fire_confidence=0.95,
            classification="very_high",
            location=(40.0, -74.0),
            timestamp=datetime.now(),
            node_count=5,
            average_risk_score=0.9
        )
        
        weather = WeatherForecast(
            temperature=45.0,
            humidity=15.0,
            wind_speed=20.0,
            precipitation=0.0,
            drought_index=0.9
        )
        
        seasonal = SeasonalRisk(
            base_risk=0.8,
            vegetation_dryness=0.85,
            historical_frequency=0.7
        )
        
        satellite = SatelliteAlert(confidence=0.9)
        
        result = self.mcda.evaluate_event(event, weather, seasonal, satellite)
        
        self.assertGreater(result["priority_score"], 0.7)
        self.assertIn(result["severity"], ["high", "critical"])
        self.assertIn(result["urgency"], ["urgent", "emergency"])
    
    def test_weather_severity_calculation(self):
        """Test weather severity calculation."""
        # Severe weather
        severe_weather = WeatherForecast(
            temperature=50.0,
            humidity=10.0,
            wind_speed=25.0,
            precipitation=0.0,
            drought_index=0.9
        )
        severity = self.mcda._calculate_weather_severity(severe_weather)
        self.assertGreater(severity, 0.7)
        
        # Mild weather
        mild_weather = WeatherForecast(
            temperature=20.0,
            humidity=80.0,
            wind_speed=2.0,
            precipitation=5.0,
            drought_index=0.1
        )
        severity = self.mcda._calculate_weather_severity(mild_weather)
        self.assertLess(severity, 0.3)
    
    def test_seasonal_score_calculation(self):
        """Test seasonal risk score calculation."""
        high_risk = SeasonalRisk(
            base_risk=0.9,
            vegetation_dryness=0.95,
            historical_frequency=0.8
        )
        score = self.mcda._calculate_seasonal_score(high_risk)
        self.assertGreater(score, 0.8)
        
        low_risk = SeasonalRisk(
            base_risk=0.1,
            vegetation_dryness=0.1,
            historical_frequency=0.1
        )
        score = self.mcda._calculate_seasonal_score(low_risk)
        self.assertLess(score, 0.2)
    
    def test_severity_classification(self):
        """Test severity classification thresholds."""
        self.assertEqual(
            self.mcda._classify_severity(0.1),
            SeverityLevel.MINIMAL
        )
        self.assertEqual(
            self.mcda._classify_severity(0.3),
            SeverityLevel.LOW
        )
        self.assertEqual(
            self.mcda._classify_severity(0.5),
            SeverityLevel.MODERATE
        )
        self.assertEqual(
            self.mcda._classify_severity(0.7),
            SeverityLevel.HIGH
        )
        self.assertEqual(
            self.mcda._classify_severity(0.9),
            SeverityLevel.CRITICAL
        )
    
    def test_urgency_determination(self):
        """Test response urgency determination."""
        self.assertEqual(
            self.mcda._determine_urgency(0.1),
            ResponseUrgency.MONITOR
        )
        self.assertEqual(
            self.mcda._determine_urgency(0.3),
            ResponseUrgency.STANDBY
        )
        self.assertEqual(
            self.mcda._determine_urgency(0.5),
            ResponseUrgency.DEPLOY
        )
        self.assertEqual(
            self.mcda._determine_urgency(0.7),
            ResponseUrgency.URGENT
        )
        self.assertEqual(
            self.mcda._determine_urgency(0.9),
            ResponseUrgency.EMERGENCY
        )
    
    def test_event_ranking(self):
        """Test event ranking functionality."""
        events = [
            EdgeEvent("e1", 0.3, "low", (40.0, -74.0), datetime.now(), 2, 0.25),
            EdgeEvent("e2", 0.8, "high", (41.0, -74.0), datetime.now(), 5, 0.75),
            EdgeEvent("e3", 0.5, "medium", (40.5, -74.0), datetime.now(), 3, 0.45)
        ]
        
        evaluations = [self.mcda.evaluate_event(e) for e in events]
        ranked = self.mcda.rank_events(evaluations)
        
        # Highest priority should be first
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertGreater(
            ranked[0]["priority_score"],
            ranked[1]["priority_score"]
        )
        self.assertGreater(
            ranked[1]["priority_score"],
            ranked[2]["priority_score"]
        )
    
    def test_alert_summary_generation(self):
        """Test alert summary generation."""
        events = [
            EdgeEvent("e1", 0.9, "very_high", (40.0, -74.0), datetime.now(), 5, 0.85),
            EdgeEvent("e2", 0.5, "medium", (41.0, -74.0), datetime.now(), 3, 0.45)
        ]
        
        evaluations = [self.mcda.evaluate_event(e) for e in events]
        ranked = self.mcda.rank_events(evaluations)
        summary = self.mcda.generate_alert_summary(ranked)
        
        self.assertEqual(summary["total_events"], 2)
        self.assertIn("severity_distribution", summary)
        self.assertIn("urgency_distribution", summary)
        self.assertIn("statistics", summary)
        self.assertIn("recommendations", summary)
    
    def test_empty_alert_summary(self):
        """Test alert summary with no events."""
        summary = self.mcda.generate_alert_summary([])
        
        self.assertEqual(summary["total_events"], 0)
        self.assertIn("No active fire events", summary["summary"])
    
    def test_output_structure(self):
        """Test that output has all required fields."""
        event = EdgeEvent(
            edge_id="edge_1",
            fire_confidence=0.6,
            classification="medium",
            location=(40.0, -74.0),
            timestamp=datetime.now(),
            node_count=3,
            average_risk_score=0.55
        )
        
        result = self.mcda.evaluate_event(event)
        
        required_fields = [
            "priority_score", "severity", "urgency", "advisory",
            "component_scores", "weights", "event", "timestamp",
            "event_id"
        ]
        for field in required_fields:
            self.assertIn(field, result)


class TestAHPWeightCalculator(unittest.TestCase):
    """Test cases for AHP weight calculation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ahp = AHPWeightCalculator([
            "edge_confidence",
            "weather_severity",
            "seasonal_risk",
            "satellite_confirmation"
        ])
    
    def test_consistent_matrix(self):
        """Test weight calculation with consistent matrix."""
        # Perfectly consistent matrix
        matrix = [
            [1.0, 2.0, 3.0, 4.0],
            [0.5, 1.0, 1.5, 2.0],
            [0.333, 0.667, 1.0, 1.333],
            [0.25, 0.5, 0.75, 1.0]
        ]
        
        weights, cr, is_consistent = self.ahp.calculate_weights(matrix)
        
        self.assertTrue(is_consistent)
        self.assertLess(cr, 0.1)
        self.assertEqual(len(weights), 4)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=2)
    
    def test_default_matrix(self):
        """Test default matrix creation."""
        matrix = self.ahp.create_default_matrix()
        
        self.assertEqual(len(matrix), 4)
        self.assertEqual(len(matrix[0]), 4)
        
        # Diagonal should be 1
        for i in range(4):
            self.assertEqual(matrix[i][i], 1.0)
    
    def test_ahp_weight_calibration(self):
        """Test AHP weight calibration in CloudMCDA."""
        mcda = CloudMCDA()
        
        result = mcda.calibrate_weights_ahp()
        
        self.assertIn("weights", result)
        self.assertIn("consistency_ratio", result)
        self.assertIn("is_consistent", result)


if __name__ == "__main__":
    unittest.main()
