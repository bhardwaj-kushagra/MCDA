"""Unit tests for Node-level MCDA."""

import unittest
from datetime import datetime

from firenet.node.mcda import NodeMCDA, NodeMCDAConfig, SensorReading


class TestNodeMCDA(unittest.TestCase):
    """Test cases for NodeMCDA."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mcda = NodeMCDA(node_id="test_node")
    
    def test_initialization(self):
        """Test NodeMCDA initializes correctly."""
        self.assertIsNotNone(self.mcda)
        self.assertEqual(self.mcda.node_id, "test_node")
    
    def test_weight_validation(self):
        """Test that invalid weights raise error."""
        invalid_config = NodeMCDAConfig(
            weight_temperature=0.5,
            weight_gas=0.5,
            weight_infrared=0.5,  # Sum > 1.0
            weight_humidity=0.5,
            weight_lux=0.5
        )
        with self.assertRaises(ValueError):
            NodeMCDA(config=invalid_config)
    
    def test_low_risk_reading(self):
        """Test risk calculation for low-risk conditions."""
        reading = SensorReading(
            temperature=20.0,   # Low temp
            humidity=80.0,      # High humidity
            gas=50.0,           # Low gas
            infrared=0.1,       # Low IR
            lux=1000.0,
            timestamp=datetime.now(),
            node_id="test_node"
        )
        
        result = self.mcda.compute_risk(reading)
        
        self.assertTrue(result["gate_passed"])
        self.assertLess(result["risk_score"], 0.3)
        self.assertIn("risk_score", result)
        self.assertIn("confidence", result)
        self.assertIn("normalized_inputs", result)
    
    def test_high_risk_reading(self):
        """Test risk calculation for high-risk conditions."""
        reading = SensorReading(
            temperature=55.0,   # High temp
            humidity=20.0,      # Low humidity
            gas=800.0,          # High gas
            infrared=0.9,       # High IR
            lux=80000.0,
            timestamp=datetime.now(),
            node_id="test_node"
        )
        
        result = self.mcda.compute_risk(reading)
        
        self.assertTrue(result["gate_passed"])
        self.assertGreater(result["risk_score"], 0.7)
    
    def test_gate_blocks_low_temperature(self):
        """Test that low temperature triggers gate block."""
        reading = SensorReading(
            temperature=5.0,    # Below threshold
            humidity=30.0,
            gas=500.0,
            infrared=0.8,
            lux=1000.0,
            timestamp=datetime.now(),
            node_id="test_node"
        )
        
        result = self.mcda.compute_risk(reading)
        
        self.assertFalse(result["gate_passed"])
        self.assertEqual(result["gate_failure_reason"], "temperature_below_threshold")
        self.assertEqual(result["risk_score"], 0.0)
    
    def test_gate_blocks_high_humidity(self):
        """Test that high humidity triggers gate block."""
        reading = SensorReading(
            temperature=40.0,
            humidity=98.0,      # Above threshold
            gas=500.0,
            infrared=0.8,
            lux=1000.0,
            timestamp=datetime.now(),
            node_id="test_node"
        )
        
        result = self.mcda.compute_risk(reading)
        
        self.assertFalse(result["gate_passed"])
        self.assertEqual(result["gate_failure_reason"], "humidity_above_threshold")
    
    def test_normalization(self):
        """Test that readings are properly normalized."""
        reading = SensorReading(
            temperature=35.0,   # Mid-range
            humidity=50.0,      # Mid-range
            gas=500.0,          # Mid-range
            infrared=0.5,
            lux=50000.0,
            timestamp=datetime.now(),
            node_id="test_node"
        )
        
        result = self.mcda.compute_risk(reading)
        
        # All normalized values should be roughly around 0.5
        normalized = result["normalized_inputs"]
        for key, value in normalized.items():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
    
    def test_weights_applied_correctly(self):
        """Test that weights are applied as configured."""
        result = self.mcda.compute_risk(SensorReading(
            temperature=35.0,
            humidity=50.0,
            gas=500.0,
            infrared=0.5,
            lux=50000.0
        ))
        
        weights = result["weights"]
        self.assertAlmostEqual(weights["temperature"], 0.30)
        self.assertAlmostEqual(weights["gas"], 0.25)
        self.assertAlmostEqual(weights["infrared"], 0.20)
        self.assertAlmostEqual(weights["humidity"], 0.15)
        self.assertAlmostEqual(weights["lux"], 0.10)
    
    def test_output_structure(self):
        """Test that output has all required fields."""
        reading = SensorReading(
            temperature=30.0,
            humidity=50.0,
            gas=200.0,
            infrared=0.3,
            lux=5000.0
        )
        
        result = self.mcda.compute_risk(reading)
        
        required_fields = [
            "risk_score", "confidence", "gate_passed",
            "gate_failure_reason", "timestamp", "node_id",
            "normalized_inputs", "weights"
        ]
        for field in required_fields:
            self.assertIn(field, result)


class TestNodeMCDAConfig(unittest.TestCase):
    """Test cases for NodeMCDAConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = NodeMCDAConfig()
        
        total_weight = (
            config.weight_temperature +
            config.weight_gas +
            config.weight_infrared +
            config.weight_humidity +
            config.weight_lux
        )
        self.assertAlmostEqual(total_weight, 1.0)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = NodeMCDAConfig(
            weight_temperature=0.4,
            weight_gas=0.3,
            weight_infrared=0.15,
            weight_humidity=0.1,
            weight_lux=0.05
        )
        
        mcda = NodeMCDA(config=config)
        self.assertAlmostEqual(mcda.config.weight_temperature, 0.4)


if __name__ == "__main__":
    unittest.main()
