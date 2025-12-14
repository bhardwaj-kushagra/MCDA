"""
Node-Level MCDA for FIRENET.

Implements lightweight fire risk estimation from sensor data on IoT nodes.
Each node performs early fire risk estimation and false-positive suppression 
before transmitting data to the edge layer.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from firenet.utils.normalization import normalize, inverse_normalize
from firenet.utils.logging import MCDALogger


@dataclass
class SensorReading:
    """Container for sensor readings from a node."""
    temperature: float  # Celsius
    humidity: float     # Percentage (0-100)
    gas: float          # Gas concentration (ppm)
    infrared: float     # IR intensity (0-1 normalized)
    lux: float          # Ambient light level
    timestamp: Optional[datetime] = None
    node_id: Optional[str] = None


@dataclass
class NodeMCDAConfig:
    """Configuration for Node-level MCDA."""
    # Default weights as specified in the system design
    weight_temperature: float = 0.30
    weight_gas: float = 0.25
    weight_infrared: float = 0.20
    weight_humidity: float = 0.15
    weight_lux: float = 0.10
    
    # Normalization ranges
    temp_min: float = -10.0   # Celsius
    temp_max: float = 60.0    # Celsius
    humidity_min: float = 0.0
    humidity_max: float = 100.0
    gas_min: float = 0.0
    gas_max: float = 1000.0   # ppm
    lux_min: float = 0.0
    lux_max: float = 100000.0
    
    # Rule gates (sanity checks)
    min_temp_for_fire: float = 15.0  # Minimum temp for fire possibility
    max_humidity_for_fire: float = 95.0  # Max humidity allowing fire


class NodeMCDA:
    """
    Node-level MCDA processor for fire risk estimation.
    
    Performs:
    - Periodic sensor sampling normalization
    - Rule-based sanity gating
    - Weighted MCDA risk scoring
    - Confidence estimation
    """
    
    def __init__(
        self,
        config: Optional[NodeMCDAConfig] = None,
        node_id: Optional[str] = None
    ):
        """
        Initialize the Node MCDA processor.
        
        Args:
            config: Configuration for MCDA processing
            node_id: Unique identifier for this node
        """
        self.config = config or NodeMCDAConfig()
        self.node_id = node_id
        self.logger = MCDALogger(f"NodeMCDA-{node_id or 'unknown'}")
        
        # Validate weights sum to 1.0
        total_weight = (
            self.config.weight_temperature +
            self.config.weight_gas +
            self.config.weight_infrared +
            self.config.weight_humidity +
            self.config.weight_lux
        )
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    def _normalize_readings(self, reading: SensorReading) -> Dict[str, float]:
        """
        Normalize sensor readings to 0-1 range.
        
        Args:
            reading: Raw sensor reading
            
        Returns:
            Dictionary of normalized values
        """
        return {
            "temperature": normalize(
                reading.temperature,
                self.config.temp_min,
                self.config.temp_max
            ),
            # Inverse normalize humidity - lower humidity = higher fire risk
            "humidity": inverse_normalize(
                reading.humidity,
                self.config.humidity_min,
                self.config.humidity_max
            ),
            "gas": normalize(
                reading.gas,
                self.config.gas_min,
                self.config.gas_max
            ),
            "infrared": reading.infrared,  # Already normalized
            "lux": normalize(
                reading.lux,
                self.config.lux_min,
                self.config.lux_max
            )
        }
    
    def _apply_rule_gates(self, reading: SensorReading) -> tuple:
        """
        Apply rule-based sanity gates to filter impossible fire conditions.
        
        Args:
            reading: Raw sensor reading
            
        Returns:
            Tuple of (passed: bool, reason: str or None)
        """
        # Gate 1: Temperature too low for fire
        if reading.temperature < self.config.min_temp_for_fire:
            self.logger.log_gate_failure(
                "node",
                "min_temperature",
                "Temperature below fire threshold",
                {"temperature": reading.temperature}
            )
            return False, "temperature_below_threshold"
        
        # Gate 2: Humidity too high for fire
        if reading.humidity > self.config.max_humidity_for_fire:
            self.logger.log_gate_failure(
                "node",
                "max_humidity",
                "Humidity above fire threshold",
                {"humidity": reading.humidity}
            )
            return False, "humidity_above_threshold"
        
        return True, None
    
    def _compute_confidence(
        self,
        normalized: Dict[str, float],
        risk_score: float
    ) -> float:
        """
        Compute confidence in the risk assessment.
        
        Confidence is based on:
        - Agreement between sensors
        - Magnitude of the risk score
        
        Args:
            normalized: Normalized sensor values
            risk_score: Computed risk score
            
        Returns:
            Confidence value (0-1)
        """
        # Calculate variance of sensor readings as a proxy for agreement
        values = list(normalized.values())
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        
        # Lower variance = higher agreement = higher confidence
        agreement_factor = 1.0 - min(variance, 1.0)
        
        # Higher risk scores get higher confidence when agreement is high
        confidence = (agreement_factor * 0.6) + (risk_score * 0.4)
        
        return min(max(confidence, 0.0), 1.0)
    
    def compute_risk(self, reading: SensorReading) -> Dict[str, Any]:
        """
        Compute fire risk score from sensor reading.
        
        This is the main MCDA computation that:
        1. Normalizes sensor readings
        2. Applies sanity gates
        3. Computes weighted risk score
        4. Estimates confidence
        
        Args:
            reading: Sensor reading from the node
            
        Returns:
            Dictionary containing:
            - risk_score: Fire risk score (0-1)
            - confidence: Confidence in the assessment (0-1)
            - gate_passed: Whether sanity gates were passed
            - gate_failure_reason: Reason if gates failed
            - timestamp: Time of assessment
            - node_id: Node identifier
            - normalized_inputs: Normalized sensor values
            - weights: Applied weights
        """
        timestamp = reading.timestamp or datetime.now(timezone.utc)
        node_id = reading.node_id or self.node_id
        
        # Apply rule gates first
        gate_passed, gate_reason = self._apply_rule_gates(reading)
        
        if not gate_passed:
            return {
                "risk_score": 0.0,
                "confidence": 1.0,  # High confidence that there's no fire
                "gate_passed": False,
                "gate_failure_reason": gate_reason,
                "timestamp": timestamp.isoformat(),
                "node_id": node_id,
                "normalized_inputs": None,
                "weights": None
            }
        
        # Normalize readings
        normalized = self._normalize_readings(reading)
        
        # Get weights
        weights = {
            "temperature": self.config.weight_temperature,
            "gas": self.config.weight_gas,
            "infrared": self.config.weight_infrared,
            "humidity": self.config.weight_humidity,
            "lux": self.config.weight_lux
        }
        
        # Compute weighted risk score
        risk_score = (
            weights["temperature"] * normalized["temperature"] +
            weights["gas"] * normalized["gas"] +
            weights["infrared"] * normalized["infrared"] +
            weights["humidity"] * normalized["humidity"] +
            weights["lux"] * normalized["lux"]
        )
        
        # Compute confidence
        confidence = self._compute_confidence(normalized, risk_score)
        
        # Log the decision for explainability
        result = {
            "risk_score": risk_score,
            "confidence": confidence,
            "gate_passed": True,
            "gate_failure_reason": None,
            "timestamp": timestamp.isoformat(),
            "node_id": node_id,
            "normalized_inputs": normalized,
            "weights": weights
        }
        
        self.logger.log_decision(
            layer="node",
            inputs=normalized,
            weights=weights,
            score=risk_score,
            confidence=confidence,
            metadata={"node_id": node_id}
        )
        
        return result
