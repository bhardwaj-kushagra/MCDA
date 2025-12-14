"""
Edge-Level Fuzzy MCDA for FIRENET.

Implements spatial and temporal reasoning with fuzzy logic to aggregate
and validate fire events from multiple sensor nodes.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from firenet.utils.logging import MCDALogger


class FuzzyLevel(Enum):
    """Fuzzy linguistic levels for variables."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class NodeReport:
    """Report from a sensor node."""
    node_id: str
    risk_score: float
    confidence: float
    timestamp: datetime
    latitude: Optional[float] = None
    longitude: Optional[float] = None


@dataclass
class WeatherContext:
    """Local weather context for edge processing."""
    wind_speed: float = 0.0       # m/s
    wind_direction: float = 0.0   # degrees
    dryness_index: float = 0.5    # 0-1 scale


@dataclass
class EdgeMCDAConfig:
    """Configuration for Edge-level Fuzzy MCDA."""
    # Fuzzy membership thresholds for risk
    risk_low_threshold: float = 0.3
    risk_medium_threshold: float = 0.5
    risk_high_threshold: float = 0.7
    
    # Fuzzy membership thresholds for agreement (% of nodes reporting)
    agreement_low_threshold: float = 0.3
    agreement_medium_threshold: float = 0.5
    agreement_high_threshold: float = 0.7
    
    # Fuzzy membership thresholds for wind (m/s)
    wind_low_threshold: float = 5.0
    wind_medium_threshold: float = 10.0
    wind_high_threshold: float = 20.0
    
    # Temporal persistence requirements
    min_persistence_seconds: float = 30.0
    
    # Minimum nodes for valid assessment
    min_nodes_for_assessment: int = 1
    
    # Output confidence thresholds
    confidence_output_map: Dict[FuzzyLevel, float] = field(default_factory=lambda: {
        FuzzyLevel.LOW: 0.2,
        FuzzyLevel.MEDIUM: 0.5,
        FuzzyLevel.HIGH: 0.75,
        FuzzyLevel.VERY_HIGH: 0.95
    })


class EdgeFuzzyMCDA:
    """
    Edge-level Fuzzy MCDA processor for event validation.
    
    Aggregates data from multiple sensor nodes, applies spatial and 
    temporal reasoning, and uses fuzzy logic for uncertainty handling.
    
    Key responsibilities:
    - Aggregate readings spatially
    - Check temporal persistence
    - Convert numeric values to fuzzy linguistic terms
    - Apply fuzzy inference rules
    - Classify event confidence
    """
    
    def __init__(
        self,
        config: Optional[EdgeMCDAConfig] = None,
        edge_id: Optional[str] = None
    ):
        """
        Initialize the Edge Fuzzy MCDA processor.
        
        Args:
            config: Configuration for fuzzy MCDA processing
            edge_id: Unique identifier for this edge node
        """
        self.config = config or EdgeMCDAConfig()
        self.edge_id = edge_id
        self.logger = MCDALogger(f"EdgeMCDA-{edge_id or 'unknown'}")
        
        # Store for temporal tracking
        self._event_history: Dict[str, List[NodeReport]] = {}
    
    def _fuzzify_risk(self, risk_score: float) -> Dict[FuzzyLevel, float]:
        """
        Convert risk score to fuzzy membership values.
        
        Args:
            risk_score: Numeric risk score (0-1)
            
        Returns:
            Dictionary mapping FuzzyLevel to membership degree
        """
        memberships = {}
        
        # Low membership (triangular: 0 to threshold)
        if risk_score <= self.config.risk_low_threshold:
            memberships[FuzzyLevel.LOW] = 1.0
        elif risk_score <= self.config.risk_medium_threshold:
            memberships[FuzzyLevel.LOW] = (
                (self.config.risk_medium_threshold - risk_score) /
                (self.config.risk_medium_threshold - self.config.risk_low_threshold)
            )
        else:
            memberships[FuzzyLevel.LOW] = 0.0
        
        # Medium membership (triangular)
        if risk_score <= self.config.risk_low_threshold:
            memberships[FuzzyLevel.MEDIUM] = 0.0
        elif risk_score <= self.config.risk_medium_threshold:
            memberships[FuzzyLevel.MEDIUM] = (
                (risk_score - self.config.risk_low_threshold) /
                (self.config.risk_medium_threshold - self.config.risk_low_threshold)
            )
        elif risk_score <= self.config.risk_high_threshold:
            memberships[FuzzyLevel.MEDIUM] = (
                (self.config.risk_high_threshold - risk_score) /
                (self.config.risk_high_threshold - self.config.risk_medium_threshold)
            )
        else:
            memberships[FuzzyLevel.MEDIUM] = 0.0
        
        # High membership (triangular: threshold to 1)
        if risk_score <= self.config.risk_medium_threshold:
            memberships[FuzzyLevel.HIGH] = 0.0
        elif risk_score <= self.config.risk_high_threshold:
            memberships[FuzzyLevel.HIGH] = (
                (risk_score - self.config.risk_medium_threshold) /
                (self.config.risk_high_threshold - self.config.risk_medium_threshold)
            )
        else:
            memberships[FuzzyLevel.HIGH] = 1.0
        
        return memberships
    
    def _fuzzify_agreement(
        self,
        high_risk_ratio: float
    ) -> Dict[FuzzyLevel, float]:
        """
        Convert node agreement ratio to fuzzy membership values.
        
        Args:
            high_risk_ratio: Ratio of nodes reporting high risk
            
        Returns:
            Dictionary mapping FuzzyLevel to membership degree
        """
        memberships = {}
        
        if high_risk_ratio <= self.config.agreement_low_threshold:
            memberships[FuzzyLevel.LOW] = 1.0
        elif high_risk_ratio <= self.config.agreement_medium_threshold:
            memberships[FuzzyLevel.LOW] = (
                (self.config.agreement_medium_threshold - high_risk_ratio) /
                (self.config.agreement_medium_threshold - self.config.agreement_low_threshold)
            )
        else:
            memberships[FuzzyLevel.LOW] = 0.0
        
        if high_risk_ratio <= self.config.agreement_low_threshold:
            memberships[FuzzyLevel.MEDIUM] = 0.0
        elif high_risk_ratio <= self.config.agreement_medium_threshold:
            memberships[FuzzyLevel.MEDIUM] = (
                (high_risk_ratio - self.config.agreement_low_threshold) /
                (self.config.agreement_medium_threshold - self.config.agreement_low_threshold)
            )
        elif high_risk_ratio <= self.config.agreement_high_threshold:
            memberships[FuzzyLevel.MEDIUM] = (
                (self.config.agreement_high_threshold - high_risk_ratio) /
                (self.config.agreement_high_threshold - self.config.agreement_medium_threshold)
            )
        else:
            memberships[FuzzyLevel.MEDIUM] = 0.0
        
        if high_risk_ratio <= self.config.agreement_medium_threshold:
            memberships[FuzzyLevel.HIGH] = 0.0
        elif high_risk_ratio <= self.config.agreement_high_threshold:
            memberships[FuzzyLevel.HIGH] = (
                (high_risk_ratio - self.config.agreement_medium_threshold) /
                (self.config.agreement_high_threshold - self.config.agreement_medium_threshold)
            )
        else:
            memberships[FuzzyLevel.HIGH] = 1.0
        
        return memberships
    
    def _fuzzify_wind(self, wind_speed: float) -> Dict[FuzzyLevel, float]:
        """
        Convert wind speed to fuzzy membership values.
        
        Args:
            wind_speed: Wind speed in m/s
            
        Returns:
            Dictionary mapping FuzzyLevel to membership degree
        """
        memberships = {}
        
        if wind_speed <= self.config.wind_low_threshold:
            memberships[FuzzyLevel.LOW] = 1.0
        elif wind_speed <= self.config.wind_medium_threshold:
            memberships[FuzzyLevel.LOW] = (
                (self.config.wind_medium_threshold - wind_speed) /
                (self.config.wind_medium_threshold - self.config.wind_low_threshold)
            )
        else:
            memberships[FuzzyLevel.LOW] = 0.0
        
        if wind_speed <= self.config.wind_low_threshold:
            memberships[FuzzyLevel.MEDIUM] = 0.0
        elif wind_speed <= self.config.wind_medium_threshold:
            memberships[FuzzyLevel.MEDIUM] = (
                (wind_speed - self.config.wind_low_threshold) /
                (self.config.wind_medium_threshold - self.config.wind_low_threshold)
            )
        elif wind_speed <= self.config.wind_high_threshold:
            memberships[FuzzyLevel.MEDIUM] = (
                (self.config.wind_high_threshold - wind_speed) /
                (self.config.wind_high_threshold - self.config.wind_medium_threshold)
            )
        else:
            memberships[FuzzyLevel.MEDIUM] = 0.0
        
        if wind_speed <= self.config.wind_medium_threshold:
            memberships[FuzzyLevel.HIGH] = 0.0
        elif wind_speed <= self.config.wind_high_threshold:
            memberships[FuzzyLevel.HIGH] = (
                (wind_speed - self.config.wind_medium_threshold) /
                (self.config.wind_high_threshold - self.config.wind_medium_threshold)
            )
        else:
            memberships[FuzzyLevel.HIGH] = 1.0
        
        return memberships
    
    def _apply_fuzzy_rules(
        self,
        risk_fuzzy: Dict[FuzzyLevel, float],
        agreement_fuzzy: Dict[FuzzyLevel, float],
        wind_fuzzy: Dict[FuzzyLevel, float]
    ) -> Dict[FuzzyLevel, float]:
        """
        Apply fuzzy inference rules to determine fire confidence.
        
        Rules as specified in the system design:
        - IF risk is High AND agreement is High AND wind is High 
          THEN fire confidence is Very High
        - IF risk is Medium AND agreement is High 
          THEN fire confidence is High
        - ELSE fire confidence is Low
        
        Args:
            risk_fuzzy: Fuzzy risk memberships
            agreement_fuzzy: Fuzzy agreement memberships
            wind_fuzzy: Fuzzy wind memberships
            
        Returns:
            Dictionary mapping output FuzzyLevel to activation strength
        """
        output = {
            FuzzyLevel.LOW: 0.0,
            FuzzyLevel.MEDIUM: 0.0,
            FuzzyLevel.HIGH: 0.0,
            FuzzyLevel.VERY_HIGH: 0.0
        }
        
        # Rule 1: IF risk=High AND agreement=High AND wind=High THEN Very High
        rule1_strength = min(
            risk_fuzzy.get(FuzzyLevel.HIGH, 0.0),
            agreement_fuzzy.get(FuzzyLevel.HIGH, 0.0),
            wind_fuzzy.get(FuzzyLevel.HIGH, 0.0)
        )
        output[FuzzyLevel.VERY_HIGH] = max(
            output[FuzzyLevel.VERY_HIGH],
            rule1_strength
        )
        
        # Rule 2: IF risk=High AND agreement=High THEN High
        rule2_strength = min(
            risk_fuzzy.get(FuzzyLevel.HIGH, 0.0),
            agreement_fuzzy.get(FuzzyLevel.HIGH, 0.0)
        )
        output[FuzzyLevel.HIGH] = max(output[FuzzyLevel.HIGH], rule2_strength)
        
        # Rule 3: IF risk=Medium AND agreement=High THEN High
        rule3_strength = min(
            risk_fuzzy.get(FuzzyLevel.MEDIUM, 0.0),
            agreement_fuzzy.get(FuzzyLevel.HIGH, 0.0)
        )
        output[FuzzyLevel.HIGH] = max(output[FuzzyLevel.HIGH], rule3_strength)
        
        # Rule 4: IF risk=High AND agreement=Medium THEN Medium
        rule4_strength = min(
            risk_fuzzy.get(FuzzyLevel.HIGH, 0.0),
            agreement_fuzzy.get(FuzzyLevel.MEDIUM, 0.0)
        )
        output[FuzzyLevel.MEDIUM] = max(
            output[FuzzyLevel.MEDIUM],
            rule4_strength
        )
        
        # Rule 5: IF risk=Medium AND agreement=Medium THEN Medium
        rule5_strength = min(
            risk_fuzzy.get(FuzzyLevel.MEDIUM, 0.0),
            agreement_fuzzy.get(FuzzyLevel.MEDIUM, 0.0)
        )
        output[FuzzyLevel.MEDIUM] = max(
            output[FuzzyLevel.MEDIUM],
            rule5_strength
        )
        
        # Default: Low (if no other rules fire significantly)
        max_other = max(
            output[FuzzyLevel.MEDIUM],
            output[FuzzyLevel.HIGH],
            output[FuzzyLevel.VERY_HIGH]
        )
        output[FuzzyLevel.LOW] = max(0.0, 1.0 - max_other)
        
        return output
    
    def _defuzzify(
        self,
        output_fuzzy: Dict[FuzzyLevel, float]
    ) -> tuple:
        """
        Defuzzify output to get crisp confidence value and classification.
        
        Uses center of gravity method for defuzzification.
        
        Args:
            output_fuzzy: Fuzzy output memberships
            
        Returns:
            Tuple of (confidence_score, classification)
        """
        # Map fuzzy levels to numeric centers
        level_centers = {
            FuzzyLevel.LOW: 0.2,
            FuzzyLevel.MEDIUM: 0.5,
            FuzzyLevel.HIGH: 0.75,
            FuzzyLevel.VERY_HIGH: 0.95
        }
        
        # Center of gravity calculation
        numerator = sum(
            output_fuzzy[level] * level_centers[level]
            for level in output_fuzzy
        )
        denominator = sum(output_fuzzy.values())
        
        if denominator == 0:
            confidence = 0.0
        else:
            confidence = numerator / denominator
        
        # Get classification from highest membership
        classification = max(output_fuzzy.keys(), key=lambda k: output_fuzzy[k])
        
        return confidence, classification
    
    def _check_temporal_persistence(
        self,
        reports: List[NodeReport]
    ) -> bool:
        """
        Check if high-risk events persist over time.
        
        Args:
            reports: List of node reports
            
        Returns:
            True if persistence requirement is met
        """
        if len(reports) < 2:
            return True  # Can't check persistence with single report
        
        timestamps = sorted([r.timestamp for r in reports])
        duration = (timestamps[-1] - timestamps[0]).total_seconds()
        
        return duration >= self.config.min_persistence_seconds
    
    def add_node_report(self, report: NodeReport) -> None:
        """
        Add a node report to the event history.
        
        Args:
            report: Report from a sensor node
        """
        if report.node_id not in self._event_history:
            self._event_history[report.node_id] = []
        
        self._event_history[report.node_id].append(report)
        
        # Keep only recent history (last 100 reports per node)
        if len(self._event_history[report.node_id]) > 100:
            self._event_history[report.node_id] = (
                self._event_history[report.node_id][-100:]
            )
    
    def evaluate_event(
        self,
        reports: List[NodeReport],
        weather: Optional[WeatherContext] = None
    ) -> Dict[str, Any]:
        """
        Evaluate fire event using Fuzzy MCDA.
        
        This is the main fuzzy MCDA computation that:
        1. Aggregates node-level risk scores
        2. Checks temporal persistence
        3. Fuzzifies inputs
        4. Applies fuzzy inference rules
        5. Defuzzifies output
        
        Args:
            reports: List of reports from sensor nodes
            weather: Optional local weather context
            
        Returns:
            Dictionary containing:
            - fire_confidence: Confidence score (0-1)
            - classification: Fuzzy classification level
            - is_validated: Whether event is validated
            - node_agreement_ratio: Ratio of nodes reporting high risk
            - average_risk_score: Average risk across nodes
            - temporal_persistence: Whether persistence requirement met
            - fuzzy_inputs: Fuzzy membership values for inputs
            - fuzzy_output: Fuzzy membership values for output
            - rules_triggered: List of rules that contributed
            - timestamp: Time of evaluation
            - edge_id: Edge node identifier
        """
        weather = weather or WeatherContext()
        timestamp = datetime.now(timezone.utc)
        
        # Check minimum nodes requirement
        if len(reports) < self.config.min_nodes_for_assessment:
            return {
                "fire_confidence": 0.0,
                "classification": FuzzyLevel.LOW.value,
                "is_validated": False,
                "reason": "insufficient_nodes",
                "timestamp": timestamp.isoformat(),
                "edge_id": self.edge_id
            }
        
        # Calculate aggregate metrics
        risk_scores = [r.risk_score for r in reports]
        average_risk = sum(risk_scores) / len(risk_scores)
        
        # Calculate agreement (ratio of nodes with risk > threshold)
        high_risk_count = sum(
            1 for r in reports 
            if r.risk_score >= self.config.risk_high_threshold
        )
        agreement_ratio = high_risk_count / len(reports)
        
        # Check temporal persistence
        persistence_met = self._check_temporal_persistence(reports)
        
        # Fuzzify inputs
        risk_fuzzy = self._fuzzify_risk(average_risk)
        agreement_fuzzy = self._fuzzify_agreement(agreement_ratio)
        wind_fuzzy = self._fuzzify_wind(weather.wind_speed)
        
        # Apply fuzzy rules
        output_fuzzy = self._apply_fuzzy_rules(
            risk_fuzzy, agreement_fuzzy, wind_fuzzy
        )
        
        # Defuzzify
        confidence, classification = self._defuzzify(output_fuzzy)
        
        # Determine if event is validated
        is_validated = (
            confidence >= self.config.confidence_output_map[FuzzyLevel.MEDIUM] and
            persistence_met
        )
        
        # Build rules triggered info
        rules_triggered = []
        if output_fuzzy[FuzzyLevel.VERY_HIGH] > 0.1:
            rules_triggered.append("risk_high_agreement_high_wind_high")
        if output_fuzzy[FuzzyLevel.HIGH] > 0.1:
            rules_triggered.append("risk_high_agreement_high")
        if output_fuzzy[FuzzyLevel.MEDIUM] > 0.1:
            rules_triggered.append("risk_medium_agreement_medium")
        
        result = {
            "fire_confidence": confidence,
            "classification": classification.value,
            "is_validated": is_validated,
            "node_agreement_ratio": agreement_ratio,
            "average_risk_score": average_risk,
            "temporal_persistence": persistence_met,
            "fuzzy_inputs": {
                "risk": {k.value: v for k, v in risk_fuzzy.items()},
                "agreement": {k.value: v for k, v in agreement_fuzzy.items()},
                "wind": {k.value: v for k, v in wind_fuzzy.items()}
            },
            "fuzzy_output": {k.value: v for k, v in output_fuzzy.items()},
            "rules_triggered": rules_triggered,
            "weather_context": {
                "wind_speed": weather.wind_speed,
                "wind_direction": weather.wind_direction,
                "dryness_index": weather.dryness_index
            },
            "timestamp": timestamp.isoformat(),
            "edge_id": self.edge_id,
            "node_count": len(reports)
        }
        
        # Log decision for explainability
        self.logger.log_decision(
            layer="edge",
            inputs={
                "average_risk": average_risk,
                "agreement_ratio": agreement_ratio,
                "wind_speed": weather.wind_speed
            },
            weights={
                "fuzzy_rules": "rule_based_inference"
            },
            score=confidence,
            confidence=confidence,
            metadata={
                "edge_id": self.edge_id,
                "classification": classification.value,
                "node_count": len(reports)
            }
        )
        
        return result
