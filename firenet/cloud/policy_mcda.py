"""
Cloud-Level Policy MCDA for FIRENET.

Implements global correlation, prioritization, and advisory generation
using AHP-weighted MCDA for response urgency estimation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from firenet.utils.logging import MCDALogger


class SeverityLevel(Enum):
    """Fire event severity classification."""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ResponseUrgency(Enum):
    """Response urgency levels."""
    MONITOR = "monitor"
    STANDBY = "standby"
    DEPLOY = "deploy"
    URGENT = "urgent"
    EMERGENCY = "emergency"


@dataclass
class EdgeEvent:
    """Validated event from an edge node."""
    edge_id: str
    fire_confidence: float
    classification: str
    location: Tuple[float, float]  # (latitude, longitude)
    timestamp: datetime
    node_count: int
    average_risk_score: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class WeatherForecast:
    """Weather forecast data for fire spread assessment."""
    temperature: float = 25.0      # Celsius
    humidity: float = 50.0         # Percentage
    wind_speed: float = 5.0        # m/s
    wind_direction: float = 0.0    # degrees
    precipitation: float = 0.0     # mm
    drought_index: float = 0.5     # 0-1 scale


@dataclass
class SeasonalRisk:
    """Seasonal fire risk data."""
    base_risk: float = 0.5         # 0-1 scale
    vegetation_dryness: float = 0.5  # 0-1 scale
    historical_frequency: float = 0.3  # Normalized historical fires


@dataclass
class SatelliteAlert:
    """Satellite-based fire alert (e.g., from FAST 3.0)."""
    confidence: float = 0.0        # 0-1 scale
    detection_time: Optional[datetime] = None
    source: str = "unknown"
    location: Optional[Tuple[float, float]] = None


@dataclass
class CloudMCDAConfig:
    """Configuration for Cloud-level Policy MCDA."""
    # Default AHP-derived weights (can be updated by expert input)
    weight_edge_confidence: float = 0.35
    weight_weather_severity: float = 0.25
    weight_seasonal_risk: float = 0.20
    weight_satellite_confirmation: float = 0.20
    
    # Severity thresholds
    severity_minimal_threshold: float = 0.2
    severity_low_threshold: float = 0.4
    severity_moderate_threshold: float = 0.6
    severity_high_threshold: float = 0.8
    
    # Response urgency thresholds
    urgency_monitor_threshold: float = 0.2
    urgency_standby_threshold: float = 0.4
    urgency_deploy_threshold: float = 0.6
    urgency_urgent_threshold: float = 0.8
    
    # AHP consistency ratio threshold
    ahp_cr_threshold: float = 0.1


class AHPWeightCalculator:
    """
    Analytic Hierarchy Process (AHP) weight calculator.
    
    Used for periodic weight calibration based on expert input
    and historical outcomes.
    """
    
    # Random consistency index values for different matrix sizes
    RI_VALUES = {
        1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
    }
    
    def __init__(self, criteria_names: List[str]):
        """
        Initialize the AHP calculator.
        
        Args:
            criteria_names: Names of the criteria to weight
        """
        self.criteria_names = criteria_names
        self.n = len(criteria_names)
    
    def calculate_weights(
        self,
        comparison_matrix: List[List[float]]
    ) -> Tuple[Dict[str, float], float, bool]:
        """
        Calculate weights from a pairwise comparison matrix.
        
        The comparison matrix uses the Saaty scale:
        1 = Equal importance
        3 = Moderate importance
        5 = Strong importance
        7 = Very strong importance
        9 = Extreme importance
        
        Args:
            comparison_matrix: n x n pairwise comparison matrix
            
        Returns:
            Tuple of (weights dict, consistency ratio, is_consistent)
        """
        n = self.n
        
        # Normalize columns
        col_sums = [
            sum(comparison_matrix[i][j] for i in range(n))
            for j in range(n)
        ]
        
        normalized = [
            [comparison_matrix[i][j] / col_sums[j] for j in range(n)]
            for i in range(n)
        ]
        
        # Calculate priority vector (average of rows)
        priorities = [
            sum(normalized[i]) / n
            for i in range(n)
        ]
        
        # Calculate lambda_max for consistency check
        weighted_sums = [
            sum(comparison_matrix[i][j] * priorities[j] for j in range(n))
            for i in range(n)
        ]
        
        lambda_max = sum(
            weighted_sums[i] / priorities[i] if priorities[i] > 0 else 0
            for i in range(n)
        ) / n
        
        # Calculate consistency index and ratio
        ci = (lambda_max - n) / (n - 1) if n > 1 else 0
        ri = self.RI_VALUES.get(n, 1.49)
        cr = ci / ri if ri > 0 else 0
        
        is_consistent = cr < 0.1
        
        weights = {
            self.criteria_names[i]: priorities[i]
            for i in range(n)
        }
        
        return weights, cr, is_consistent
    
    def create_default_matrix(self) -> List[List[float]]:
        """
        Create a default comparison matrix based on FIRENET priorities.
        
        Default priority order:
        Edge Confidence > Weather > Seasonal Risk = Satellite
        
        Returns:
            Default pairwise comparison matrix
        """
        # Based on typical expert judgment for fire detection
        return [
            [1.0, 2.0, 2.0, 2.0],     # Edge confidence
            [0.5, 1.0, 1.5, 1.5],     # Weather severity
            [0.5, 0.67, 1.0, 1.0],    # Seasonal risk
            [0.5, 0.67, 1.0, 1.0]     # Satellite confirmation
        ]


class CloudMCDA:
    """
    Cloud-level Policy MCDA processor for global correlation.
    
    Performs:
    - Event ranking across all edges
    - Response urgency estimation
    - Resource allocation suggestions
    - Weight calibration via AHP
    
    Key responsibilities:
    - Global event correlation
    - Priority scoring
    - Severity classification
    - Response advisories
    """
    
    def __init__(
        self,
        config: Optional[CloudMCDAConfig] = None
    ):
        """
        Initialize the Cloud MCDA processor.
        
        Args:
            config: Configuration for policy MCDA processing
        """
        self.config = config or CloudMCDAConfig()
        self.logger = MCDALogger("CloudMCDA")
        
        self._ahp_calculator = AHPWeightCalculator([
            "edge_confidence",
            "weather_severity",
            "seasonal_risk",
            "satellite_confirmation"
        ])
        
        # Validate weights sum to 1.0
        total_weight = (
            self.config.weight_edge_confidence +
            self.config.weight_weather_severity +
            self.config.weight_seasonal_risk +
            self.config.weight_satellite_confirmation
        )
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    def calibrate_weights_ahp(
        self,
        comparison_matrix: Optional[List[List[float]]] = None
    ) -> Dict[str, Any]:
        """
        Calibrate MCDA weights using AHP.
        
        This should be called periodically with updated expert input
        and historical outcome analysis.
        
        Args:
            comparison_matrix: Optional pairwise comparison matrix.
                              If None, uses default based on expert input.
                              
        Returns:
            Dictionary with new weights and consistency information
        """
        if comparison_matrix is None:
            comparison_matrix = self._ahp_calculator.create_default_matrix()
        
        weights, cr, is_consistent = self._ahp_calculator.calculate_weights(
            comparison_matrix
        )
        
        result = {
            "weights": weights,
            "consistency_ratio": cr,
            "is_consistent": is_consistent,
            "applied": False
        }
        
        if is_consistent:
            # Update configuration with new weights
            self.config.weight_edge_confidence = weights["edge_confidence"]
            self.config.weight_weather_severity = weights["weather_severity"]
            self.config.weight_seasonal_risk = weights["seasonal_risk"]
            self.config.weight_satellite_confirmation = weights[
                "satellite_confirmation"
            ]
            result["applied"] = True
            
            self.logger.logger.info(
                f"AHP weights calibrated: {weights}, CR={cr:.4f}"
            )
        else:
            self.logger.logger.warning(
                f"AHP weights inconsistent (CR={cr:.4f} > 0.1), not applied"
            )
        
        return result
    
    def _calculate_weather_severity(
        self,
        weather: WeatherForecast
    ) -> float:
        """
        Calculate weather severity score for fire spread risk.
        
        Args:
            weather: Weather forecast data
            
        Returns:
            Weather severity score (0-1)
        """
        # High temp, low humidity, high wind, no precipitation = high severity
        temp_factor = min(1.0, max(0.0, (weather.temperature - 20) / 30))
        humidity_factor = 1.0 - (weather.humidity / 100.0)
        wind_factor = min(1.0, weather.wind_speed / 30.0)
        precip_factor = 1.0 - min(1.0, weather.precipitation / 10.0)
        drought_factor = weather.drought_index
        
        # Weighted combination
        severity = (
            0.25 * temp_factor +
            0.25 * humidity_factor +
            0.20 * wind_factor +
            0.15 * precip_factor +
            0.15 * drought_factor
        )
        
        return min(1.0, max(0.0, severity))
    
    def _calculate_seasonal_score(
        self,
        seasonal: SeasonalRisk
    ) -> float:
        """
        Calculate seasonal risk score.
        
        Args:
            seasonal: Seasonal risk data
            
        Returns:
            Seasonal risk score (0-1)
        """
        score = (
            0.40 * seasonal.base_risk +
            0.35 * seasonal.vegetation_dryness +
            0.25 * seasonal.historical_frequency
        )
        
        return min(1.0, max(0.0, score))
    
    def _classify_severity(self, priority_score: float) -> SeverityLevel:
        """
        Classify event severity based on priority score.
        
        Args:
            priority_score: Computed priority score
            
        Returns:
            Severity level classification
        """
        if priority_score >= self.config.severity_high_threshold:
            return SeverityLevel.CRITICAL
        elif priority_score >= self.config.severity_moderate_threshold:
            return SeverityLevel.HIGH
        elif priority_score >= self.config.severity_low_threshold:
            return SeverityLevel.MODERATE
        elif priority_score >= self.config.severity_minimal_threshold:
            return SeverityLevel.LOW
        else:
            return SeverityLevel.MINIMAL
    
    def _determine_urgency(self, priority_score: float) -> ResponseUrgency:
        """
        Determine response urgency based on priority score.
        
        Args:
            priority_score: Computed priority score
            
        Returns:
            Response urgency level
        """
        if priority_score >= self.config.urgency_urgent_threshold:
            return ResponseUrgency.EMERGENCY
        elif priority_score >= self.config.urgency_deploy_threshold:
            return ResponseUrgency.URGENT
        elif priority_score >= self.config.urgency_standby_threshold:
            return ResponseUrgency.DEPLOY
        elif priority_score >= self.config.urgency_monitor_threshold:
            return ResponseUrgency.STANDBY
        else:
            return ResponseUrgency.MONITOR
    
    def _generate_advisory(
        self,
        severity: SeverityLevel,
        urgency: ResponseUrgency,
        event: EdgeEvent,
        weather: WeatherForecast
    ) -> str:
        """
        Generate response advisory text.
        
        Args:
            severity: Classified severity level
            urgency: Determined urgency level
            event: The edge event
            weather: Weather forecast
            
        Returns:
            Advisory text for authorities
        """
        advisories = {
            ResponseUrgency.EMERGENCY: (
                f"EMERGENCY: Immediate response required at location "
                f"({event.location[0]:.4f}, {event.location[1]:.4f}). "
                f"Fire confidence: {event.fire_confidence:.1%}. "
                f"Wind speed: {weather.wind_speed} m/s. "
                f"Deploy all available resources immediately."
            ),
            ResponseUrgency.URGENT: (
                f"URGENT: Rapid response needed at location "
                f"({event.location[0]:.4f}, {event.location[1]:.4f}). "
                f"Fire confidence: {event.fire_confidence:.1%}. "
                f"Prepare aerial and ground units for deployment."
            ),
            ResponseUrgency.DEPLOY: (
                f"DEPLOY: Active response recommended at location "
                f"({event.location[0]:.4f}, {event.location[1]:.4f}). "
                f"Fire confidence: {event.fire_confidence:.1%}. "
                f"Dispatch ground crews for assessment and containment."
            ),
            ResponseUrgency.STANDBY: (
                f"STANDBY: Enhanced monitoring at location "
                f"({event.location[0]:.4f}, {event.location[1]:.4f}). "
                f"Fire confidence: {event.fire_confidence:.1%}. "
                f"Prepare resources for potential deployment."
            ),
            ResponseUrgency.MONITOR: (
                f"MONITOR: Continue surveillance at location "
                f"({event.location[0]:.4f}, {event.location[1]:.4f}). "
                f"Fire confidence: {event.fire_confidence:.1%}. "
                f"No immediate action required."
            )
        }
        
        return advisories.get(urgency, "No advisory generated.")
    
    def evaluate_event(
        self,
        event: EdgeEvent,
        weather: Optional[WeatherForecast] = None,
        seasonal: Optional[SeasonalRisk] = None,
        satellite: Optional[SatelliteAlert] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single edge event using policy MCDA.
        
        Computes priority score using weighted sum:
        Priority = w1 × Edge Confidence + w2 × Weather Severity
                 + w3 × Seasonal Risk + w4 × Satellite Confirmation
        
        Args:
            event: Validated event from edge node
            weather: Weather forecast data
            seasonal: Seasonal risk data
            satellite: Satellite-based alert data
            
        Returns:
            Dictionary containing:
            - priority_score: Overall priority score (0-1)
            - severity: Severity classification
            - urgency: Response urgency level
            - advisory: Response advisory text
            - component_scores: Individual factor scores
            - weights: Applied weights
            - timestamp: Evaluation timestamp
            - event_id: Unique event identifier
        """
        weather = weather or WeatherForecast()
        seasonal = seasonal or SeasonalRisk()
        satellite = satellite or SatelliteAlert()
        
        timestamp = datetime.now(timezone.utc)
        
        # Calculate component scores
        weather_severity = self._calculate_weather_severity(weather)
        seasonal_score = self._calculate_seasonal_score(seasonal)
        satellite_score = satellite.confidence
        
        # Get weights
        weights = {
            "edge_confidence": self.config.weight_edge_confidence,
            "weather_severity": self.config.weight_weather_severity,
            "seasonal_risk": self.config.weight_seasonal_risk,
            "satellite_confirmation": self.config.weight_satellite_confirmation
        }
        
        # Compute priority score
        priority_score = (
            weights["edge_confidence"] * event.fire_confidence +
            weights["weather_severity"] * weather_severity +
            weights["seasonal_risk"] * seasonal_score +
            weights["satellite_confirmation"] * satellite_score
        )
        
        # Classify and determine response
        severity = self._classify_severity(priority_score)
        urgency = self._determine_urgency(priority_score)
        advisory = self._generate_advisory(severity, urgency, event, weather)
        
        result = {
            "priority_score": priority_score,
            "severity": severity.value,
            "urgency": urgency.value,
            "advisory": advisory,
            "component_scores": {
                "edge_confidence": event.fire_confidence,
                "weather_severity": weather_severity,
                "seasonal_risk": seasonal_score,
                "satellite_confirmation": satellite_score
            },
            "weights": weights,
            "event": {
                "edge_id": event.edge_id,
                "location": event.location,
                "node_count": event.node_count,
                "original_confidence": event.fire_confidence
            },
            "timestamp": timestamp.isoformat(),
            "event_id": f"{event.edge_id}_{timestamp.strftime('%Y%m%d%H%M%S')}"
        }
        
        # Log decision for explainability
        self.logger.log_decision(
            layer="cloud",
            inputs={
                "edge_confidence": event.fire_confidence,
                "weather_severity": weather_severity,
                "seasonal_risk": seasonal_score,
                "satellite_confirmation": satellite_score
            },
            weights=weights,
            score=priority_score,
            confidence=priority_score,
            metadata={
                "severity": severity.value,
                "urgency": urgency.value,
                "location": event.location
            }
        )
        
        return result
    
    def rank_events(
        self,
        evaluations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rank multiple evaluated events by priority.
        
        Args:
            evaluations: List of evaluation results from evaluate_event
            
        Returns:
            Sorted list of evaluations (highest priority first)
        """
        ranked = sorted(
            evaluations,
            key=lambda e: e["priority_score"],
            reverse=True
        )
        
        # Add rank information
        for i, evaluation in enumerate(ranked):
            evaluation["rank"] = i + 1
        
        return ranked
    
    def generate_alert_summary(
        self,
        ranked_events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a summary report of all active alerts.
        
        Args:
            ranked_events: Ranked list of event evaluations
            
        Returns:
            Summary report with statistics and recommendations
        """
        if not ranked_events:
            return {
                "total_events": 0,
                "summary": "No active fire events detected.",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        # Count by severity
        severity_counts = {}
        urgency_counts = {}
        
        for event in ranked_events:
            sev = event["severity"]
            urg = event["urgency"]
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            urgency_counts[urg] = urgency_counts.get(urg, 0) + 1
        
        # Calculate statistics
        priority_scores = [e["priority_score"] for e in ranked_events]
        avg_priority = sum(priority_scores) / len(priority_scores)
        max_priority = max(priority_scores)
        
        return {
            "total_events": len(ranked_events),
            "severity_distribution": severity_counts,
            "urgency_distribution": urgency_counts,
            "statistics": {
                "average_priority": avg_priority,
                "max_priority": max_priority,
                "min_priority": min(priority_scores)
            },
            "top_priority_event": ranked_events[0] if ranked_events else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recommendations": self._generate_recommendations(ranked_events)
        }
    
    def _generate_recommendations(
        self,
        ranked_events: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate operational recommendations based on events.
        
        Args:
            ranked_events: Ranked list of events
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        critical_count = sum(
            1 for e in ranked_events if e["severity"] == "critical"
        )
        emergency_count = sum(
            1 for e in ranked_events if e["urgency"] == "emergency"
        )
        
        if emergency_count > 0:
            recommendations.append(
                f"IMMEDIATE ACTION: {emergency_count} emergency-level event(s) "
                f"require immediate response deployment."
            )
        
        if critical_count > 0:
            recommendations.append(
                f"CRITICAL ALERT: {critical_count} critical-severity event(s) "
                f"detected. Escalate to incident command."
            )
        
        if len(ranked_events) > 5:
            recommendations.append(
                "RESOURCE ALLOCATION: Multiple active events detected. "
                "Consider resource prioritization based on ranking."
            )
        
        if not recommendations:
            recommendations.append(
                "NORMAL OPERATIONS: Continue standard monitoring procedures."
            )
        
        return recommendations
