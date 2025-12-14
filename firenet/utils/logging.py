"""
Logging utilities for MCDA explainability.

Provides structured logging of MCDA decisions for audit trails
and explainability requirements.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class MCDALogger:
    """
    Logger for MCDA decision tracking and explainability.
    
    Logs intermediate scores, weights, and decision factors for
    audit trails and debugging.
    """
    
    def __init__(self, name: str, level: int = logging.INFO):
        """
        Initialize the MCDA logger.
        
        Args:
            name: Logger name (typically layer name)
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_decision(
        self,
        layer: str,
        inputs: Dict[str, Any],
        weights: Dict[str, float],
        score: float,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log an MCDA decision with full explainability.
        
        Args:
            layer: The MCDA layer (node/edge/cloud)
            inputs: Input values used in decision
            weights: Weights applied to each criterion
            score: Final computed score
            confidence: Optional confidence indicator
            metadata: Additional metadata
            
        Returns:
            The complete decision record
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "layer": layer,
            "inputs": inputs,
            "weights": weights,
            "score": score,
            "confidence": confidence,
            "metadata": metadata or {}
        }
        
        self.logger.info(f"MCDA Decision: {record}")
        return record
    
    def log_gate_failure(
        self,
        layer: str,
        gate_name: str,
        reason: str,
        values: Dict[str, Any]
    ) -> None:
        """
        Log a rule gate failure (sanity check).
        
        Args:
            layer: The MCDA layer
            gate_name: Name of the gate that blocked
            reason: Reason for blocking
            values: Values that triggered the block
        """
        self.logger.warning(
            f"Gate Blocked - Layer: {layer}, Gate: {gate_name}, "
            f"Reason: {reason}, Values: {values}"
        )
