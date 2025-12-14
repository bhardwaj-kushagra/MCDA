"""
FIRENET - Fire Identification and Response Network for Early Tracking

A hierarchical, MCDA-driven, edge-first wildfire detection and decision-support 
system that fuses multi-modal ground sensing with contextual intelligence to 
deliver explainable, low-latency fire alerts.
"""

__version__ = "1.0.0"

from firenet.node.mcda import NodeMCDA
from firenet.edge.fuzzy_mcda import EdgeFuzzyMCDA
from firenet.cloud.policy_mcda import CloudMCDA

__all__ = ["NodeMCDA", "EdgeFuzzyMCDA", "CloudMCDA"]
