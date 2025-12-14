"""
Normalization utilities for MCDA scoring.

Provides functions to normalize sensor readings to a 0-1 range for 
consistent MCDA processing.
"""

from typing import Union


def normalize(
    value: Union[int, float],
    min_val: Union[int, float],
    max_val: Union[int, float]
) -> float:
    """
    Normalize a value to a 0-1 range.
    
    Args:
        value: The value to normalize
        min_val: The minimum expected value
        max_val: The maximum expected value
        
    Returns:
        Normalized value between 0 and 1
        
    Raises:
        ValueError: If min_val equals max_val
    """
    if min_val == max_val:
        raise ValueError("min_val and max_val cannot be equal")
    
    # Clamp value to range
    clamped = max(min_val, min(max_val, value))
    normalized = (clamped - min_val) / (max_val - min_val)
    return float(normalized)


def inverse_normalize(
    value: Union[int, float],
    min_val: Union[int, float],
    max_val: Union[int, float]
) -> float:
    """
    Inverse normalize a value to a 0-1 range.
    
    Higher original values result in lower normalized values.
    Used for metrics like humidity where low values indicate higher fire risk.
    
    Args:
        value: The value to normalize
        min_val: The minimum expected value
        max_val: The maximum expected value
        
    Returns:
        Inverse normalized value between 0 and 1
        
    Raises:
        ValueError: If min_val equals max_val
    """
    return 1.0 - normalize(value, min_val, max_val)
