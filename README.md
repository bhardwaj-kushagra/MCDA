# FIRENET MCDA

**Fire Identification and Response Network for Early Tracking**

A hierarchical, MCDA-driven, edge-first wildfire detection and decision-support system that fuses multi-modal ground sensing with contextual intelligence to deliver explainable, low-latency fire alerts.

## Overview

FIRENET is a distributed wildfire detection and decision-support system designed to:

- **Detect fires earlier** than satellite-only systems
- **Reduce false positives** from single-sensor triggers  
- **Operate in data-scarce**, remote forest environments
- **Provide explainable**, policy-grade alerts to authorities
- **Complement existing satellite systems** such as FAST 3.0

The system uses Multi-Criteria Decision Analysis (MCDA) instead of black-box ML, ensuring decisions are **deterministic, tunable, and transparent**.

## Architecture

FIRENET follows a three-tier architecture:

```
Sensor Nodes → Edge Node → Cloud Platform → Authorities
```

### 1. Node Layer (Sensor-Level MCDA)
- Lightweight fire risk estimation from IoT sensors
- Weighted scoring of temperature, humidity, gas, IR, and light
- Rule-based sanity gating to filter impossible conditions

### 2. Edge Layer (Fuzzy MCDA)  
- Spatial and temporal aggregation of node reports
- Fuzzy logic for uncertainty handling
- Validates events before forwarding to cloud

### 3. Cloud Layer (Policy MCDA)
- Global event correlation and prioritization
- AHP-based weight calibration
- Response urgency and severity classification

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Node-Level MCDA

```python
from firenet import NodeMCDA
from firenet.node.mcda import SensorReading

# Initialize the node processor
node = NodeMCDA(node_id="sensor_001")

# Process a sensor reading
reading = SensorReading(
    temperature=45.0,    # Celsius
    humidity=25.0,       # Percentage
    gas=600.0,           # ppm
    infrared=0.7,        # 0-1 normalized
    lux=50000.0          # Ambient light
)

result = node.compute_risk(reading)
print(f"Fire Risk Score: {result['risk_score']:.2f}")
print(f"Confidence: {result['confidence']:.2f}")
```

### Edge-Level Fuzzy MCDA

```python
from datetime import datetime
from firenet import EdgeFuzzyMCDA
from firenet.edge.fuzzy_mcda import NodeReport, WeatherContext

# Initialize the edge processor
edge = EdgeFuzzyMCDA(edge_id="edge_001")

# Aggregate reports from multiple nodes
reports = [
    NodeReport(node_id="n1", risk_score=0.8, confidence=0.9, timestamp=datetime.now()),
    NodeReport(node_id="n2", risk_score=0.85, confidence=0.88, timestamp=datetime.now()),
    NodeReport(node_id="n3", risk_score=0.75, confidence=0.92, timestamp=datetime.now())
]

weather = WeatherContext(wind_speed=15.0, dryness_index=0.7)
result = edge.evaluate_event(reports, weather)

print(f"Fire Confidence: {result['fire_confidence']:.2f}")
print(f"Classification: {result['classification']}")
print(f"Validated: {result['is_validated']}")
```

### Cloud-Level Policy MCDA

```python
from datetime import datetime
from firenet import CloudMCDA
from firenet.cloud.policy_mcda import EdgeEvent, WeatherForecast, SeasonalRisk

# Initialize the cloud processor
cloud = CloudMCDA()

# Evaluate an edge event
event = EdgeEvent(
    edge_id="edge_001",
    fire_confidence=0.85,
    classification="high",
    location=(40.7128, -74.0060),
    timestamp=datetime.now(),
    node_count=5,
    average_risk_score=0.8
)

weather = WeatherForecast(temperature=40.0, humidity=20.0, wind_speed=20.0)
seasonal = SeasonalRisk(base_risk=0.7, vegetation_dryness=0.8)

result = cloud.evaluate_event(event, weather, seasonal)

print(f"Priority Score: {result['priority_score']:.2f}")
print(f"Severity: {result['severity']}")
print(f"Urgency: {result['urgency']}")
print(f"Advisory: {result['advisory']}")
```

### AHP Weight Calibration

```python
from firenet import CloudMCDA

cloud = CloudMCDA()

# Calibrate weights using expert comparison matrix
comparison_matrix = [
    [1.0, 2.0, 2.0, 2.0],     # Edge confidence
    [0.5, 1.0, 1.5, 1.5],     # Weather severity  
    [0.5, 0.67, 1.0, 1.0],    # Seasonal risk
    [0.5, 0.67, 1.0, 1.0]     # Satellite confirmation
]

result = cloud.calibrate_weights_ahp(comparison_matrix)
print(f"New Weights: {result['weights']}")
print(f"Consistency Ratio: {result['consistency_ratio']:.4f}")
print(f"Consistent: {result['is_consistent']}")
```

## MCDA Weights

### Node Layer (Default)
| Criterion | Weight |
|-----------|--------|
| Temperature | 0.30 |
| Gas | 0.25 |
| Infrared | 0.20 |
| Humidity | 0.15 |
| Lux | 0.10 |

### Cloud Layer (AHP-Calibrated)
| Criterion | Default Weight |
|-----------|----------------|
| Edge Confidence | 0.35 |
| Weather Severity | 0.25 |
| Seasonal Risk | 0.20 |
| Satellite Confirmation | 0.20 |

## Explainability

Every alert can be traced back to:
- Individual sensor contributions
- Applied MCDA weights  
- Fuzzy rules triggered
- Contextual factors considered

All decisions are logged with full audit trails for government adoption and policy decisions.

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=firenet --cov-report=term-missing
```

## Project Structure

```
firenet/
├── __init__.py           # Package initialization
├── node/
│   ├── __init__.py
│   └── mcda.py           # Node-level MCDA
├── edge/
│   ├── __init__.py
│   └── fuzzy_mcda.py     # Edge-level Fuzzy MCDA
├── cloud/
│   ├── __init__.py
│   └── policy_mcda.py    # Cloud-level Policy MCDA
└── utils/
    ├── __init__.py
    ├── normalization.py  # Normalization utilities
    └── logging.py        # MCDA logging utilities
tests/
├── test_node_mcda.py
├── test_edge_mcda.py
├── test_cloud_mcda.py
└── test_utils.py
```

## License

MIT License
