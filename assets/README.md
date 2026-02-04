# Assets Directory

This directory is for INFRAMIND project assets (images, diagrams, etc.) that will be referenced in the main README.

## Suggested Assets for Your Paper

When preparing for publication, consider adding:

### 1. Architecture Diagram
**File**: `architecture.png` or `inframind_architecture.png`
- Overview of the hierarchical CMDP system
- Planner and Executor components
- Infrastructure monitoring integration
- Data flow between components

### 2. System Overview
**File**: `system_overview.png`
- High-level view of INFRAMIND
- Comparison with baseline MAS Router
- Key differences highlighted

### 3. Performance Plots
**File**: `performance_comparison.png`
- Accuracy vs. latency trade-offs
- Performance under different load conditions
- Comparison with baseline across datasets

### 4. Load-Adaptive Behavior
**File**: `load_adaptive.png` or `adaptive_routing.png`
- How INFRAMIND adapts to different loads
- Model selection patterns under high/low load
- Strategy selection dynamics

### 5. Infrastructure Monitoring
**File**: `metrics_dashboard.png`
- Example vLLM metrics visualization
- Queue depth, cache usage, latency tracking
- Real-time system state representation

## Usage in README

Once you add assets, reference them in the main README like this:

```markdown
## Architecture

![INFRAMIND Architecture](assets/architecture.png)

## Performance

![Performance Comparison](assets/performance_comparison.png)
```

## Asset Guidelines

- **Format**: PNG or SVG (SVG preferred for diagrams)
- **Resolution**: At least 1200px wide for publication quality
- **Color Scheme**: Professional academic color palette
- **Labels**: Clear, readable text even when scaled down
- **File Size**: Optimize images (compress PNGs, use SVG when possible)

## Generating Plots from Results

Use the visualization scripts to generate performance plots from your experiments:

```bash
python visualization/generate_motivation_plots.py
```

Or use the Jupyter notebook:
```bash
jupyter notebook visualization/motivation_plots.ipynb
```
