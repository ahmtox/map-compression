Refined Research Question: "How can layering strategies and block size optimization balance compression ratio, decode time, and memory usage for map streaming applications across different device classes?"

This refinement:
1. Emphasizes the trade-offs between competing factors (compression vs. speed vs. memory)
2. Acknowledges different device constraints (mobile vs. desktop)
3. Keeps your core interests in layering and block size optimization

Code Structure:
```
mapcompression/
│
├── src/
│   ├── __init__.py
│   ├── data_acquisition.py     # Get and preprocess OSM data
│   ├── preprocess.py           # Split into layers and blocks
│   ├── encoder.py              # Context-based encoding
│   ├── decoder.py              # Decoding algorithms
│   ├── container.py            # File format and block storage
│   ├── viewer.py               # Viewport rendering and panning
│   ├── main.py                 # Main experiments
│   └── analyze.py              # Results analysis
│
├── data/
│   ├── raw/                    # Downloaded OSM data
│   └── processed/              # Converted raster images
│
├── results/                    # Experiment results and charts
│
├── notebooks/                  # Jupyter notebooks for exploration
│
└── requirements.txt            # Dependencies
```

Suggested Research Modifications

1. Consider these additional research directions to make your project more impactful:
2. Mixed Bitplane + Semantic Layering: Combine color-plane decomposition with selective semantic layers to get the best of both worlds.
3. Dynamic Block Size Selection: Analyze if certain map areas benefit from different block sizes and implement a variable block size scheme.
4. Device Profile Optimization: Create optimized compression variants targeting desktop, mobile, and low-power devices with different block sizes and layer strategies.
5. Hybrid Progressive Approach: Implement a two-stage loading where coarse blocks load first and higher-resolution refinements come later.
6. Adaptive Context Model: Experiment with different context templates for different map content types.

Experimentation Guidelines
When running your experiments, I recommend:

1. Varied Input Maps: Test with at least 3 different input maps (urban dense, suburban sparse, mixed terrain).
2. Block Size Range: Try more block sizes (64, 96, 128, 192, 256) to find the optimal point.
3. Realistic Interaction Patterns: Simulate not just left-right panning but random jumps and zoom transitions.
4. Memory Usage Tracking: Add memory profiling to track peak RAM during decoding.
5. Device Testing: Test the same map on different hardware profiles if possible.