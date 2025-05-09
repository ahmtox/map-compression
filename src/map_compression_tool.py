#!/usr/bin/env python3
"""
Map Compression Tool - A comprehensive platform for evaluating map compression techniques.

Usage:
    ./map_compression_tool.py --help
"""
import os
import time
import json
import argparse
import numpy as np
from PIL import Image

from src.core import BlockCompressedImage, ContextEncoder, ArithmeticContextEncoder
from src.processing import load_and_split_layers, block_decomposition, prepare_additional_maps
from src.viewer import MapViewer, InstrumentedViewer
from src.metrics import profile_memory, batch_visual_assessment, analyze_results, print_statistical_summary
from src.advanced_metrics import (run_enhanced_experiment, analyze_block_size_anomaly, 
                                simulate_network_performance, compare_with_standard_formats)

def ensure_dirs_exist():
    """Ensure necessary directories exist"""
    os.makedirs("results", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)

@profile_memory
def compress_map(input_path, output_path, block_size=128, use_arithmetic=False):
    """Compress a map image with block-based approach"""
    print(f"Compressing {input_path} with block size {block_size}")
    
    # Load and split into layers
    layers = load_and_split_layers(input_path)
    
    # Create the container
    img = Image.open(input_path)
    width, height = img.size
    container = BlockCompressedImage()
    container.metadata["block_size"] = block_size
    
    # Process each layer
    encoder = ArithmeticContextEncoder() if use_arithmetic else ContextEncoder()
    start_time = time.time()
    
    for layer_name, layer_data in layers.items():
        print(f"Processing layer {layer_name}")
        container.add_layer(layer_name, width, height)
        
        # Split into blocks
        blocks = block_decomposition(layer_data, block_size)
        
        # Compress each block
        for coords, block in blocks.items():
            compressed = encoder.encode_block(block)
            container.add_block(layer_name, coords[0], coords[1], compressed)
    
    # Save the result
    container.save(output_path)
    
    compression_time = time.time() - start_time
    print(f"Compression completed in {compression_time:.2f} seconds")
    
    # Calculate compression ratio
    original_size = os.path.getsize(input_path)
    compressed_size = os.path.getsize(output_path)
    ratio = original_size / compressed_size
    print(f"Original size: {original_size} bytes")
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Compression ratio: {ratio:.2f}:1")
    
    return {
        "input_path": input_path,
        "output_path": output_path,
        "block_size": block_size,
        "use_arithmetic": use_arithmetic,
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compression_ratio": ratio,
        "compression_time": compression_time
    }

@profile_memory
def run_experiment(map_paths, block_sizes=[64, 128, 256], use_arithmetic=False, viewport_size=(800, 600)):
    """Run comprehensive experiments with different block sizes"""
    results = []
    
    for input_path in map_paths:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        
        for block_size in block_sizes:
            output_path = f"results/{base_name}_b{block_size}_{'arith' if use_arithmetic else 'ctx'}.mcmp"
            
            # Compress the map
            compress_result = compress_map(input_path, output_path, block_size, use_arithmetic)
            
            # Test viewport rendering performance
            viewer = MapViewer(output_path)
            
            # Simulate panning
            pan_results = []
            img = Image.open(input_path)
            width, height = img.size
            
            # Do 20 steps of panning
            pan_steps = 20
            pan_results = viewer.simulate_panning(0, 0, pan_steps, step_size=50)
            
            # Combine results
            experiment_result = {
                **compress_result,
                "pan_results": pan_results,
                "avg_decode_time": sum(r["decode_time"] for r in pan_results) / len(pan_results)
            }
            
            results.append(experiment_result)
    
    # Save aggregated results
    results_file = "results/arithmetic_experiment_results.json" if use_arithmetic else "results/experiment_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Run visual quality assessment
    quality_metrics = batch_visual_assessment(results, MapViewer)
    
    # Add quality metrics to results
    for result in results:
        for metric in quality_metrics:
            if metric["map"] == os.path.basename(result["input_path"]) and metric["block_size"] == result["block_size"]:
                result["psnr"] = metric["psnr"]
                result["ssim"] = metric["ssim"]
    
    # Save updated results
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    return results, results_file

def investigate_anomaly(map_paths, block_sizes=[64, 128, 256]):
    """Investigate performance anomalies across different block sizes"""
    results = []
    
    for input_path in map_paths:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        
        print(f"\n=== Investigating {base_name} ===")
        
        timing_data = {}
        
        for block_size in block_sizes:
            output_path = f"results/{base_name}_b{block_size}_ctx.mcmp"
            
            print(f"\nTesting block size {block_size}px:")
            
            # Use instrumented viewer
            viewer = InstrumentedViewer(output_path)
            
            # Do several viewport renders to get stable timing
            for i in range(3):
                viewport, _ = viewer.get_viewport(0, 0, 800, 600)
            
            # Print timing information
            viewer.print_timing_summary()
            
            # Store timing breakdown
            timing_data[block_size] = {op: sum(times)/len(times) for op, times in viewer.instrumentation.items()}
            
            # Create timing breakdown plot
            plot = viewer.plot_timing_breakdown(f"Timing Breakdown - {base_name}, {block_size}px blocks")
            plot.savefig(f"results/timing_breakdown_{base_name}_b{block_size}.png", dpi=300)
        
        # Compare timing across block sizes
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        
        operations = list(next(iter(timing_data.values())).keys())
        x = np.arange(len(operations))
        width = 0.25
        offsets = np.linspace(-width, width, len(block_sizes))
        
        for i, block_size in enumerate(block_sizes):
            times = [timing_data[block_size].get(op, 0) * 1000 for op in operations]  # Convert to ms
            plt.bar(x + offsets[i], times, width, label=f'{block_size}px blocks')
        
        plt.title(f"Operation Timing Comparison - {base_name}")
        plt.xlabel("Operation")
        plt.ylabel("Average Time (ms)")
        plt.xticks(x, operations, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/timing_comparison_{base_name}.png", dpi=300)
        plt.close()
        
        results.append({
            "map": base_name,
            "timing_data": timing_data
        })
    
    # Save results
    with open("results/anomaly_investigation.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def load_existing_results(filename="results/experiment_results.json"):
    """Load existing experiment results from JSON file"""
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Error: {filename} is not a valid JSON file")
            return []
    else:
        print(f"Warning: {filename} does not exist")
        return []

def debug_container(filename):
    """Debug a container file to diagnose issues"""
    return BlockCompressedImage.debug_container(filename)

def main():
    """Main entry point for the map compression tool"""
    parser = argparse.ArgumentParser(description="Map Compression Tool")
    
    # Mode selection
    mode_group = parser.add_argument_group("Operating Mode")
    mode = mode_group.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="Run complete compression experiments")
    mode.add_argument("--analyze", action="store_true", help="Analyze existing results")
    mode.add_argument("--investigate", action="store_true", help="Investigate performance anomalies")
    mode.add_argument("--debug", action="store_true", help="Debug container files")
    
    # Experiment options
    experiment_group = parser.add_argument_group("Experiment Options")
    experiment_group.add_argument("--arithmetic", action="store_true", help="Use arithmetic coding instead of context-based")
    experiment_group.add_argument("--block-sizes", type=int, nargs="+", default=[64, 128, 256], 
                        help="Block sizes to test (default: 64 128 256)")
    experiment_group.add_argument("--additional-maps", action="store_true", help="Include additional map types")
    experiment_group.add_argument("--skip-quality", action="store_true", help="Skip visual quality assessment")
    # Add enhanced options
    experiment_group.add_argument("--enhanced", action="store_true", 
                               help="Run enhanced experiments with multiple trials")
    experiment_group.add_argument("--trials", type=int, default=5,
                               help="Number of trials for enhanced experiments")
    
    # Analysis options
    analysis_group = parser.add_argument_group("Analysis Options")
    analysis_group.add_argument("--results-file", default=None, 
                       help="Path to results file (default: arithmetic_experiment_results.json or experiment_results.json)")
    analysis_group.add_argument("--network-simulation", action="store_true",
                               help="Run network simulation analysis")
    analysis_group.add_argument("--compare-formats", action="store_true",
                               help="Compare with standard image formats")
    analysis_group.add_argument("--analyze-anomaly", type=int, default=128,
                               help="Analyze behavior around a specific block size")
    
    # Debug options
    debug_group = parser.add_argument_group("Debug Options")
    debug_group.add_argument("--container-file", help="Container file to debug")
    
    args = parser.parse_args()
    
    # Ensure necessary directories exist
    ensure_dirs_exist()
    
    # Default maps
    map_paths = [
        "data/processed/berlin_roads.png",
        "data/processed/berlin_buildings.png",
        "data/processed/berlin_combined.png"
    ]
    
    if args.run:
        # Add additional maps if requested
        if args.additional_maps:
            additional_maps = prepare_additional_maps()
            if additional_maps:
                map_paths.extend(additional_maps)
        
        # Add handling for enhanced experiment
        if args.enhanced:
            print("\nRunning enhanced experiments with multiple trials...")
            results = run_enhanced_experiment(
                map_paths, 
                block_sizes=args.block_sizes,
                trials=args.trials,
                use_arithmetic=args.arithmetic
            )
            return  # Enhanced run already includes analysis
                
        # Run standard experiments
        print("\nRunning compression experiments...")
        results, results_file = run_experiment(map_paths, args.block_sizes, args.arithmetic)
        
        # Analysis
        print("\nAnalyzing results...")
        df, stats = analyze_results(results_file)
        print_statistical_summary(stats)
        
        # Print summary
        print("\nExperiment Summary:")
        print("------------------")
        for result in results:
            print(f"Map: {os.path.basename(result['input_path'])}, Block Size: {result['block_size']}")
            print(f"  Compression Ratio: {result['compression_ratio']:.2f}:1")
            print(f"  Avg Decode Time: {result['avg_decode_time']*1000:.2f}ms")
            if 'psnr' in result:
                print(f"  PSNR: {result['psnr']:.2f}dB, SSIM: {result['ssim']:.4f}")
            if 'memory_profile' in result:
                print(f"  Peak Memory: {result['memory_profile']['peak_memory_mb']:.2f}MB")
            print()
    
    elif args.analyze:
        # Choose the right results file
        if args.results_file:
            results_file = args.results_file
        else:
            results_file = "results/arithmetic_experiment_results.json" if args.arithmetic else "results/experiment_results.json"
        
        # Load existing results
        print(f"\nLoading results from {results_file}...")
        results = load_existing_results(results_file)
        
        if not results:
            print("No results found to analyze.")
            return
        
        # Run analysis
        print("\nAnalyzing results...")
        df, stats = analyze_results(results_file)
        print_statistical_summary(stats)
        
        # Add options for specific analyses
        if args.compare_formats:
            print("\nComparing with standard formats...")
            for input_path in map_paths:
                compare_with_standard_formats(input_path)
        
        if args.network_simulation:
            print("\nRunning network simulation...")
            simulate_network_performance(results_file)
        
        if args.analyze_anomaly:
            print(f"\nAnalyzing block size anomaly at {args.analyze_anomaly}px...")
            analyze_block_size_anomaly(results_file, args.analyze_anomaly)
            
        # Run visual quality assessment if needed
        if not args.skip_quality:
            print("\nRunning visual quality assessment...")
            quality_metrics = batch_visual_assessment(results, MapViewer)
            
            # Add quality metrics to results
            for result in results:
                for metric in quality_metrics:
                    if (metric["map"] == os.path.basename(result["input_path"]) and 
                        metric["block_size"] == result["block_size"]):
                        result["psnr"] = metric["psnr"]
                        result["ssim"] = metric["ssim"]
            
            # Save updated results
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
    
    elif args.investigate:
        # Use all available map paths for investigation
        print("\nRunning anomaly investigation...")
        anomaly_results = investigate_anomaly(map_paths, args.block_sizes)
        print("\nAnomaly investigation completed.")
    
    elif args.debug:
        if not args.container_file:
            parser.error("--debug requires --container-file")
        
        print(f"\nDebugging container file: {args.container_file}")
        debug_container(args.container_file)

if __name__ == "__main__":
    main()