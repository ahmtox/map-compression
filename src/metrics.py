"""Map compression metrics and analysis tools."""
import os
import time
import json
import psutil
import tracemalloc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import wraps
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy import stats
from PIL import Image

# -------------------------------------------------------------------------
# Memory and Performance Profiling
# -------------------------------------------------------------------------
def memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # Convert to MB

class MemoryProfiler:
    """Class to track memory usage during operations"""
    
    def __init__(self):
        self.peak_memory = 0
        self.start_memory = 0
        self.tracemalloc_on = False
        
    def start(self):
        """Start memory profiling"""
        self.start_memory = memory_usage()
        self.peak_memory = self.start_memory
        if not self.tracemalloc_on:
            tracemalloc.start()
            self.tracemalloc_on = True
        
    def stop(self):
        """Stop memory profiling and return statistics"""
        current = memory_usage()
        if current > self.peak_memory:
            self.peak_memory = current
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.tracemalloc_on = False
        
        memory_increase = self.peak_memory - self.start_memory
        
        return {
            "start_memory_mb": self.start_memory,
            "peak_memory_mb": self.peak_memory,
            "memory_increase_mb": memory_increase,
            "tracemalloc_current_mb": current / 1024 / 1024,
            "tracemalloc_peak_mb": peak / 1024 / 1024
        }

def profile_memory(func):
    """Decorator to profile memory usage of a function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = MemoryProfiler()
        profiler.start()
        
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        memory_stats = profiler.stop()
        
        print(f"\nMemory Profile for {func.__name__}:")
        print(f"  Peak Memory: {memory_stats['peak_memory_mb']:.2f} MB")
        print(f"  Memory Increase: {memory_stats['memory_increase_mb']:.2f} MB")
        print(f"  Execution Time: {elapsed_time:.2f} seconds")
        
        # Add profile data to result if it's a dict
        if isinstance(result, dict):
            result["memory_profile"] = memory_stats
            result["execution_time"] = elapsed_time
        
        return result
    
    return wrapper

# -------------------------------------------------------------------------
# Image Quality Assessment
# -------------------------------------------------------------------------
def calculate_psnr(original, decoded):
    """Calculate Peak Signal-to-Noise Ratio between original and decoded images"""
    if original.shape != decoded.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Ensure images are properly scaled
    if original.dtype != np.uint8:
        original = (original * 255).astype(np.uint8)
    if decoded.dtype != np.uint8:
        decoded = (decoded * 255).astype(np.uint8)
    
    return psnr(original, decoded)

def calculate_ssim(original, decoded):
    """Calculate Structural Similarity Index between original and decoded images"""
    if original.shape != decoded.shape:
        print(f"WARNING: Shapes don't match for SSIM calculation")
        if len(original.shape) != len(decoded.shape):
            return 0.0  # Unable to calculate
    
    # Ensure images are properly scaled
    if original.dtype != np.uint8:
        original = (original * 255).astype(np.uint8)
    if decoded.dtype != np.uint8:
        decoded = (decoded * 255).astype(np.uint8)
    
    # Check if image is too small for default window size
    min_dim = min(original.shape[0], original.shape[1])
    
    if min_dim < 7:
        print(f"WARNING: Image dimension {original.shape} too small for SSIM with default window")
        # Return a default value if image is too small
        return 0.0
    
    # Use a smaller window size if needed
    win_size = min(7, min_dim - (min_dim % 2) - 1)
    if win_size < 7:
        print(f"Using smaller window size of {win_size} for SSIM calculation")
    
    # Multichannel is now called channel_axis in newer versions
    try:
        if len(original.shape) > 2:
            # For RGB images, set channel_axis to -1 (last dimension)
            return ssim(original, decoded, data_range=255, win_size=win_size, channel_axis=-1)
        else:
            # For grayscale images
            return ssim(original, decoded, data_range=255, win_size=win_size, channel_axis=None)
    except ValueError as e:
        print(f"SSIM calculation failed: {e}")
        return 0.0  # Fallback value
    
def visual_assessment(original_path, decoded_image, output_path=None):
    """Perform visual assessment between original and decoded images"""
    # Load original image
    original_img = Image.open(original_path)
    original_array = np.array(original_img)
    
    # Convert decoded image to array if it's not already
    if isinstance(decoded_image, Image.Image):
        decoded_array = np.array(decoded_image)
    else:
        decoded_array = decoded_image
    
    # Handle dimension mismatch (grayscale vs RGB)
    if len(original_array.shape) != len(decoded_array.shape):
        print(f"Adjusting dimensions for {os.path.basename(original_path)}")
        
        if len(original_array.shape) == 2 and len(decoded_array.shape) == 3:
            # Original is grayscale, decoded is RGB - convert original to RGB
            if original_array.dtype != np.uint8:
                original_array = (original_array * 255).astype(np.uint8)
            original_array = np.stack([original_array] * 3, axis=2)
            
        elif len(original_array.shape) == 3 and len(decoded_array.shape) == 2:
            # Original is RGB, decoded is grayscale - convert decoded to RGB
            if decoded_array.dtype != np.uint8:
                decoded_array = (decoded_array * 255).astype(np.uint8)
            decoded_array = np.stack([decoded_array] * 3, axis=2)
    
    # Verify dimensions now match
    if original_array.shape != decoded_array.shape:
        print(f"WARNING: After conversion, shapes still don't match: {original_array.shape} vs {decoded_array.shape}")
        
        # As a fallback for measurement, convert grayscale to luminance if needed
        if len(original_array.shape) == 3 and original_array.shape[2] == 3 and len(decoded_array.shape) == 2:
            # Convert RGB original to grayscale for comparison
            original_gray = np.mean(original_array, axis=2).astype(np.uint8)
            original_array = original_gray
        elif len(decoded_array.shape) == 3 and decoded_array.shape[2] == 3 and len(original_array.shape) == 2:
            # Convert RGB decoded to grayscale for comparison
            decoded_gray = np.mean(decoded_array, axis=2).astype(np.uint8)
            decoded_array = decoded_gray
    
    # Calculate metrics (will use the appropriate version)
    is_multichannel = len(original_array.shape) > 2
    psnr_value = calculate_psnr(original_array, decoded_array)
    ssim_value = calculate_ssim(original_array, decoded_array)
    
    # Create visual comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(original_array, cmap='gray' if len(original_array.shape) == 2 else None)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Decoded image
    axes[1].imshow(decoded_array, cmap='gray' if len(decoded_array.shape) == 2 else None)
    axes[1].set_title('Decoded')
    axes[1].axis('off')
    
    # For difference image, ensure dimensions match
    if original_array.shape != decoded_array.shape:
        print(f"WARNING: Can't create difference map due to mismatched shapes")
        axes[2].text(0.5, 0.5, "Difference unavailable\ndue to shape mismatch", 
                    ha='center', va='center', transform=axes[2].transAxes)
    else:
        # Difference image
        difference = np.abs(original_array.astype(float) - decoded_array.astype(float))
        # Normalize difference for better visibility
        if np.max(difference) > 0:
            difference = difference / np.max(difference) * 255
        
        axes[2].imshow(difference, cmap='hot')
        axes[2].set_title('Difference Map')
    
    axes[2].axis('off')
    
    plt.suptitle(f'PSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.4f}')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()
    
    plt.close()
    
    return {
        "psnr": psnr_value,
        "ssim": ssim_value
    }

def batch_visual_assessment(results, viewer_cls):
    """Evaluate visual quality for a batch of compression results"""
    quality_metrics = []
    
    for result in results:
        input_path = result["input_path"]
        output_path = result["output_path"]
        block_size = result["block_size"]
        
        # Create a viewer to decode the entire image
        viewer = viewer_cls(output_path)
        
        try:
            decoded_img = viewer.render_full_image()
            
            # Check if the decoded image is valid
            if decoded_img.size == 0 or decoded_img.shape[0] < 7 or decoded_img.shape[1] < 7:
                print(f"WARNING: Decoded image for {output_path} is too small: {decoded_img.shape}")
                continue
                
            # Base name for output
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            
            # Calculate metrics and save comparison
            metrics_path = f"results/visual_quality_{base_name}_b{block_size}.png"
            
            try:
                metrics = visual_assessment(input_path, decoded_img, metrics_path)
                
                # Store results
                quality_result = {
                    "map": os.path.basename(input_path),
                    "block_size": block_size,
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "compression_ratio": result["compression_ratio"]
                }
                
                quality_metrics.append(quality_result)
                
            except Exception as e:
                print(f"Error assessing visual quality for {input_path} with block size {block_size}: {e}")
        
        except Exception as e:
            print(f"Error rendering full image for {output_path}: {e}")
    
    return quality_metrics

# -------------------------------------------------------------------------
# Statistical Analysis
# -------------------------------------------------------------------------
def json_safe(obj):
    """Convert non-serializable objects to serializable ones"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def analyze_results(results_file="results/experiment_results.json"):
    """Extended analysis of experiment results with statistical tests"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame([
        {
            'map': os.path.basename(r['input_path']),
            'block_size': r['block_size'],
            'compression_ratio': r['compression_ratio'],
            'avg_decode_time': r['avg_decode_time'] * 1000,  # Convert to ms
            'compressed_size': r['compressed_size'],
            'original_size': r['original_size'],
            'compression_time': r['compression_time'],
            'psnr': r.get('psnr', np.nan),
            'ssim': r.get('ssim', np.nan),
            'peak_memory_mb': r.get('memory_profile', {}).get('peak_memory_mb', np.nan),
            'memory_increase_mb': r.get('memory_profile', {}).get('memory_increase_mb', np.nan)
        }
        for r in results
    ])
    
    # Create comprehensive visualizations
    plt.figure(figsize=(15, 12))
    
    # 1. Compression ratio vs block size
    plt.subplot(3, 2, 1)
    sns.barplot(x='block_size', y='compression_ratio', hue='map', data=df)
    plt.title('Compression Ratio vs Block Size')
    plt.xlabel('Block Size (px)')
    plt.ylabel('Compression Ratio (higher is better)')
    
    # 2. Decode time vs block size
    plt.subplot(3, 2, 2)
    sns.barplot(x='block_size', y='avg_decode_time', hue='map', data=df)
    plt.title('Average Decode Time vs Block Size')
    plt.xlabel('Block Size (px)')
    plt.ylabel('Decode Time (ms)')
    
    # 3. Memory usage vs block size
    if not df['peak_memory_mb'].isna().all():
        plt.subplot(3, 2, 3)
        sns.barplot(x='block_size', y='peak_memory_mb', hue='map', data=df)
        plt.title('Peak Memory Usage vs Block Size')
        plt.xlabel('Block Size (px)')
        plt.ylabel('Memory Usage (MB)')
    
    # 4. PSNR vs block size
    if not df['psnr'].isna().all():
        plt.subplot(3, 2, 4)
        sns.barplot(x='block_size', y='psnr', hue='map', data=df)
        plt.title('Image Quality (PSNR) vs Block Size')
        plt.xlabel('Block Size (px)')
        plt.ylabel('PSNR (dB)')
    
    # 5. Create a trade-off plot
    plt.subplot(3, 2, 5)
    for map_name in df['map'].unique():
        map_df = df[df['map'] == map_name]
        plt.plot(map_df['compression_ratio'], map_df['avg_decode_time'], 'o-', label=map_name)
        # Annotate points with block size
        for _, row in map_df.iterrows():
            plt.annotate(f"{int(row['block_size'])}px", 
                        (row['compression_ratio'], row['avg_decode_time']),
                        xytext=(5, 5), textcoords='offset points')
    
    plt.title('Trade-off: Compression Ratio vs Decode Time')
    plt.xlabel('Compression Ratio (higher is better)')
    plt.ylabel('Decode Time (ms) (lower is better)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 6. Quality vs Compression trade-off
    if not df['psnr'].isna().all():
        plt.subplot(3, 2, 6)
        for map_name in df['map'].unique():
            map_df = df[df['map'] == map_name]
            plt.plot(map_df['compression_ratio'], map_df['psnr'], 'o-', label=map_name)
            # Annotate points with block size
            for _, row in map_df.iterrows():
                plt.annotate(f"{int(row['block_size'])}px", 
                            (row['compression_ratio'], row['psnr']),
                            xytext=(5, 5), textcoords='offset points')
        
        plt.title('Trade-off: Compression Ratio vs Image Quality')
        plt.xlabel('Compression Ratio (higher is better)')
        plt.ylabel('PSNR (dB) (higher is better)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/comprehensive_analysis.png', dpi=300)
    plt.close()
    
    # Statistical analysis
    stats_results = {}
    
    # ANOVA test for block size effect on decode time
    maps = df['map'].unique()
    for map_name in maps:
        map_df = df[df['map'] == map_name]
        if len(map_df) > 2:  # Need at least 3 points for ANOVA
            groups = [map_df[map_df['block_size'] == size]['avg_decode_time'].values 
                     for size in map_df['block_size'].unique()]
            try:
                f_val, p_val = stats.f_oneway(*groups)
                stats_results[f"{map_name}_block_size_anova"] = {
                    "f_value": f_val,
                    "p_value": p_val,
                    "significant": p_val < 0.05
                }
            except ValueError:
                print(f"Warning: Could not perform ANOVA for {map_name} (insufficient data variation)")
    
    # Correlation tests
    corr_metrics = {}
    for map_name in maps:
        map_df = df[df['map'] == map_name]
        if len(map_df) < 2:
            continue
            
        # Correlation: block size vs compression ratio
        try:
            corr, p = stats.pearsonr(map_df['block_size'], map_df['compression_ratio'])
            corr_metrics[f"{map_name}_blocksize_vs_compression"] = {
                "correlation": corr,
                "p_value": p,
                "significant": p < 0.05
            }
        except ValueError:
            pass
            
        # Correlation: block size vs decode time
        try:
            corr, p = stats.pearsonr(map_df['block_size'], map_df['avg_decode_time'])
            corr_metrics[f"{map_name}_blocksize_vs_decodetime"] = {
                "correlation": corr,
                "p_value": p,
                "significant": p < 0.05
            }
        except ValueError:
            pass
    
    stats_results["correlations"] = corr_metrics
    
    # Convert any NumPy types before saving
    def sanitize_dict(d):
        """Recursively sanitize a dictionary for JSON serialization"""
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = sanitize_dict(v)
            else:
                result[k] = json_safe(v)
        return result
    
    # Sanitize the stats_results before saving
    stats_results = sanitize_dict(stats_results)
    
    # Save statistics
    with open('results/statistical_analysis.json', 'w') as f:
        json.dump(stats_results, f, indent=2)
    
    # Create detailed per-map analysis
    for map_name in df['map'].unique():
        map_df = df[df['map'] == map_name]
        
        plt.figure(figsize=(15, 10))
        
        # Extract pan results for this map
        pan_data = {}
        for result in results:
            if os.path.basename(result['input_path']) == map_name:
                block_size = result['block_size']
                if 'pan_results' in result:
                    pan_data[block_size] = result['pan_results']
        
        # Plot panning performance over steps
        plt.subplot(2, 2, 1)
        for block_size, steps in pan_data.items():
            times = [step['decode_time'] * 1000 for step in steps]  # Convert to ms
            plt.plot(range(len(times)), times, label=f'{block_size}px blocks')
        
        plt.title(f'Panning Performance: {map_name}')
        plt.xlabel('Pan Step')
        plt.ylabel('Decode Time (ms)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Plot memory usage if available
        map_memory = map_df[~map_df['peak_memory_mb'].isna()]
        if len(map_memory) > 0:
            plt.subplot(2, 2, 2)
            sns.barplot(x='block_size', y='peak_memory_mb', data=map_memory)
            plt.title(f'Memory Usage: {map_name}')
            plt.xlabel('Block Size (px)')
            plt.ylabel('Peak Memory (MB)')
            plt.grid(True, linestyle='--', alpha=0.5)
        
        # Block distribution
        plt.subplot(2, 2, 3)
        for block_size, steps in pan_data.items():
            if steps and 'blocks_decoded' in steps[0]:
                blocks_decoded = [step['blocks_decoded'] for step in steps]
                plt.plot(range(len(blocks_decoded)), blocks_decoded, label=f'{block_size}px blocks')
        
        plt.title(f'Blocks Decoded per Step: {map_name}')
        plt.xlabel('Pan Step')
        plt.ylabel('Number of Blocks')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Quality metrics if available
        map_quality = map_df[~map_df['psnr'].isna()]
        if len(map_quality) > 0:
            plt.subplot(2, 2, 4)
            plt.scatter(map_quality['block_size'], map_quality['psnr'], s=100)
            plt.plot(map_quality['block_size'], map_quality['psnr'], 'o-')
            
            # Annotate SSIM values
            for _, row in map_quality.iterrows():
                plt.annotate(f"SSIM: {row['ssim']:.4f}", 
                            (row['block_size'], row['psnr']),
                            xytext=(5, -15), textcoords='offset points')
            
            plt.title(f'Image Quality: {map_name}')
            plt.xlabel('Block Size (px)')
            plt.ylabel('PSNR (dB)')
            plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f'results/detailed_analysis_{map_name}.png', dpi=300)
        plt.close()
    
    return df, stats_results

def print_statistical_summary(stats_results):
    """Print a human-readable summary of statistical analysis"""
    print("\nStatistical Analysis Summary:")
    print("============================")
    
    # ANOVA results
    anova_tests = [k for k in stats_results.keys() if k != "correlations" and k.endswith("_anova")]
    if anova_tests:
        print("\nANOVA Tests (Effect of block size on decode time):")
        for test in anova_tests:
            map_name = test.split('_block_size_anova')[0]
            result = stats_results[test]
            significance = "Significant" if result["significant"] else "Not significant"
            print(f"  {map_name}: F={result['f_value']:.2f}, p={result['p_value']:.4f} - {significance}")
    
    # Correlation results
    if "correlations" in stats_results:
        corr = stats_results["correlations"]
        
        print("\nCorrelations:")
        for metric, result in corr.items():
            parts = metric.split('_')
            map_name = parts[0]
            comparison = '_'.join(parts[1:])
            
            strength = "Strong" if abs(result["correlation"]) > 0.7 else "Moderate" if abs(result["correlation"]) > 0.4 else "Weak"
            direction = "positive" if result["correlation"] > 0 else "negative"
            significance = "Significant" if result["significant"] else "Not significant"
            
            print(f"  {map_name}, {comparison}: r={result['correlation']:.2f} ({strength} {direction}, p={result['p_value']:.4f}) - {significance}")