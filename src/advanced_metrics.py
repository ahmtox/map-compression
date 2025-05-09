"""Advanced metrics and comparative analysis for map compression research."""
import os
import time
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageSequence
from io import BytesIO
import requests
from collections import defaultdict
import subprocess

# For caching efficiency metrics
class CacheEfficiencyTracker:
    """Track block cache hits/misses during panning operations"""
    
    def __init__(self):
        self.cache = {}
        self.hits = 0
        self.misses = 0
        self.hit_history = []
        self.miss_history = []
        self.cache_size_history = []
        
    def access(self, block_key):
        """Record a block access, tracking whether it was a hit or miss"""
        if block_key in self.cache:
            self.hits += 1
            self.cache[block_key] += 1
        else:
            self.misses += 1
            self.cache[block_key] = 1
        
        # Record current stats
        self.hit_history.append(self.hits)
        self.miss_history.append(self.misses)
        self.cache_size_history.append(len(self.cache))
        
    def get_stats(self):
        """Get cache efficiency statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_accesses": total,
            "hit_rate": hit_rate,
            "unique_blocks": len(self.cache),
            "hit_history": self.hit_history,
            "miss_history": self.miss_history,
            "cache_size_history": self.cache_size_history
        }
    
    def plot_efficiency(self, title="Cache Efficiency"):
        """Plot cache efficiency metrics"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Hit rate over time
        total_history = [h + m for h, m in zip(self.hit_history, self.miss_history)]
        hit_rate_history = [h/t if t > 0 else 0 for h, t in zip(self.hit_history, total_history)]
        
        ax1.plot(hit_rate_history, label='Hit Rate')
        ax1.set_title('Cache Hit Rate Over Time')
        ax1.set_xlabel('Access Number')
        ax1.set_ylabel('Hit Rate')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # Cache size and hits/misses
        ax2.plot(self.cache_size_history, label='Cache Size (Unique Blocks)')
        ax2.set_title('Cache Size and Hits/Misses')
        ax2.set_xlabel('Access Number')
        ax2.set_ylabel('Count')
        ax2.plot(self.hit_history, label='Cumulative Hits')
        ax2.plot(self.miss_history, label='Cumulative Misses')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        
        return fig

# Network simulation
class NetworkSimulator:
    """Simulate network conditions for streaming scenarios"""
    
    def __init__(self, bandwidth_mbps=10, latency_ms=50, jitter_ms=10, packet_loss=0.01):
        """Initialize network simulator with specified parameters
        
        Args:
            bandwidth_mbps: Bandwidth in Mbps
            latency_ms: Base latency in milliseconds  
            jitter_ms: Random jitter in milliseconds
            packet_loss: Probability of packet loss (0-1)
        """
        self.bandwidth_mbps = bandwidth_mbps
        self.latency_ms = latency_ms
        self.jitter_ms = jitter_ms
        self.packet_loss = packet_loss
        
    def simulate_transfer(self, size_bytes):
        """Simulate transfer of data and return statistics
        
        Args:
            size_bytes: Size of data in bytes
            
        Returns:
            dict: Transfer statistics including time and effective bandwidth
        """
        # Convert size to bits
        size_bits = size_bytes * 8
        
        # Calculate base transfer time (seconds)
        bandwidth_bps = self.bandwidth_mbps * 1_000_000
        transfer_time = size_bits / bandwidth_bps
        
        # Add latency and jitter
        actual_latency = self.latency_ms / 1000
        actual_jitter = (random.random() * 2 - 1) * self.jitter_ms / 1000  # Random jitter between -jitter and +jitter
        
        # Simulate packet loss - increase transfer time if loss occurs
        if random.random() < self.packet_loss:
            # Simple model: packet loss adds latency due to retransmission
            transfer_time += actual_latency * 2  # Assume one retransmission
        
        # Calculate total time
        total_time = transfer_time + actual_latency + actual_jitter
        
        # Calculate effective bandwidth
        effective_bandwidth_mbps = (size_bytes * 8) / (total_time * 1_000_000) if total_time > 0 else 0
        
        return {
            "size_bytes": size_bytes,
            "transfer_time_sec": total_time,
            "effective_bandwidth_mbps": effective_bandwidth_mbps,
            "configured_bandwidth_mbps": self.bandwidth_mbps,
            "latency_ms": self.latency_ms + (actual_jitter * 1000),
            "packet_loss_occurred": random.random() < self.packet_loss
        }

# Multiple trials support
def run_multiple_trials(experiment_func, num_trials=5, **kwargs):
    """Run multiple trials of an experiment to get statistically significant results"""
    all_results = []
    
    for trial in range(1, num_trials + 1):
        print(f"\nRunning trial {trial}/{num_trials}...")
        result = experiment_func(**kwargs)
        
        # Add trial number
        if isinstance(result, dict):
            result["trial_number"] = trial
        elif isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    item["trial_number"] = trial
        
        all_results.append(result)
    
    return all_results

# Standard format comparison
def compare_with_standard_formats(input_path, output_dir="results/format_comparison"):
    """Compare with standard image formats like PNG, WebP, etc."""
    os.makedirs(output_dir, exist_ok=True)
    
    image = Image.open(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # Save in different formats
    formats = {
        "png": {"format": "PNG", "params": {}},
        "jpg": {"format": "JPEG", "params": {"quality": 90}},
        "webp": {"format": "WEBP", "params": {"quality": 90, "method": 6}},
        "webp_lossless": {"format": "WEBP", "params": {"lossless": True, "method": 6}}
    }
    
    # Try to use pillow-avif if available - with better error handling
    have_avif = False
    try:
        # Try different import approaches
        try:
            from pillow_avif import AvifImagePlugin
            have_avif = True
        except ImportError:
            try:
                from PIL import AvifImagePlugin
                have_avif = True
            except ImportError:
                have_avif = False
        
        if have_avif:
            # Test if AVIF actually works by encoding and decoding a small image
            test_img = Image.new('RGB', (10, 10))
            test_buffer = BytesIO()
            try:
                test_img.save(test_buffer, format="AVIF")
                Image.open(test_buffer)
                # If we got here, AVIF works
                formats["avif"] = {"format": "AVIF", "params": {"quality": 90}}
            except Exception:
                print("AVIF support available but not working properly. Skipping AVIF comparison.")
        else:
            print("AVIF support not available. Install pillow-avif for AVIF comparison.")
    except Exception as e:
        print(f"Error testing AVIF support: {e}. Skipping AVIF comparison.")
    
    results = {}
    
    for format_name, config in formats.items():
        output_path = f"{output_dir}/{base_name}.{format_name}"
        
        try:
            # Time the encoding
            start_time = time.time()
            image.save(output_path, **config)
            encoding_time = time.time() - start_time
            
            # Get file size
            file_size = os.path.getsize(output_path)
            original_size = os.path.getsize(input_path)
            
            # Time the decoding
            start_time = time.time()
            Image.open(output_path).load()  # Force decoding
            decoding_time = time.time() - start_time
            
            results[format_name] = {
                "file_size_bytes": file_size,
                "compression_ratio": original_size / file_size,
                "encoding_time_sec": encoding_time,
                "decoding_time_sec": decoding_time,
                "format": format_name,
            }
        except Exception as e:
            print(f"Error processing {format_name} format for {base_name}: {e}")
    
    # Create comparison chart
    plt.figure(figsize=(12, 10))
    
    # Compare file sizes
    plt.subplot(2, 2, 1)
    formats_list = list(results.keys())
    sizes = [results[f]["file_size_bytes"] / 1024 for f in formats_list]
    plt.bar(formats_list, sizes)
    plt.title('File Size Comparison')
    plt.ylabel('Size (KB)')
    plt.xticks(rotation=45)
    
    # Compare compression ratios
    plt.subplot(2, 2, 2)
    ratios = [results[f]["compression_ratio"] for f in formats_list]
    plt.bar(formats_list, ratios)
    plt.title('Compression Ratio Comparison')
    plt.ylabel('Compression Ratio (higher is better)')
    plt.xticks(rotation=45)
    
    # Compare decode times
    plt.subplot(2, 2, 3)
    decode_times = [results[f]["decoding_time_sec"] * 1000 for f in formats_list]
    plt.bar(formats_list, decode_times)
    plt.title('Decode Time Comparison')
    plt.ylabel('Time (ms)')
    plt.xticks(rotation=45)
    
    # Compare encode times
    plt.subplot(2, 2, 4)
    encode_times = [results[f]["encoding_time_sec"] * 1000 for f in formats_list]
    plt.bar(formats_list, encode_times)
    plt.title('Encode Time Comparison')
    plt.ylabel('Time (ms)')
    plt.xticks(rotation=45)
    
    plt.suptitle(f'Standard Format Comparison - {base_name}')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{base_name}_format_comparison.png", dpi=300)
    plt.close()
    
    # Save detailed results
    with open(f"{output_dir}/{base_name}_format_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

# Analyze 128px block size anomaly
def analyze_block_size_anomaly(results_file="results/experiment_results.json", focus_size=128):
    """Specifically analyze the anomalous behavior at 128px block size"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract relevant metrics by block size
    block_stats = defaultdict(lambda: defaultdict(list))
    
    for result in results:
        block_size = result["block_size"]
        map_name = os.path.basename(result["input_path"])
        
        # Collect metrics
        block_stats[map_name]["compression_ratio"].append((block_size, result["compression_ratio"]))
        block_stats[map_name]["avg_decode_time"].append((block_size, result["avg_decode_time"] * 1000))  # ms
        
        if "memory_profile" in result:
            block_stats[map_name]["peak_memory"].append(
                (block_size, result["memory_profile"]["peak_memory_mb"])
            )
    
    # Create anomaly analysis
    plt.figure(figsize=(15, 12))
    
    # Focus on the trend around the anomaly point
    for i, (metric, title, ylabel) in enumerate([
        ("compression_ratio", "Compression Ratio", "Ratio"),
        ("avg_decode_time", "Decode Time", "Time (ms)"),
        ("peak_memory", "Peak Memory", "Memory (MB)")
    ]):
        plt.subplot(3, 1, i+1)
        
        for map_name, data in block_stats.items():
            if metric in data:
                # Sort by block size
                sorted_data = sorted(data[metric], key=lambda x: x[0])
                x = [point[0] for point in sorted_data]
                y = [point[1] for point in sorted_data]
                
                # Plot the data
                plt.plot(x, y, 'o-', label=map_name)
                
                # Add vertical line at focus size
                if focus_size in x:
                    plt.axvline(x=focus_size, color='r', linestyle='--', alpha=0.5)
                
                # Add trend analysis
                try:
                    # Fit polynomial curve
                    z = np.polyfit(x, y, 2)
                    p = np.poly1d(z)
                    
                    # Generate smooth curve
                    x_smooth = np.linspace(min(x), max(x), 100)
                    plt.plot(x_smooth, p(x_smooth), '--', alpha=0.5)
                    
                    # Calculate relative efficiency at focus point
                    focus_idx = x.index(focus_size) if focus_size in x else -1
                    if focus_idx >= 0:
                        expected = p(focus_size)
                        actual = y[focus_idx]
                        anomaly = (actual - expected) / expected * 100
                        
                        # Annotate the anomaly
                        plt.annotate(f"{anomaly:.1f}% {'above' if anomaly > 0 else 'below'} trend",
                                    (focus_size, y[focus_idx]),
                                    xytext=(10, 10),
                                    textcoords='offset points',
                                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
                except:
                    # Not enough data points for polynomial fit
                    pass
        
        plt.title(f"{title} vs Block Size (Focus on {focus_size}px)")
        plt.xlabel('Block Size (px)')
        plt.ylabel(ylabel)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"results/block_size_anomaly_analysis_{focus_size}px.png", dpi=300)
    plt.close()
    
    # Simple statistical test to determine if there's a significant anomaly
    anomaly_stats = {}
    
    for map_name, data in block_stats.items():
        anomaly_stats[map_name] = {}
        
        for metric, values in data.items():
            # Sort by block size
            sorted_data = sorted(values, key=lambda x: x[0])
            x = [point[0] for point in sorted_data]
            y = [point[1] for point in sorted_data]
            
            if focus_size in x and len(x) >= 3:
                # Find focus index
                focus_idx = x.index(focus_size)
                
                # Check if it's a local minimum or maximum
                if focus_idx > 0 and focus_idx < len(x) - 1:
                    is_local_min = y[focus_idx] < y[focus_idx-1] and y[focus_idx] < y[focus_idx+1]
                    is_local_max = y[focus_idx] > y[focus_idx-1] and y[focus_idx] > y[focus_idx+1]
                    
                    # Calculate percent difference from neighbors
                    avg_neighbors = (y[focus_idx-1] + y[focus_idx+1]) / 2
                    percent_diff = (y[focus_idx] - avg_neighbors) / avg_neighbors * 100 if avg_neighbors != 0 else 0
                    
                    anomaly_stats[map_name][metric] = {
                        "is_local_minimum": is_local_min,
                        "is_local_maximum": is_local_max,
                        "percent_diff_from_neighbors": percent_diff,
                        "value": y[focus_idx],
                        "neighbor_values": [y[focus_idx-1], y[focus_idx+1]]
                    }
    
    # Save analysis
    with open(f"results/block_size_anomaly_stats_{focus_size}px.json", "w") as f:
        json.dump(anomaly_stats, f, indent=2)
    
    return anomaly_stats

# Simulated network performance
def simulate_network_performance(results_file="results/experiment_results.json", 
                                 network_profiles=None):
    """Test performance under different network conditions"""
    if network_profiles is None:
        network_profiles = [
            {"name": "5G", "bandwidth_mbps": 100, "latency_ms": 20, "jitter_ms": 5, "packet_loss": 0.001},
            {"name": "4G", "bandwidth_mbps": 20, "latency_ms": 50, "jitter_ms": 15, "packet_loss": 0.01},
            {"name": "3G", "bandwidth_mbps": 2, "latency_ms": 100, "jitter_ms": 40, "packet_loss": 0.05},
            {"name": "Satellite", "bandwidth_mbps": 5, "latency_ms": 600, "jitter_ms": 100, "packet_loss": 0.02}
        ]
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    network_results = []
    
    for profile in network_profiles:
        simulator = NetworkSimulator(
            bandwidth_mbps=profile["bandwidth_mbps"],
            latency_ms=profile["latency_ms"],
            jitter_ms=profile["jitter_ms"],
            packet_loss=profile["packet_loss"]
        )
        
        profile_results = []
        
        for result in results:
            # Get compressed size
            compressed_size = result["compressed_size"]
            
            # Simulate network transfer
            transfer_stats = simulator.simulate_transfer(compressed_size)
            
            # Add transfer time to decode time for total user-perceived latency
            decode_time = result["avg_decode_time"]
            total_latency = transfer_stats["transfer_time_sec"] + decode_time
            
            profile_results.append({
                "map": os.path.basename(result["input_path"]),
                "block_size": result["block_size"],
                "compressed_size": compressed_size,
                "transfer_time_sec": transfer_stats["transfer_time_sec"],
                "decode_time_sec": decode_time,
                "total_latency_sec": total_latency,
                "effective_bandwidth_mbps": transfer_stats["effective_bandwidth_mbps"]
            })
        
        network_results.append({
            "profile": profile,
            "results": profile_results
        })
    
    # Create visualization
    plt.figure(figsize=(16, 12))
    
    # Prepare data for plotting
    # Group by map and block size, then collect latencies
    latency_data = defaultdict(lambda: defaultdict(list))
    transfer_time_data = defaultdict(lambda: defaultdict(list))
    
    for network in network_results:
        profile_name = network["profile"]["name"]
        
        for result in network["results"]:
            key = f"{result['map']}-{result['block_size']}"
            latency_data[profile_name][key] = result["total_latency_sec"] * 1000  # ms
            transfer_time_data[profile_name][key] = result["transfer_time_sec"] * 1000  # ms
    
    # Plot total latency
    plt.subplot(2, 1, 1)
    
    x_labels = []
    grouped_latencies = []
    
    for network in network_profiles:
        profile_name = network["name"]
        values = list(latency_data[profile_name].values())
        grouped_latencies.append(values)
        x_labels = list(latency_data[profile_name].keys())
    
    # Plot grouped bar chart
    x = np.arange(len(x_labels))
    width = 0.8 / len(network_profiles)
    
    for i, (profile, latencies) in enumerate(zip(network_profiles, grouped_latencies)):
        offset = (i - len(network_profiles)/2 + 0.5) * width
        plt.bar(x + offset, latencies, width, label=profile["name"])
    
    plt.title("Total Latency (Transfer + Decode) by Network Profile")
    plt.xlabel("Map-Block Size")
    plt.ylabel("Latency (ms)")
    plt.xticks(x, x_labels, rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot latency breakdown for one specific configuration
    plt.subplot(2, 1, 2)
    
    # Pick a representative map and block size
    if x_labels:
        selected_key = x_labels[0]
        
        transfer_times = []
        decode_times = []
        profile_names = []
        
        for network in network_results:
            profile_name = network["profile"]["name"]
            for result in network["results"]:
                key = f"{result['map']}-{result['block_size']}"
                if key == selected_key:
                    transfer_times.append(result["transfer_time_sec"] * 1000)  # ms
                    decode_times.append(result["decode_time_sec"] * 1000)  # ms
                    profile_names.append(profile_name)
                    break
        
        # Plot stacked bar
        x = np.arange(len(profile_names))
        plt.bar(x, transfer_times, label='Transfer Time')
        plt.bar(x, decode_times, bottom=transfer_times, label='Decode Time')
        
        plt.title(f"Latency Breakdown for {selected_key}")
        plt.xlabel("Network Profile")
        plt.ylabel("Time (ms)")
        plt.xticks(x, profile_names)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("results/network_simulation_results.png", dpi=300)
    plt.close()
    
    # Save detailed results
    with open("results/network_simulation_results.json", "w") as f:
        json.dump(network_results, f, indent=2)
    
    return network_results

# Enhanced run_experiment function with extended metrics
def run_enhanced_experiment(map_paths, block_sizes=[64, 96, 128, 192, 256], trials=5, use_arithmetic=False):
    """Run comprehensive experiment with multiple trials and enhanced metrics"""
    from src.map_compression_tool import run_experiment
    
    all_results = []
    
    # Add standard format comparison as baseline
    format_baselines = {}
    for input_path in map_paths:
        try:
            format_baselines[os.path.basename(input_path)] = compare_with_standard_formats(input_path)
        except Exception as e:
            print(f"Error comparing standard formats for {input_path}: {e}")
    
    # Run trials
    for trial in range(1, trials + 1):
        print(f"\n=== Running Trial {trial}/{trials} ===")
        
        # Run the experiment with this trial's block sizes
        trial_results, results_file = run_experiment(map_paths, block_sizes, use_arithmetic)
        
        # Add trial number to each result
        for result in trial_results:
            result["trial_number"] = trial
        
        all_results.extend(trial_results)
    
    # Save all results
    results_file = "results/enhanced_experiment_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Analyze anomalies - focus on intermediate block sizes
    for focus_size in [96, 128, 192]:
        try:
            analyze_block_size_anomaly(results_file, focus_size)
        except Exception as e:
            print(f"Error analyzing anomaly for block size {focus_size}: {e}")
    
    # Run network simulation
    try:
        simulate_network_performance(results_file)
    except Exception as e:
        print(f"Error running network simulation: {e}")
    
    # Analyze results statistically
    from src.metrics import analyze_results, print_statistical_summary
    df, stats = analyze_results(results_file)
    print_statistical_summary(stats)
    
    # Create enhanced visualization
    create_enhanced_visualizations(results_file, format_baselines)
    
    return all_results

# Enhanced visualizations for paper-quality figures
def create_enhanced_visualizations(results_file, format_baselines=None):
    """Create higher-quality visualizations for publication"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'map': os.path.basename(r['input_path']),
            'block_size': r['block_size'],
            'trial': r.get('trial_number', 1),
            'compression_ratio': r['compression_ratio'],
            'avg_decode_time': r['avg_decode_time'] * 1000,  # Convert to ms
            'compressed_size': r['compressed_size'],
            'original_size': r['original_size'],
            'psnr': r.get('psnr', np.nan),
            'ssim': r.get('ssim', np.nan),
            'peak_memory_mb': r.get('memory_profile', {}).get('peak_memory_mb', np.nan)
        }
        for r in results
    ])
    
    # Set plotting style for publication-quality figures
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Create a comprehensive comparison across block sizes with error bars
    plt.figure(figsize=(15, 10))
    
    # Group by map and block size, compute means and std devs
    means = df.groupby(['map', 'block_size']).agg({
        'compression_ratio': 'mean',
        'avg_decode_time': 'mean',
        'peak_memory_mb': 'mean',
        'psnr': 'mean',
        'ssim': 'mean'
    }).reset_index()
    
    errors = df.groupby(['map', 'block_size']).agg({
        'compression_ratio': 'std',
        'avg_decode_time': 'std',
        'peak_memory_mb': 'std',
        'psnr': 'std',
        'ssim': 'std'
    }).reset_index()
    
    # Fill NaN with 0 for error bars
    errors = errors.fillna(0)
    
    # For each map, create a subplot with error bars
    map_names = df['map'].unique()
    
    for i, metric in enumerate([
        ('compression_ratio', 'Compression Ratio'),
        ('avg_decode_time', 'Decode Time (ms)'),
        ('peak_memory_mb', 'Memory Usage (MB)')
    ]):
        col, title = metric
        plt.subplot(2, 2, i+1)
        
        for map_name in map_names:
            map_means = means[means['map'] == map_name]
            map_errors = errors[errors['map'] == map_name]
            
            x = map_means['block_size']
            y = map_means[col]
            err = map_errors[col]
            
            plt.errorbar(x, y, yerr=err, marker='o', label=map_name)
        
        plt.title(title)
        plt.xlabel('Block Size (px)')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    # Add a quality metric (SSIM)
    if not df['ssim'].isna().all():
        plt.subplot(2, 2, 4)
        
        for map_name in map_names:
            map_means = means[means['map'] == map_name]
            map_errors = errors[errors['map'] == map_name]
            
            x = map_means['block_size']
            y = map_means['ssim']
            err = map_errors['ssim']
            
            plt.errorbar(x, y, yerr=err, marker='o', label=map_name)
        
        plt.title('Image Quality (SSIM)')
        plt.xlabel('Block Size (px)')
        plt.ylabel('SSIM (higher is better)')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/enhanced_block_size_comparison.png', dpi=300)
    plt.close()
    
    # 2. Create optimal block size analysis
    plt.figure(figsize=(12, 10))
    
    # Create a scoring system combining metrics (normalized)
    # Higher score is better
    means['decode_time_norm'] = 1 - (means['avg_decode_time'] / means['avg_decode_time'].max())
    means['memory_norm'] = 1 - (means['peak_memory_mb'] / means['peak_memory_mb'].max())
    means['compression_norm'] = means['compression_ratio'] / means['compression_ratio'].max()
    
    # Calculate composite score
    means['composite_score'] = (means['decode_time_norm'] + means['memory_norm'] + means['compression_norm']) / 3
    
    if not means['psnr'].isna().all():
        means['quality_norm'] = means['psnr'] / means['psnr'].max()
        means['composite_score_with_quality'] = (
            means['decode_time_norm'] + means['memory_norm'] + 
            means['compression_norm'] + means['quality_norm']
        ) / 4
    
    # Plot the composite score by block size
    plt.subplot(2, 1, 1)
    
    for map_name in map_names:
        map_data = means[means['map'] == map_name]
        
        plt.plot(map_data['block_size'], map_data['composite_score'], 'o-', label=map_name)
        
        # Find and annotate the optimal block size
        if len(map_data) > 0:
            best_idx = map_data['composite_score'].idxmax()
            best_block_size = map_data.loc[best_idx, 'block_size']
            best_score = map_data.loc[best_idx, 'composite_score']
            
            plt.annotate(f"Optimal: {int(best_block_size)}px", 
                        (best_block_size, best_score),
                        xytext=(10, 5),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    plt.title('Optimal Block Size Analysis')
    plt.xlabel('Block Size (px)')
    plt.ylabel('Composite Score (higher is better)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add comparison with standard formats if available
    if format_baselines:
        plt.subplot(2, 1, 2)
        
        for map_name in map_names:
            if map_name in format_baselines:
                baseline = format_baselines[map_name]
                
                # Get our method's results
                map_data = means[means['map'] == map_name]
                
                # Find best block size for this map
                if len(map_data) > 0:
                    best_idx = map_data['composite_score'].idxmax()
                    best_block = map_data.loc[best_idx]
                    
                    # Gather comparison data
                    formats = list(baseline.keys())
                    compression_ratios = [baseline[f]['compression_ratio'] for f in formats]
                    decode_times = [baseline[f]['decoding_time_sec'] * 1000 for f in formats]
                    
                    # Add our method
                    formats.append(f"Our Method ({int(best_block['block_size'])}px)")
                    compression_ratios.append(best_block['compression_ratio'])
                    decode_times.append(best_block['avg_decode_time'])
                    
                    # Plot compression vs decode time
                    for i, fmt in enumerate(formats):
                        plt.scatter(compression_ratios[i], decode_times[i], 
                                   s=100, label=fmt if 'Our Method' in fmt else f"{fmt}")
                    
                    # Connect our method with a line
                    for i, row in map_data.iterrows():
                        plt.scatter(row['compression_ratio'], row['avg_decode_time'], 
                                   s=50, alpha=0.5)
                    
                    plt.plot(map_data['compression_ratio'], map_data['avg_decode_time'], 
                            'o--', alpha=0.5, label=f"{map_name} (all blocks)")
                    
                    # Highlight the optimal point
                    plt.scatter(best_block['compression_ratio'], best_block['avg_decode_time'], 
                               s=150, edgecolor='red', facecolor='none')
                    
                    plt.title(f"Comparison with Standard Formats: {map_name}")
                    plt.xlabel("Compression Ratio (higher is better)")
                    plt.ylabel("Decode Time (ms, lower is better)")
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    
                    # Only do one map for clarity
                    break
    
    plt.tight_layout()
    plt.savefig('results/optimal_block_size_analysis.png', dpi=300)
    plt.close()
    
    return "Enhanced visualizations created"