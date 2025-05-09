"""Viewer and rendering tools for displaying compressed maps."""
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
from src.core import BlockCompressedImage, ContextDecoder, ArithmeticContextDecoder

class MapViewer:
    """Map viewer for displaying compressed maps"""
    
    def __init__(self, compressed_file, use_arithmetic=False):
        self.image = BlockCompressedImage.load(compressed_file)
        self.decoder = ArithmeticContextDecoder() if use_arithmetic else ContextDecoder()
        self.block_size = self.image.metadata["block_size"]
        self.cache = {}  # Cache for decoded blocks
        
    def get_viewport(self, x, y, width, height, layers=None):
        """Get a viewport at the specified coordinates"""
        if layers is None:
            layers = list(self.image.metadata["layers"].keys())
        
        # Calculate which blocks we need
        start_block_y = y // self.block_size
        start_block_x = x // self.block_size
        end_block_y = (y + height - 1) // self.block_size + 1
        end_block_x = (x + width - 1) // self.block_size + 1
        
        # Create the viewport image (RGB for this example)
        viewport = np.zeros((height, width, 3), dtype=np.uint8)
        
        start_time = time.time()
        blocks_decoded = 0
        
        # For each layer, load the needed blocks
        for i, layer_name in enumerate(layers):
            layer_index = i % 3  # Map to R, G, or B channel for visualization
            
            for block_y in range(start_block_y, end_block_y):
                for block_x in range(start_block_x, end_block_x):
                    # Check cache first
                    cache_key = (layer_name, block_y, block_x)
                    if cache_key not in self.cache:
                        try:
                            if not self.image.has_block(layer_name, block_y, block_x):
                                continue
                            compressed_data = self.image.get_block(layer_name, block_y, block_x)
                            decoded_block = self.decoder.decode_block(compressed_data, self.block_size)
                            self.cache[cache_key] = decoded_block
                            blocks_decoded += 1
                        except ValueError:
                            # Block not found, use empty block
                            self.cache[cache_key] = np.zeros((self.block_size, self.block_size), dtype=np.uint8)
                    
                    # Calculate where this block goes in the viewport
                    block = self.cache[cache_key]
                    block_pos_y = block_y * self.block_size
                    block_pos_x = block_x * self.block_size
                    
                    # Calculate offsets and sizes
                    src_y = max(0, y - block_pos_y)
                    src_x = max(0, x - block_pos_x)
                    dst_y = max(0, block_pos_y - y)
                    dst_x = max(0, block_pos_x - x)
                    
                    copy_height = min(self.block_size - src_y, height - dst_y)
                    copy_width = min(self.block_size - src_x, width - dst_x)
                    
                    if copy_height > 0 and copy_width > 0:
                        viewport[dst_y:dst_y+copy_height, dst_x:dst_x+copy_width, layer_index] = \
                            block[src_y:src_y+copy_height, src_x:src_x+copy_width]
        
        decode_time = time.time() - start_time
        if blocks_decoded > 0:
            print(f"Decoded {blocks_decoded} blocks in {decode_time:.4f} seconds")
        
        return viewport, {"blocks_decoded": blocks_decoded, "decode_time": decode_time}
    
    def render_full_image(self):
        """Render the entire map image"""
        try:
            # Get the dimensions of the first layer
            if not self.image.metadata["layers"]:
                print("WARNING: No layers found in image")
                return np.zeros((100, 100, 3), dtype=np.uint8)  # Return dummy image
                
            first_layer = list(self.image.metadata["layers"].keys())[0]
            width = self.image.metadata["layers"][first_layer]["width"]
            height = self.image.metadata["layers"][first_layer]["height"]
            
            # Render the full viewport
            viewport, _ = self.get_viewport(0, 0, width, height)
            return viewport
        except Exception as e:
            print(f"Error rendering full image: {e}")
            return np.zeros((100, 100, 3), dtype=np.uint8)  # Return dummy image on error
    
    def simulate_panning(self, start_x, start_y, steps, step_size=50, save_images=False):
        """Simulate panning across the map, measuring performance"""
        results = []
        
        for i in range(steps):
            x = start_x + i * step_size
            y = start_y
            
            viewport, stats = self.get_viewport(x, y, 800, 600)
            
            results.append({
                'step': i,
                'x': x,
                'y': y,
                'decode_time': stats['decode_time'],
                'blocks_decoded': stats['blocks_decoded']
            })
            
            # Optionally save the viewport
            if save_images:
                img = Image.fromarray(viewport)
                img.save(f"results/pan_step_{i:03d}.png")
        
        return results

class InstrumentedViewer(MapViewer):
    """Instrumented version of MapViewer to track time spent in each operation"""
    
    def __init__(self, image_path, use_arithmetic=False):
        super().__init__(image_path, use_arithmetic)
        self.instrumentation = {}
    
    def _record_time(self, operation, elapsed):
        """Record time taken for an operation"""
        if operation not in self.instrumentation:
            self.instrumentation[operation] = []
        self.instrumentation[operation].append(elapsed)
    
    def get_viewport(self, x, y, width, height, layers=None):
        """Instrumented version of get_viewport"""
        # Track overall time
        start_time = time.time()
        
        # Calculate which blocks we need to decode
        start_block_calc = time.time()
        blocks_needed = set()
        
        if layers is None:
            layers = list(self.image.metadata["layers"].keys())
        
        for layer_name in layers:
            layer_width = self.image.metadata["layers"][layer_name]["width"]
            layer_height = self.image.metadata["layers"][layer_name]["height"]
            
            # Calculate block coordinates that overlap with viewport
            start_block_x = max(0, x // self.block_size)
            start_block_y = max(0, y // self.block_size)
            end_block_x = min(layer_width // self.block_size, (x + width + self.block_size - 1) // self.block_size)
            end_block_y = min(layer_height // self.block_size, (y + height + self.block_size - 1) // self.block_size)
            
            for block_y in range(start_block_y, end_block_y):
                for block_x in range(start_block_x, end_block_x):
                    blocks_needed.add((layer_name, block_x, block_y))
        
        block_calc_time = time.time() - start_block_calc
        self._record_time("block_calculation", block_calc_time)
        
        # Initialize the result canvas
        start_init = time.time()
        result = np.zeros((height, width, 3), dtype=np.uint8)
        init_time = time.time() - start_init
        self._record_time("result_init", init_time)
        
        # Decode and combine blocks
        blocks_decoded = 0
        total_decode_time = 0
        
        for layer_name, block_x, block_y in blocks_needed:
            # Get block if it exists
            if not self.image.has_block(layer_name, block_y, block_x):
                continue
                
            start_get = time.time()
            compressed_data = self.image.get_block(layer_name, block_y, block_x)
            get_time = time.time() - start_get
            self._record_time("get_block", get_time)
            
            # Decode the block
            start_decode = time.time()
            decoded_block = self.decoder.decode_block(compressed_data, self.block_size)
            decode_time = time.time() - start_decode
            self._record_time("decode_block", decode_time)
            total_decode_time += decode_time
            blocks_decoded += 1
            
            # Calculate where this block should go in the viewport
            start_place = time.time()
            block_pixel_x = block_x * self.block_size
            block_pixel_y = block_y * self.block_size
            
            # Calculate overlap between block and viewport
            overlap_left = max(0, block_pixel_x - x)
            overlap_top = max(0, block_pixel_y - y)
            overlap_right = min(width, block_pixel_x + self.block_size - x)
            overlap_bottom = min(height, block_pixel_y + self.block_size - y)
            
            # Skip if no overlap
            if overlap_left >= overlap_right or overlap_top >= overlap_bottom:
                continue
                
            # Copy the relevant portion of the decoded block to the result
            block_start_x = max(0, x - block_pixel_x)
            block_start_y = max(0, y - block_pixel_y)
            block_end_x = min(self.block_size, x + width - block_pixel_x)
            block_end_y = min(self.block_size, y + height - block_pixel_y)
            
            # Convert grayscale to RGB if needed
            layer_index = layers.index(layer_name) % 3  # Map to R, G, B channel
            if len(decoded_block.shape) == 2:
                result[overlap_top:overlap_bottom, overlap_left:overlap_right, layer_index] = \
                    decoded_block[block_start_y:block_end_y, block_start_x:block_end_x]
            else:
                # RGB block
                result[overlap_top:overlap_bottom, overlap_left:overlap_right] = \
                    decoded_block[block_start_y:block_end_y, block_start_x:block_end_x]
            
            place_time = time.time() - start_place
            self._record_time("place_block", place_time)
        
        if blocks_decoded > 0:
            print(f"Decoded {blocks_decoded} blocks in {total_decode_time:.4f} seconds")
            
        total_time = time.time() - start_time
        self._record_time("total_viewport", total_time)
        
        return result, {"blocks_decoded": blocks_decoded, "decode_time": total_time}
    
    def print_timing_summary(self):
        """Print summary of timing metrics"""
        print("\nTiming Summary:")
        print("--------------")
        
        for operation, times in self.instrumentation.items():
            avg_time = sum(times) / len(times) if times else 0
            max_time = max(times) if times else 0
            total_time = sum(times)
            
            print(f"  {operation}:")
            print(f"    Avg: {avg_time*1000:.2f}ms")
            print(f"    Max: {max_time*1000:.2f}ms")
            print(f"    Total: {total_time*1000:.2f}ms")
    
    def plot_timing_breakdown(self, title="Timing Breakdown by Operation", save_path=None):
        """Create a plot showing timing breakdown by operation"""
        operations = list(self.instrumentation.keys())
        avg_times = [sum(times) / len(times) if times else 0 for times in self.instrumentation.values()]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(operations, [t*1000 for t in avg_times])  # Convert to ms
        
        plt.title(title)
        plt.xlabel('Operation')
        plt.ylabel('Average Time (ms)')
        plt.xticks(rotation=45, ha='right')
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            return plt.gcf()