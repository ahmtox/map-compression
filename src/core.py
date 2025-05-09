"""Core functionality for map compression including encoders, decoders, and container format."""
import numpy as np
import struct
import json
from collections import defaultdict
import os

class ContextEncoder:
    """Context-based encoder for binary images"""
    
    def __init__(self, context_pixels=4):
        self.context_pixels = context_pixels
        self.context_counts = defaultdict(lambda: [1, 1])  # Start with 1,1 to avoid division by zero
    
    def get_context(self, img, y, x):
        """Generate context from neighboring pixels"""
        context = 0
        bit = 1
        # Use pixels from left and above (already processed)
        for dy, dx in [(-1, 0), (0, -1), (-1, -1), (-1, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < img.shape[0] and 0 <= nx < img.shape[1]:
                if img[ny, nx] > 0:
                    context |= bit
            bit <<= 1
        return context
    
    def encode_block(self, block):
        """Encode a binary block using context-based coding"""
        # Reset context counts for this block
        self.context_counts = defaultdict(lambda: [1, 1])
        
        # First pass: gather statistics
        for y in range(block.shape[0]):
            for x in range(block.shape[1]):
                context = self.get_context(block, y, x)
                pixel_val = 1 if block[y, x] > 0 else 0
                self.context_counts[context][pixel_val] += 1
        
        # Prepare the output buffer
        # First, write the context model 
        buffer = bytearray()
        for context in range(16):  # 16 possible contexts
            zeros, ones = self.context_counts.get(context, [1, 1])
            buffer.extend(struct.pack('>II', zeros, ones))
        
        # Now encode the actual pixel data
        bit_buffer = 0
        bits_in_buffer = 0
        
        for y in range(block.shape[0]):
            for x in range(block.shape[1]):
                # Add bit to buffer
                pixel_val = 1 if block[y, x] > 0 else 0
                bit_buffer = (bit_buffer << 1) | pixel_val
                bits_in_buffer += 1
                
                # If buffer is full, output a byte
                if bits_in_buffer == 8:
                    buffer.append(bit_buffer)
                    bit_buffer = 0
                    bits_in_buffer = 0
        
        # Handle any remaining bits
        if bits_in_buffer > 0:
            bit_buffer = bit_buffer << (8 - bits_in_buffer)  # Pad with zeros
            buffer.append(bit_buffer)
        
        # Ensure we return at least one data byte to avoid empty blocks
        if len(buffer) <= 128:  # Only context model was written
            buffer.append(0)    # Add a byte of data
        
        return buffer

class ArithmeticEncoder:
    """MQ-style arithmetic encoder for binary images"""
    
    def __init__(self):
        self.LOW = 0
        self.HIGH = 0x7FFF
        self.follow = 0
        self.code = 0
        self.range_value = self.HIGH
        
    def _get_probability(self, context_counts):
        """Estimate probability from context counts"""
        zeros, ones = context_counts
        total = zeros + ones
        # Ensure probabilities are never 0 or 1 (would cause arithmetic coding issues)
        p_zero = max(0.001, min(0.999, zeros / total))
        return p_zero
        
    def encode_symbol(self, symbol, context_counts, output_bits):
        """Encode a single binary symbol (0 or 1) using arithmetic coding"""
        p_zero = self._get_probability(context_counts)
        range_width = self.range_value
        
        if symbol == 0:
            # Symbol is 0, adjust high point
            self.range_value = int(range_width * p_zero)
        else:
            # Symbol is 1, adjust low point
            low_increment = int(range_width * p_zero)
            self.LOW += low_increment
            self.range_value = range_width - low_increment
            
        # Renormalization step
        while self.range_value < 0x8000:
            if self.LOW < 0x4000:
                # Output 0 + any pending bits
                output_bits.append(0)
                for _ in range(self.follow):
                    output_bits.append(1)
                self.follow = 0
            elif self.LOW >= 0x4000:
                # Output 1 + any pending bits (as 0s)
                output_bits.append(1)
                for _ in range(self.follow):
                    output_bits.append(0)
                self.follow = 0
                self.LOW -= 0x4000
            else:
                # Middle range, track follow bits
                self.follow += 1
                self.LOW -= 0x2000
            
            # Scale up
            self.LOW <<= 1
            self.range_value <<= 1
            
    def flush(self, output_bits):
        """Flush remaining bits"""
        # Output one more bit to resolve ambiguity
        if self.LOW < 0x2000:
            output_bits.append(0)
            for _ in range(self.follow):
                output_bits.append(1)
        else:
            output_bits.append(1)
            for _ in range(self.follow):
                output_bits.append(0)

class ArithmeticContextEncoder(ContextEncoder):
    """Context-based encoder using arithmetic coding"""
    
    def encode_block(self, block):
        """Encode a binary block using context-based arithmetic coding"""
        # Reset context counts for this block
        self.context_counts = defaultdict(lambda: [1, 1])
        
        # First pass: gather statistics
        for y in range(block.shape[0]):
            for x in range(block.shape[1]):
                context = self.get_context(block, y, x)
                pixel_val = 1 if block[y, x] > 0 else 0
                self.context_counts[context][pixel_val] += 1
        
        # Prepare the output buffer
        # First, write the context model 
        buffer = bytearray()
        for context in range(16):  # 16 possible contexts
            zeros, ones = self.context_counts.get(context, [1, 1])
            buffer.extend(struct.pack('>II', zeros, ones))
        
        # Now encode the actual pixel data using arithmetic coding
        output_bits = []
        encoder = ArithmeticEncoder()
        
        for y in range(block.shape[0]):
            for x in range(block.shape[1]):
                context = self.get_context(block, y, x)
                pixel_val = 1 if block[y, x] > 0 else 0
                encoder.encode_symbol(pixel_val, self.context_counts[context], output_bits)
        
        # Flush any remaining bits
        encoder.flush(output_bits)
        
        # Pack bits into bytes
        bits_buffer = bytearray()
        for i in range(0, len(output_bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(output_bits) and output_bits[i + j]:
                    byte |= (1 << (7 - j))
            bits_buffer.append(byte)
        
        # Append encoded data length and the actual encoded data
        buffer.extend(struct.pack('>I', len(bits_buffer)))
        buffer.extend(bits_buffer)
        
        return buffer

class ContextDecoder:
    """Context-based decoder for binary images"""
    
    def __init__(self):
        self.context_counts = defaultdict(lambda: [1, 1])
    
    def get_context(self, img, y, x):
        """Generate context from neighboring pixels"""
        context = 0
        bit = 1
        # Same context definition as encoder
        for dy, dx in [(-1, 0), (0, -1), (-1, -1), (-1, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < img.shape[0] and 0 <= nx < img.shape[1]:
                if img[ny, nx] > 0:
                    context |= bit
            bit <<= 1
        return context
    
    def decode_block(self, encoded_data, block_size=128):
        """Decode a block of data"""
        # Handle empty blocks
        if not encoded_data or len(encoded_data) == 0:
            return np.zeros((block_size, block_size), dtype=np.uint8)
            
        # First, read the context model
        offset = 0
        for context in range(16):
            # Make sure we have enough data for unpacking
            if offset + 8 > len(encoded_data):
                return np.zeros((block_size, block_size), dtype=np.uint8)
                
            zeros, ones = struct.unpack_from('>II', encoded_data, offset)
            self.context_counts[context] = [zeros, ones]
            offset += 8  # Each 'I' is 4 bytes, so 2*4=8
        
        # Now decode the block data
        result = np.zeros((block_size, block_size), dtype=np.uint8)
        
        # Check if there's any data beyond the context model
        if len(encoded_data) <= offset:
            return result  # Return all zeros if no additional data
            
        bit_buffer = 0
        bits_remaining = 0
        
        for y in range(block_size):
            for x in range(block_size):
                # Get more bits if needed
                if bits_remaining == 0:
                    # Check if we've reached the end of the data
                    if offset >= len(encoded_data):
                        break
                    bit_buffer = encoded_data[offset]
                    offset += 1
                    bits_remaining = 8
                
                # Extract next bit
                pixel_val = (bit_buffer >> 7) & 1
                bit_buffer <<= 1
                bits_remaining -= 1
                
                # Set pixel
                result[y, x] = 255 if pixel_val else 0
        
        return result

class ArithmeticDecoder:
    """MQ-style arithmetic decoder for binary images"""
    
    def __init__(self, input_bits):
        self.LOW = 0
        self.HIGH = 0x7FFF
        self.range_value = self.HIGH
        self.input_bits = input_bits
        self.bit_pos = 0
        
        # Initialize code value
        self.code = 0
        for _ in range(16):
            if self.bit_pos < len(self.input_bits):
                self.code = (self.code << 1) | self.input_bits[self.bit_pos]
                self.bit_pos += 1
            else:
                self.code <<= 1
    
    def _get_probability(self, context_counts):
        """Estimate probability from context counts"""
        zeros, ones = context_counts
        total = zeros + ones
        # Ensure probabilities are never 0 or 1
        p_zero = max(0.001, min(0.999, zeros / total))
        return p_zero
        
    def decode_symbol(self, context_counts):
        """Decode a single binary symbol using arithmetic coding"""
        p_zero = self._get_probability(context_counts)
        range_width = self.range_value
        
        # Determine the split point
        split_point = self.LOW + int(range_width * p_zero)
        
        # Determine the symbol
        if self.code <= split_point:
            symbol = 0
            self.range_value = int(range_width * p_zero)
        else:
            symbol = 1
            low_increment = int(range_width * p_zero)
            self.LOW += low_increment
            self.range_value = range_width - low_increment
            
        # Renormalization step
        while self.range_value < 0x8000:
            self.range_value <<= 1
            self.LOW <<= 1
            
            self.code <<= 1
            if self.bit_pos < len(self.input_bits):
                self.code |= self.input_bits[self.bit_pos]
                self.bit_pos += 1
                
            if self.LOW >= 0x10000:
                self.LOW -= 0x10000
                self.code -= 0x10000
                
        return symbol

class ArithmeticContextDecoder(ContextDecoder):
    """Context-based decoder using arithmetic coding"""
    
    def decode_block(self, encoded_data, block_size=128):
        """Decode a block of data using arithmetic coding"""
        # Handle empty blocks
        if not encoded_data or len(encoded_data) == 0:
            return np.zeros((block_size, block_size), dtype=np.uint8)
            
        # First, read the context model
        offset = 0
        for context in range(16):
            # Make sure we have enough data for unpacking
            if offset + 8 > len(encoded_data):
                return np.zeros((block_size, block_size), dtype=np.uint8)
                
            zeros, ones = struct.unpack_from('>II', encoded_data, offset)
            self.context_counts[context] = [zeros, ones]
            offset += 8
        
        # Read the length of encoded data
        if offset + 4 > len(encoded_data):
            return np.zeros((block_size, block_size), dtype=np.uint8)
        encoded_len = struct.unpack_from('>I', encoded_data, offset)[0]
        offset += 4
        
        # Make sure we have enough data
        if offset + encoded_len > len(encoded_data):
            return np.zeros((block_size, block_size), dtype=np.uint8)
        
        # Extract the encoded bits
        encoded_bytes = encoded_data[offset:offset+encoded_len]
        input_bits = []
        for byte in encoded_bytes:
            for i in range(8):
                input_bits.append((byte >> (7 - i)) & 1)
        
        # Decode using arithmetic coding
        decoder = ArithmeticDecoder(input_bits)
        result = np.zeros((block_size, block_size), dtype=np.uint8)
        
        for y in range(block_size):
            for x in range(block_size):
                context = self.get_context(result, y, x)
                pixel_val = decoder.decode_symbol(self.context_counts[context])
                result[y, x] = 255 if pixel_val else 0
        
        return result

class BlockCompressedImage:
    """Container format for block-based compressed maps"""
    
    def __init__(self):
        self.metadata = {
            "version": 1,
            "layers": {},
            "block_size": 128
        }
        self.blocks = {}  # {layer_name: {block_coords: compressed_data}}
        
    def add_layer(self, layer_name, width, height):
        """Register a new layer"""
        self.metadata["layers"][layer_name] = {
            "width": width,
            "height": height,
            "block_count_x": (width + self.metadata["block_size"] - 1) // self.metadata["block_size"],
            "block_count_y": (height + self.metadata["block_size"] - 1) // self.metadata["block_size"]
        }
        self.blocks[layer_name] = {}
    
    def add_block(self, layer_name, block_y, block_x, compressed_data):
        """Add a compressed block to the image"""
        if layer_name not in self.blocks:
            raise ValueError(f"Layer {layer_name} not registered")
        
        self.blocks[layer_name][(block_y, block_x)] = compressed_data
    
    def save(self, filename):
        """Save the compressed image to a file"""
        with open(filename, 'wb') as f:
            # Write metadata
            metadata_bytes = json.dumps(self.metadata).encode('utf-8')
            f.write(struct.pack('>I', len(metadata_bytes)))
            f.write(metadata_bytes)
            
            # Calculate offsets first to ensure offset table is correct
            data_offsets = {}
            current_offset = 0  # Start at 0, will add base offset later
            
            for layer_name in self.blocks:
                data_offsets[layer_name] = {}
                for block_coords, data in self.blocks[layer_name].items():
                    # Record offset and increment
                    data_offsets[layer_name][str(block_coords)] = current_offset
                    # 4 bytes for length + actual data
                    current_offset += 4 + len(data)
            
            # Write complete offset table
            offset_bytes = json.dumps(data_offsets).encode('utf-8')
            f.write(struct.pack('>I', len(offset_bytes)))
            f.write(offset_bytes)
            
            # Remember where data starts
            data_start = f.tell()
            
            # Write actual block data using the pre-calculated offsets
            for layer_name in self.blocks:
                for block_coords, data in self.blocks[layer_name].items():
                    f.write(struct.pack('>I', len(data)))
                    f.write(data)
    
    @classmethod
    def load(cls, filename):
        """Load a compressed image from a file"""
        result = cls()
        with open(filename, 'rb') as f:
            # Read metadata
            metadata_size = struct.unpack('>I', f.read(4))[0]
            metadata_bytes = f.read(metadata_size)
            result.metadata = json.loads(metadata_bytes.decode('utf-8'))
            
            # Read offset table
            offset_size = struct.unpack('>I', f.read(4))[0]
            offset_bytes = f.read(offset_size)
            offset_table = json.loads(offset_bytes.decode('utf-8'))
            
            # Data starts here
            data_start = f.tell()
            
            # Setup layers
            for layer_name, layer_info in result.metadata["layers"].items():
                result.blocks[layer_name] = {}
                
            # Load blocks on demand (we don't load them all now)
            result._file = filename
            result._data_start = data_start
            result._offset_table = offset_table
            
        return result
    
    def has_block(self, layer_name, block_y, block_x):
        """Check if a block exists"""
        if layer_name not in self._offset_table:
            return False
        
        coord_key = f"({block_y}, {block_x})"
        return coord_key in self._offset_table[layer_name]
    
    def get_block(self, layer_name, block_y, block_x):
        """Load and decompress a specific block"""
        if layer_name not in self._offset_table:
            raise ValueError(f"Layer {layer_name} not found")
        
        coord_key = f"({block_y}, {block_x})"
        if coord_key not in self._offset_table[layer_name]:
            raise ValueError(f"Block {coord_key} not found in layer {layer_name}")
        
        offset = self._offset_table[layer_name][coord_key]
        
        try:
            with open(self._file, 'rb') as f:
                # Get file size to avoid reading past EOF
                f.seek(0, 2)  # Seek to end
                file_size = f.tell()
                
                # Check if our read position is valid
                read_pos = self._data_start + offset
                if read_pos + 4 > file_size:
                    print(f"WARNING: Can't read block data at offset {offset} - file too small")
                    return b''  # Return empty data
                    
                # Position file pointer and read the data
                f.seek(read_pos)
                data_size = struct.unpack('>I', f.read(4))[0]
                
                # Verify we have enough data to read
                if read_pos + 4 + data_size > file_size:
                    print(f"WARNING: Block data truncated at offset {offset}")
                    return b''  # Return empty data
                    
                compressed_data = f.read(data_size)
                return compressed_data
                
        except Exception as e:
            print(f"Error reading block: {e}")
            return b''  # Return empty data on any error

    @staticmethod
    def debug_container(filename):
        """Debug utility to examine a container file"""
        container = BlockCompressedImage.load(filename)
        print(f"File: {filename}")
        print(f"Metadata: {container.metadata}")
        print(f"Layers: {list(container._offset_table.keys())}")
        
        # Check first few blocks
        for layer_name in container._offset_table:
            print(f"\nLayer: {layer_name}")
            blocks = list(container._offset_table[layer_name].items())[:5]
            
            for block_key, offset in blocks:
                print(f"  Block {block_key}: offset={offset}")
                
                # Try to read this block
                try:
                    block_data = container.get_block(layer_name, 
                                int(block_key.strip("()").split(",")[0]), 
                                int(block_key.strip("()").split(",")[1]))
                    print(f"    Read {len(block_data)} bytes of data")
                except Exception as e:
                    print(f"    Error: {e}")
        
        return container