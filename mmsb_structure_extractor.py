#!/usr/bin/env python3
"""
MMSB Structure Extractor
Programmatically extracts struct, impl, fn declarations from Rust
and struct, function declarations from Julia
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class CodeElement:
    element_type: str  # 'struct', 'impl', 'fn', 'module'
    name: str
    file_path: str
    line_number: int
    language: str  # 'rust' or 'julia'
    layer: str  # e.g., '01_page', '07_intention'

class MMSBStructureExtractor:
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.elements: List[CodeElement] = []
        
    def extract_layer(self, path: Path) -> str:
        """Extract layer from path like '01_page', '07_intention'"""
        parts = path.parts
        for part in parts:
            if re.match(r'^\d+_\w+', part):
                return part
        return 'root'
    
    def extract_rust_structures(self, file_path: Path):
        """Extract struct, impl, fn from Rust files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            layer = self.extract_layer(file_path)
            rel_path = file_path.relative_to(self.root_path)
            
            for i, line in enumerate(lines, 1):
                line = line.strip()
                
                # Match struct definitions
                struct_match = re.match(r'^pub\s+struct\s+(\w+)', line)
                if struct_match:
                    self.elements.append(CodeElement(
                        element_type='struct',
                        name=struct_match.group(1),
                        file_path=str(rel_path),
                        line_number=i,
                        language='rust',
                        layer=layer
                    ))
                
                # Match impl blocks
                impl_match = re.match(r'^impl(?:\s*<[^>]+>)?\s+(\w+)', line)
                if impl_match:
                    self.elements.append(CodeElement(
                        element_type='impl',
                        name=impl_match.group(1),
                        file_path=str(rel_path),
                        line_number=i,
                        language='rust',
                        layer=layer
                    ))
                
                # Match function definitions
                fn_match = re.match(r'^(?:pub\s+)?fn\s+(\w+)', line)
                if fn_match:
                    self.elements.append(CodeElement(
                        element_type='fn',
                        name=fn_match.group(1),
                        file_path=str(rel_path),
                        line_number=i,
                        language='rust',
                        layer=layer
                    ))
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    def extract_julia_structures(self, file_path: Path):
        """Extract struct, function from Julia files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            layer = self.extract_layer(file_path)
            rel_path = file_path.relative_to(self.root_path)
            
            for i, line in enumerate(lines, 1):
                line = line.strip()
                
                # Match struct definitions
                struct_match = re.match(r'^(?:mutable\s+)?struct\s+(\w+)', line)
                if struct_match:
                    self.elements.append(CodeElement(
                        element_type='struct',
                        name=struct_match.group(1),
                        file_path=str(rel_path),
                        line_number=i,
                        language='julia',
                        layer=layer
                    ))
                
                # Match function definitions
                fn_match = re.match(r'^function\s+(\w+)', line)
                if fn_match:
                    self.elements.append(CodeElement(
                        element_type='fn',
                        name=fn_match.group(1),
                        file_path=str(rel_path),
                        line_number=i,
                        language='julia',
                        layer=layer
                    ))
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    def scan_directory(self):
        """Recursively scan directory for Rust and Julia files"""
        for rust_file in self.root_path.rglob('*.rs'):
            if 'target' not in rust_file.parts and 'examples' not in rust_file.parts:
                self.extract_rust_structures(rust_file)
        
        for julia_file in self.root_path.rglob('*.jl'):
            if 'test' not in julia_file.parts and 'examples' not in julia_file.parts:
                self.extract_julia_structures(julia_file)
    
    def generate_report(self, output_file: str):
        """Generate structured report of code elements"""
        with open(output_file, 'w') as f:
            f.write("# MMSB Code Structure Analysis\n\n")
            
            # Group by layer
            layers = {}
            for elem in self.elements:
                if elem.layer not in layers:
                    layers[elem.layer] = []
                layers[elem.layer].append(elem)
            
            # Sort layers
            sorted_layers = sorted(layers.keys())
            
            for layer in sorted_layers:
                f.write(f"\n## {layer}\n\n")
                
                # Group by language
                rust_elems = [e for e in layers[layer] if e.language == 'rust']
                julia_elems = [e for e in layers[layer] if e.language == 'julia']
                
                if rust_elems:
                    f.write("### Rust\n\n")
                    for elem in sorted(rust_elems, key=lambda x: (x.file_path, x.line_number)):
                        f.write(f"- {elem.element_type} `{elem.name}` @ {elem.file_path}:{elem.line_number}\n")
                
                if julia_elems:
                    f.write("\n### Julia\n\n")
                    for elem in sorted(julia_elems, key=lambda x: (x.file_path, x.line_number)):
                        f.write(f"- {elem.element_type} `{elem.name}` @ {elem.file_path}:{elem.line_number}\n")
            
            # Summary statistics
            f.write("\n## Summary Statistics\n\n")
            f.write(f"Total elements: {len(self.elements)}\n")
            f.write(f"Rust elements: {len([e for e in self.elements if e.language == 'rust'])}\n")
            f.write(f"Julia elements: {len([e for e in self.elements if e.language == 'julia'])}\n")
            
            # By type
            type_counts = {}
            for elem in self.elements:
                key = f"{elem.language}_{elem.element_type}"
                type_counts[key] = type_counts.get(key, 0) + 1
            
            f.write("\nBy type:\n")
            for key, count in sorted(type_counts.items()):
                f.write(f"- {key}: {count}\n")

if __name__ == "__main__":
    mmsb_path = "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB"
    extractor = MMSBStructureExtractor(mmsb_path)
    
    print("Scanning MMSB directory structure...")
    extractor.scan_directory()
    
    output = "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/mmsb_structure_analysis.md"
    print(f"Generating report to {output}...")
    extractor.generate_report(output)
    
    print(f"\nExtracted {len(extractor.elements)} code elements")
    print(f"Report saved to: {output}")
