#!/usr/bin/env python3
# apply_import_replacements.py
import re
from pathlib import Path

def should_skip(p: Path):
    exclude_dirs = {".git", ".venv_test", "venv", "dist", "build", "__pycache__"}
    return any(part in exclude_dirs for part in p.parts)

def apply_file(p: Path):
    text = p.read_text(encoding="utf8")
    original_text = text
    
    # Replace import mlx.core as mx
    text = re.sub(r'^(\\s*)import\\s+mlx\\.core\\s+as\\s+mx\\s*$', r'\\1import trm_ml.core as mx', text, flags=re.MULTILINE)
    
    # Replace import mlx.core
    text = re.sub(r'^(\\s*)import\\s+mlx\\.core\\s*$', r'\\1import trm_ml.core', text, flags=re.MULTILINE)
    
    # Replace from mlx.core import ...
    text = re.sub(r'^(\\s*)from\\s+mlx\\.core\\s+import\\s+(.*)$', r'\\1from trm_ml.core import \\2', text, flags=re.MULTILINE)
    
    # Replace import mlx (be more conservative with this one)
    text = re.sub(r'^(\\s*)import\\s+mlx\\s*$', r'\\1import trm_ml', text, flags=re.MULTILINE)
    
    if text != original_text:
        p.write_text(text, encoding="utf8")
        return True
    return False

def main():
    changed = []
    for p in Path(".").rglob("*.py"):
        if should_skip(p): 
            continue
        if apply_file(p):
            changed.append(str(p))
    print(f"Modified {len(changed)} files:")
    for c in changed: 
        print(" -", c)

if __name__ == "__main__":
    main()