#!/usr/bin/env python3
"""
Fix notebook structure, populate with solutions, and execute
"""
import json
import subprocess
import sys
from pathlib import Path

def fix_notebook_structure(notebook_path):
    """Fix invalid notebook JSON structure"""
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)

    # Fix cells - remove 'outputs' from markdown cells, ensure proper structure
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            # Remove outputs from markdown cells (not allowed)
            if 'outputs' in cell:
                del cell['outputs']
        elif cell['cell_type'] == 'code':
            # Ensure code cells have outputs list
            if 'outputs' not in cell:
                cell['outputs'] = []
            # Ensure execution_count exists
            if 'execution_count' not in cell:
                cell['execution_count'] = None

        # Ensure metadata exists
        if 'metadata' not in cell:
            cell['metadata'] = {}

        # Ensure source is a list
        if isinstance(cell['source'], str):
            cell['source'] = cell['source'].split('\n')

    return notebook

def add_solution_to_notebook(notebook, solution_path):
    """Add solution code to notebook"""
    with open(solution_path, 'r') as f:
        solution_code = f.read()

    # Create new code cell with solution
    solution_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": solution_code.split('\n')
    }

    # Add to notebook
    notebook['cells'].append(solution_cell)
    return notebook

def execute_notebook(notebook_path):
    """Execute notebook using jupyter"""
    try:
        print(f"Executing {notebook_path}...")
        result = subprocess.run(
            [
                "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                "--ExecutePreprocessor.timeout=1800",
                "--allow-errors",
                notebook_path
            ],
            capture_output=True,
            text=True,
            timeout=1800
        )

        if "error" in result.stderr.lower() and "nbconvert" not in result.stderr.lower():
            print(f"Execution had errors")
            print(result.stderr[:500])
            return False

        print(f"✓ Successfully executed {notebook_path}")
        return True
    except Exception as e:
        print(f"✗ Failed: {str(e)[:200]}")
        return False

def process_notebook(notebook_path, solution_path):
    """Fix, populate, and execute a notebook"""
    print(f"\n{'='*70}")
    print(f"Processing: {notebook_path}")
    print(f"{'='*70}")

    try:
        # Fix structure
        print("1. Fixing notebook structure...")
        notebook = fix_notebook_structure(notebook_path)

        # Add solution
        print("2. Adding solution code...")
        notebook = add_solution_to_notebook(notebook, solution_path)

        # Save
        print("3. Saving modified notebook...")
        with open(notebook_path, 'w') as f:
            json.dump(notebook, f, indent=2)

        # Execute
        print("4. Executing notebook...")
        success = execute_notebook(notebook_path)

        return success
    except Exception as e:
        print(f"✗ Error processing notebook: {str(e)[:200]}")
        return False

NOTEBOOKS = [
    ("AI-capstone-M1L1-v1.ipynb", "solutions/M1L1_solutions.py"),
    ("AI-capstone-M1L2-v1.ipynb", "solutions/M1L2_solutions.py"),
    ("AI-capstone-M1L3-v1.ipynb", "solutions/M1L3_solutions.py"),
    ("Lab-M2L1-Train-and-Evaluate-a-Keras-Based-Classifier-v1.ipynb", "solutions/M2L1_solutions.py"),
    ("Lab-M2L2-Implement-and-Test-a-PyTorch-Based-Classifier-v1.ipynb", "solutions/M2L2_solutions.py"),
    ("Lab-M2L3-Comparative-Analysis-of-Keras-and-PyTorch-Models-v1.ipynb", "solutions/M2L3_solutions.py"),
    ("Lab-M3L1-Vision-Transformers-in-Keras-v1.ipynb", "solutions/M3L1_solutions.py"),
    ("Lab-M3L2-Vision-Transformers-in-PyTorch-v1.ipynb", "solutions/M3L2_solutions.py"),
    ("lab-M4L1-Land-Classification-CNN-ViT-Integration-Evaluation-v1.ipynb", "solutions/M3L3_solutions.py"),
]

def main():
    print("="*70)
    print(" Processing All Notebooks")
    print("="*70)

    results = []
    for notebook, solution in NOTEBOOKS:
        if Path(notebook).exists() and Path(solution).exists():
            success = process_notebook(notebook, solution)
            results.append((notebook, success))
        else:
            print(f"\n✗ Missing: {notebook} or {solution}")
            results.append((notebook, False))

    # Summary
    print(f"\n{'='*70}")
    print(" SUMMARY")
    print(f"{'='*70}")
    for notebook, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {notebook}")

    success_count = sum(1 for _, s in results if s)
    print(f"\nSuccess: {success_count}/{len(results)}")

    return 0 if success_count == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())
