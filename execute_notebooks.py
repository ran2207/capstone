#!/usr/bin/env python3
"""
Script to integrate solution code into notebooks and execute them
"""
import json
import subprocess
import sys
from pathlib import Path

# Notebook to solution mapping
NOTEBOOK_MAPPING = {
    "AI-capstone-M1L1-v1.ipynb": "solutions/M1L1_solutions.py",
    "AI-capstone-M1L2-v1.ipynb": "solutions/M1L2_solutions.py",
    "AI-capstone-M1L3-v1.ipynb": "solutions/M1L3_solutions.py",
    "Lab-M2L1-Train-and-Evaluate-a-Keras-Based-Classifier-v1.ipynb": "solutions/M2L1_solutions.py",
    "Lab-M2L2-Implement-and-Test-a-PyTorch-Based-Classifier-v1.ipynb": "solutions/M2L2_solutions.py",
    "Lab-M2L3-Comparative-Analysis-of-Keras-and-PyTorch-Models-v1.ipynb": "solutions/M2L3_solutions.py",
    "Lab-M3L1-Vision-Transformers-in-Keras-v1.ipynb": "solutions/M3L1_solutions.py",
    "Lab-M3L2-Vision-Transformers-in-PyTorch-v1.ipynb": "solutions/M3L2_solutions.py",
    "lab-M4L1-Land-Classification-CNN-ViT-Integration-Evaluation-v1.ipynb": "solutions/M3L3_solutions.py",
}

def read_solution_file(solution_path):
    """Read solution file and extract code sections"""
    with open(solution_path, 'r') as f:
        content = f.read()
    return content

def execute_notebook(notebook_path):
    """Execute notebook using jupyter nbconvert"""
    try:
        print(f"Executing {notebook_path}...")
        result = subprocess.run(
            [
                "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                "--ExecutePreprocessor.timeout=600",
                "--ExecutePreprocessor.kernel_name=python3",
                notebook_path
            ],
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode == 0:
            print(f"✓ Successfully executed {notebook_path}")
            return True
        else:
            print(f"✗ Failed to execute {notebook_path}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Exception while executing {notebook_path}: {e}")
        return False

def integrate_and_execute(notebook_name, solution_path):
    """Integrate solution into notebook and execute"""
    print(f"\n{'='*60}")
    print(f"Processing: {notebook_name}")
    print(f"Solution: {solution_path}")
    print(f"{'='*60}")

    # Read the notebook
    with open(notebook_name, 'r') as f:
        notebook = json.load(f)

    # Read the solution
    solution_code = read_solution_file(solution_path)

    # Create a new code cell with all the solution code
    # This is a simple approach - add solution as a new cell at the end
    solution_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": solution_code.split('\n')
    }

    # Add solution cell to notebook
    notebook['cells'].append(solution_cell)

    # Save modified notebook
    with open(notebook_name, 'w') as f:
        json.dump(notebook, f, indent=2)

    # Execute the notebook
    return execute_notebook(notebook_name)

def main():
    success_count = 0
    fail_count = 0

    for notebook, solution in NOTEBOOK_MAPPING.items():
        if Path(notebook).exists() and Path(solution).exists():
            if integrate_and_execute(notebook, solution):
                success_count += 1
            else:
                fail_count += 1
        else:
            print(f"✗ Missing files: {notebook} or {solution}")
            fail_count += 1

    print(f"\n{'='*60}")
    print(f"Summary: {success_count} succeeded, {fail_count} failed")
    print(f"{'='*60}")

    return 0 if fail_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
