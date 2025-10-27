#!/usr/bin/env python3
"""
Populate notebooks with solutions and execute them
"""
import json
import subprocess
import sys
import re
from pathlib import Path

def parse_solution_by_tasks(solution_path):
    """Parse solution file and extract code by task numbers"""
    with open(solution_path, 'r') as f:
        content = f.read()

    tasks = {}
    # Split by task markers
    task_pattern = r'# =+\n# TASK (\d+):.*?\n# =+\n(.*?)(?=# =+\n# TASK|\Z)'
    matches = re.findall(task_pattern, content, re.DOTALL)

    for task_num, task_code in matches:
        tasks[int(task_num)] = task_code.strip()

    # If no task markers found, return all code as one block
    if not tasks:
        # Remove comments and return all imports and code
        tasks[0] = content

    return tasks

def populate_notebook(notebook_path, solution_path):
    """Populate notebook cells with solution code"""
    print(f"Processing {notebook_path}...")

    # Read notebook
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)

    # Read solution
    solution_tasks = parse_solution_by_tasks(solution_path)

    # Read entire solution as backup
    with open(solution_path, 'r') as f:
        full_solution = f.read()

    # Find code cells that are empty or have TODO/Your code here
    current_task = 1
    cells_modified = 0

    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

            # Check if cell is empty or has placeholder
            is_empty = len(source.strip()) == 0
            has_placeholder = any(marker in source for marker in ['# Your code here', '# TODO', 'pass', '### START CODE HERE', '### END CODE HERE'])

            if is_empty or has_placeholder:
                # Insert solution for current task
                if current_task in solution_tasks:
                    cell['source'] = solution_tasks[current_task].split('\n')
                    cells_modified += 1
                    print(f"  ✓ Populated Task {current_task}")
                    current_task += 1

    # If we didn't find specific tasks, append full solution at the end
    if cells_modified == 0:
        print(f"  No task cells found, appending complete solution...")
        solution_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": full_solution.split('\n')
        }
        notebook['cells'].append(solution_cell)
        cells_modified = 1

    # Save modified notebook
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=2)

    print(f"  Modified {cells_modified} cells")
    return True

def execute_notebook(notebook_path):
    """Execute notebook with error handling"""
    try:
        print(f"  Executing {notebook_path}...")
        result = subprocess.run(
            [
                "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                "--ExecutePreprocessor.timeout=1200",
                "--allow-errors",
                notebook_path
            ],
            capture_output=True,
            text=True,
            timeout=1200
        )

        if result.returncode == 0:
            print(f"  ✓ Executed successfully")
            return True
        else:
            print(f"  ⚠ Execution completed with errors")
            if result.stderr:
                print(f"  Error details: {result.stderr[:500]}")
            return False
    except Exception as e:
        print(f"  ✗ Exception: {str(e)[:200]}")
        return False

# Mapping of notebooks to solutions
NOTEBOOKS = [
    ("AI-capstone-M1L1-v1.ipynb", "solutions/M1L1_solutions.py"),
    ("AI-capstone-M1L2-v1.ipynb", "solutions/M1L2_solutions.py"),
    ("AI-capstone-M1L3-v1.ipynb", "solutions/M1L3_solutions.py"),
]

def main():
    print("="*70)
    print(" Populating and Executing Notebooks")
    print("="*70)

    results = []

    for notebook, solution in NOTEBOOKS:
        print(f"\n{'='*70}")
        print(f"Processing: {notebook}")
        print(f"{'='*70}")

        if not Path(notebook).exists():
            print(f"  ✗ Notebook not found: {notebook}")
            results.append((notebook, False))
            continue

        if not Path(solution).exists():
            print(f"  ✗ Solution not found: {solution}")
            results.append((notebook, False))
            continue

        # Populate notebook with solutions
        populate_success = populate_notebook(notebook, solution)

        # Execute notebook
        if populate_success:
            execute_success = execute_notebook(notebook)
            results.append((notebook, execute_success))
        else:
            results.append((notebook, False))

    # Print summary
    print(f"\n{'='*70}")
    print(" SUMMARY")
    print(f"{'='*70}")

    for notebook, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {notebook}")

    success_count = sum(1 for _, s in results if s)
    print(f"\nCompleted: {success_count}/{len(results)} notebooks")

    return 0 if success_count == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())
