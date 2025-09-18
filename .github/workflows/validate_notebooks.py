#!/usr/bin/env python3
"""
Notebook validation script for CI/CD pipeline using nbstripout.

This script validates that all Jupyter notebooks in the repository are "clean"
(i.e., have no outputs or execution counts) using the standard nbstripout tool.

Exit codes:
    0: All notebooks are clean
    1: One or more notebooks contain outputs/execution counts
    2: Script error (nbstripout not available, etc.)
"""

import subprocess
import sys
from pathlib import Path
from typing import List


def find_notebooks(root_dir: str = ".") -> List[Path]:
    """Find all notebook files, excluding backups and build artifacts."""
    root = Path(root_dir)
    notebooks = list(root.glob("**/*.ipynb"))

    # Filter out backup files, build artifacts, and hidden directories
    filtered = []
    for nb in notebooks:
        # Skip backup files
        if nb.name.endswith('.backup.ipynb') or nb.name.endswith('_backup.ipynb'):
            continue

        # Skip files in _book, .git, or other hidden directories
        if any(part.startswith('.') or part == '_book' for part in nb.parts):
            continue

        filtered.append(nb)

    return sorted(filtered)


def check_nbstripout_available() -> bool:
    """Check if nbstripout is available."""
    try:
        subprocess.run(['nbstripout', '--version'],
                      capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def validate_notebooks(notebooks: List[Path]) -> bool:
    """Validate notebooks using nbstripout --verify."""
    if not notebooks:
        print("ℹ️  No notebooks found to validate")
        return True

    print(f"🔍 Validating {len(notebooks)} notebook(s) with nbstripout...")

    try:
        # Use nbstripout --verify which returns non-zero if files would be changed
        result = subprocess.run(
            ['nbstripout', '--verify'] + [str(nb) for nb in notebooks],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print("✅ All notebooks are clean!")
            return True
        else:
            print("❌ Some notebooks contain outputs or execution counts:")
            if result.stdout:
                # nbstripout prints which files would be stripped
                for line in result.stdout.strip().split('\n'):
                    if line.startswith('Dry run: would have stripped'):
                        filename = line.replace('Dry run: would have stripped ', '')
                        print(f"  • {filename}")
            return False

    except subprocess.CalledProcessError as e:
        print(f"❌ Error running nbstripout: {e}")
        return False


def print_help_message():
    """Print instructions for cleaning notebooks."""
    print("\n" + "="*60)
    print("❌ NOTEBOOK VALIDATION FAILED")
    print("="*60)

    print(f"\n📚 WHY THIS MATTERS:")
    print("  • Large notebooks with outputs can cause build failures")
    print("  • Embedded images/plots create unnecessarily large files")
    print("  • Clean notebooks ensure faster CI/CD and better git diffs")

    print(f"\n🔧 HOW TO FIX:")
    print("  1. Install nbstripout: pip install nbstripout")
    print("  2. Clean all notebooks: nbstripout *.ipynb")
    print("  3. Or use the provided script: ./clean_notebooks.sh")
    print("  4. Commit the cleaned notebooks")

    print(f"\n💡 ALTERNATIVE METHODS:")
    print("  # Using Jupyter interface:")
    print("  1. Open notebook → Kernel → Restart & Clear Output → Save")
    print("")
    print("  # Using jupyter command:")
    print("  jupyter nbconvert --clear-output --inplace notebook.ipynb")
    print("")
    print("  # Clean all notebooks at once:")
    print("  find . -name '*.ipynb' -not -path './_book/*' \\")
    print("    -exec nbstripout {} +")

    print(f"\n🤖 AUTOMATION:")
    print("  Set up automatic cleaning with git hooks:")
    print("  nbstripout --install")
    print("  (This automatically cleans notebooks when you commit)")

    print("\n" + "="*60)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate that Jupyter notebooks are clean using nbstripout"
    )
    parser.add_argument(
        "--root-dir",
        default=".",
        help="Root directory to search for notebooks (default: current directory)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only show errors and final result"
    )

    args = parser.parse_args()

    # Check if nbstripout is available
    if not check_nbstripout_available():
        print("❌ Error: nbstripout is not available")
        print("Please install it with: pip install nbstripout")
        sys.exit(2)

    try:
        notebooks = find_notebooks(args.root_dir)
        is_valid = validate_notebooks(notebooks)

        if not is_valid:
            if not args.quiet:
                print_help_message()
            sys.exit(1)
        else:
            if not args.quiet:
                print("🎉 All notebooks are clean!")
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n⏹️  Validation cancelled by user")
        sys.exit(2)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
