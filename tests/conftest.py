import sys
from pathlib import Path

# The benchmark lives outside the installed package, so tests reach it by path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))
