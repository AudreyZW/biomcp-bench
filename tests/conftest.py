import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The benchmark lives outside the installed package, so tests reach it by path.
sys.path.insert(0, str(HERE.parent / "evals"))
# And this directory, so shared fakes import as plain modules.
sys.path.insert(0, str(HERE))
