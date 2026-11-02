"""One-off: recompress a plain-JSON cache directory in place.

The fixture set is committed, and uncompressed it was 29 MB for 640 entries --
mostly the verbose upstream records this project exists to complain about.
Gzipping takes it under 4 MB. `ResponseCache` still reads plain `.json`, so
this is a size optimisation rather than a migration anything depends on.

    uv run python scripts/compress_cache.py evals/cache
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    directory = Path(argv[1] if len(argv) > 1 else "evals/cache")
    plain = sorted(directory.glob("*.json"))
    if not plain:
        print(f"nothing to compress in {directory}")
        return 0

    before = sum(f.stat().st_size for f in plain)
    converted = skipped = 0
    for path in plain:
        target = path.with_suffix(".json.gz")
        try:
            record = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            skipped += 1
            continue
        with gzip.open(target, "wt", encoding="utf-8") as handle:
            json.dump(record, handle)
        path.unlink()
        converted += 1

    after = sum(f.stat().st_size for f in directory.glob("*.json.gz"))
    print(f"compressed {converted} entries ({skipped} skipped)")
    print(f"{before / 1e6:.1f} MB -> {after / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
