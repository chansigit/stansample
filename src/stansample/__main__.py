"""CLI: rank the sample column(s) of an .h5ad file.

stdout always carries a single JSON object: {"method": ..., "candidates": [...]}.
On an IO error a diagnostic line is written to stderr and the exit code is 1.
"""

import argparse
import json
import sys
from dataclasses import asdict

from .rank import rank_sample_columns


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="stansample",
        description="Rank which .obs column identifies the sample each cell came "
                    "from. Emits a JSON object on stdout.")
    parser.add_argument("path", help="path to an .h5ad file")
    parser.add_argument("--no-llm", action="store_true",
                        help="force the offline heuristic ranker (no API call)")
    parser.add_argument("--top", type=int, default=5,
                        help="keep top K candidates (default 5; 0 = all)")
    parser.add_argument("--model", default="claude-opus-4-8",
                        help="LLM model id (default claude-opus-4-8)")
    args = parser.parse_args(argv)

    try:
        import anndata
        adata = anndata.read_h5ad(args.path, backed="r")
    except Exception as exc:
        print(f"error: cannot read {args.path!r}: {exc}", file=sys.stderr)
        return 1

    result = rank_sample_columns(
        adata, use_llm=not args.no_llm, model=args.model, top_k=args.top)

    print(json.dumps(
        {"method": result.method,
         "candidates": [asdict(c) for c in result.candidates]}, indent=2))

    return 0 if result.candidates else 2


if __name__ == "__main__":
    raise SystemExit(main())
