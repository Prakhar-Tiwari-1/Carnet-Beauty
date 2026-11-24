"""One command rebuilds everything: python -m sillage.build [--stage NAME]."""
from __future__ import annotations

import argparse
import json
import random

import numpy as np

from sillage.config import load_config

STAGES = ["step0"]


def _seed() -> None:
    seed = load_config()["seed"]
    random.seed(seed)
    np.random.seed(seed)


def stage_step0() -> None:
    from sillage import step0

    print(json.dumps(step0.run(), indent=2))
    print("Wrote docs/step0.md")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sillage.build")
    parser.add_argument("--stage", choices=STAGES + ["all"], default="all")
    args = parser.parse_args(argv)
    _seed()
    for name in STAGES if args.stage == "all" else [args.stage]:
        print(f"== stage: {name}")
        globals()[f"stage_{name}"]()


if __name__ == "__main__":
    main()
