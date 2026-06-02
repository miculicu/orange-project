"""Backward-compatible demo entry point for the random small scenario."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scenario_random_small import SCENARIO
from cybergame.scenario import run_live_scenario


if __name__ == "__main__":
    run_live_scenario(SCENARIO)
