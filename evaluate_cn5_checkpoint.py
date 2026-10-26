#!/usr/bin/env python3
"""CN5 entry point for the repository's unified checkpoint evaluator."""

import evaluate_cn6_dense_grid_checkpoint as evaluator

evaluator.N_AGENTS = 5
evaluator.N_LANDMARKS = 5
evaluator.MAX_STEPS = 25
evaluator.OCCUPANCY_RADIUS = 0.3

if __name__ == "__main__":
    evaluator.main()
