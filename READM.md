# Post-Quantum Verkle Tree Prototype

This repository contains the prototype implementation and experiment scripts for the paper on a lattice-based vector commitment scheme for Verkle trees.

## Files
- `pqvc_verkle_impl.py` — implementation of the commitment scheme
- `run_experiments.py` — experiment runner
- `experiment_report.json` — example output report
- `README_experiments.md` — short usage notes

## Run
```bash
python run_experiments.py --q 12289 --n 32 --m 48 --k 16 --depth 3 --repeat 5 --output experiment_report.json