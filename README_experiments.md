
# Prototype implementation and experiment scripts

## Files
- `pqvc_verkle_impl.py` — executable prototype of the lattice-based vector commitment and a simple Verkle-tree benchmark model
- `run_experiments.py` — experiment runner that outputs a JSON report

## Notes
This prototype is designed for **functional evaluation and benchmarking**.
It uses a single global public matrix `A` and precomputed short vectors `R_j` such that:
- `U_j = A R_j mod q`
- commitments, openings, verification, updates, and proof updates remain algebraically correct

This choice keeps the code executable and suitable for timing experiments.

## Example run
```bash
python run_experiments.py --q 12289 --n 32 --m 48 --k 16 --depth 3 --repeat 5 --output experiment_report.json
```

## Suggested paper-use workflow
1. Run the prototype on toy or moderate parameters to measure timings.
2. Use the JSON report to build tables for:
   - commit time
   - open time
   - verify time
   - update time
   - proof update time
   - proof size
   - public parameter size
   - memory usage
3. Use the built-in `analytic_paper_profile` field for the fixed-arity `k=256` parameter profile.

## Important
For the paper, fill in the hardware and software platform details in `--platform-note`.
