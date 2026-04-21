
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from pqvc_verkle_impl import (
    BinaryMerkleTree,
    LatticeVectorCommitment,
    VerkleTreePrototype,
    benchmark_call,
    measure_peak_memory_bytes,
    paper_profile,
)


def make_messages(count: int, prefix: str) -> List[bytes]:
    return [f"{prefix}_{i:06d}".encode("utf-8") for i in range(count)]


def run_node_experiment(vc: LatticeVectorCommitment, repeat: int) -> Dict[str, Any]:
    messages = make_messages(vc.k, "node")
    peak_mem_commit, state = measure_peak_memory_bytes(vc.commit, messages)

    commit_stats = benchmark_call(vc.commit, messages, repeat=repeat)
    state = commit_stats["result"]

    open_index = vc.k // 3
    open_stats = benchmark_call(vc.open, state, open_index, repeat=repeat)
    proof = open_stats["result"]

    verify_stats = benchmark_call(vc.verify, state.commitment, open_index, state.messages[open_index], proof, repeat=repeat)

    updated_state, delta = vc.update(state, open_index, b"node_updated")
    update_stats = benchmark_call(vc.update, state, open_index, b"node_updated", repeat=repeat)

    another_index = (open_index + 1) % vc.k
    another_proof = vc.open(state, another_index)
    proof_update_stats = benchmark_call(vc.proof_update, another_proof, another_index, open_index, delta, repeat=repeat)

    size_report = vc.size_report()

    return {
        "scenario": "single_node",
        "parameters": {
            "q": vc.q,
            "n": vc.n,
            "m": vc.m,
            "k": vc.k,
            "l": vc.l,
            "N": vc.N,
        },
        "metrics": {
            "commit_time_mean_s": commit_stats["mean_seconds"],
            "open_time_mean_s": open_stats["mean_seconds"],
            "verify_time_mean_s": verify_stats["mean_seconds"],
            "update_time_mean_s": update_stats["mean_seconds"],
            "proof_update_time_mean_s": proof_update_stats["mean_seconds"],
            "proof_size_bytes": size_report["proof_bytes"],
            "public_parameter_size_bytes": size_report["public_parameter_bytes"],
            "memory_usage_peak_commit_bytes": peak_mem_commit,
        },
        "correctness": {
            "single_open_verifies": bool(vc.verify(state.commitment, open_index, state.messages[open_index], proof)),
            "updated_commitment_changes": bool(np.any(updated_state.commitment != state.commitment)),
            "updated_proof_verifies": bool(
                vc.verify(updated_state.commitment, another_index, state.messages[another_index], proof_update_stats["result"])
            ),
        },
    }


def run_tree_path_experiment(vc: LatticeVectorCommitment, depth: int, repeat: int) -> Dict[str, Any]:
    leaf_count = vc.k ** depth
    leaves = make_messages(leaf_count, "leaf")

    peak_mem_build, tree = measure_peak_memory_bytes(VerkleTreePrototype, vc, depth, leaves)
    build_stats = benchmark_call(VerkleTreePrototype, vc, depth, leaves, repeat=max(1, min(3, repeat)))
    tree = build_stats["result"]

    target_index = leaf_count // 2
    open_stats = benchmark_call(tree.open_leaf_path, target_index, repeat=repeat)
    path_proofs = open_stats["result"]

    verify_stats = benchmark_call(tree.verify_leaf_path, path_proofs, repeat=repeat)

    update_stats = benchmark_call(tree.update_leaf, target_index, b"leaf_updated", repeat=max(1, min(3, repeat)))

    return {
        "scenario": "verkle_path",
        "parameters": {
            "depth": depth,
            "arity": vc.k,
            "leaf_count": leaf_count,
        },
        "metrics": {
            "tree_build_time_mean_s": build_stats["mean_seconds"],
            "path_open_time_mean_s": open_stats["mean_seconds"],
            "path_verify_time_mean_s": verify_stats["mean_seconds"],
            "single_leaf_update_time_mean_s": update_stats["mean_seconds"],
            "memory_usage_peak_build_bytes": peak_mem_build,
            "path_length": len(path_proofs),
        },
        "correctness": {
            "path_verifies": bool(tree.verify_leaf_path(path_proofs)),
        },
    }


def run_multiple_openings_experiment(vc: LatticeVectorCommitment, openings: int, repeat: int) -> Dict[str, Any]:
    messages = make_messages(vc.k, "multi")
    state = vc.commit(messages)
    indices = list(range(min(openings, vc.k)))

    def open_many():
        return [vc.open(state, idx) for idx in indices]

    proofs_stats = benchmark_call(open_many, repeat=repeat)
    proofs = proofs_stats["result"]

    def verify_many():
        return all(vc.verify(state.commitment, idx, state.messages[idx], proof) for idx, proof in zip(indices, proofs))

    verify_stats = benchmark_call(verify_many, repeat=repeat)

    return {
        "scenario": "multiple_openings",
        "parameters": {
            "openings": len(indices),
        },
        "metrics": {
            "multiple_open_time_mean_s": proofs_stats["mean_seconds"],
            "multiple_verify_time_mean_s": verify_stats["mean_seconds"],
        },
        "correctness": {
            "all_verify": bool(verify_many()),
        },
    }


def run_merkle_baseline(leaf_count: int, repeat: int) -> Dict[str, Any]:
    leaves = make_messages(leaf_count, "merkle")
    peak_mem_build, tree = measure_peak_memory_bytes(BinaryMerkleTree, leaves)
    build_stats = benchmark_call(BinaryMerkleTree, leaves, repeat=max(1, min(3, repeat)))
    tree = build_stats["result"]

    target_index = leaf_count // 2
    open_stats = benchmark_call(tree.open, target_index, repeat=repeat)
    proof = open_stats["result"]
    original_root = tree.root

    verify_stats = benchmark_call(BinaryMerkleTree.verify, original_root, proof, repeat=repeat)
    update_stats = benchmark_call(tree.update, target_index, b"merkle_updated", repeat=max(1, min(3, repeat)))

    return {
        "scenario": "merkle_baseline",
        "parameters": {
            "leaf_count": leaf_count,
        },
        "metrics": {
            "build_time_mean_s": build_stats["mean_seconds"],
            "open_time_mean_s": open_stats["mean_seconds"],
            "verify_time_mean_s": verify_stats["mean_seconds"],
            "update_time_mean_s": update_stats["mean_seconds"],
            "proof_size_bytes": tree.proof_size_bytes(proof),
            "memory_usage_peak_build_bytes": peak_mem_build,
        },
        "correctness": {
            "proof_verifies": bool(BinaryMerkleTree.verify(original_root, proof)),
        },
    }


def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    vc = LatticeVectorCommitment(
        q=args.q,
        n=args.n,
        m=args.m,
        k=args.k,
        r_bound=args.r_bound,
        trapdoor_bound=args.trapdoor_bound,
        gamma=None,
        seed=args.seed,
    )

    report: Dict[str, Any] = {
        "implementation": {
            "language": "Python",
            "libraries": ["numpy", "hashlib", "tracemalloc", "time"],
            "platform_note": args.platform_note,
            "prototype_note": (
                "The executable prototype uses one global matrix A and precomputed short vectors R_j "
                "such that U_j = A R_j (mod q). This keeps the construction algebraically consistent "
                "for benchmarking and update experiments."
            ),
        },
        "runtime_parameters": {
            "q": args.q,
            "n": args.n,
            "m": args.m,
            "k": args.k,
            "depth": args.depth,
            "repeat": args.repeat,
            "r_bound": args.r_bound,
            "trapdoor_bound": args.trapdoor_bound,
            "seed": args.seed,
        },
        "experiments": [],
        "analytic_paper_profile": paper_profile(),
    }

    report["experiments"].append(run_node_experiment(vc, args.repeat))
    report["experiments"].append(run_tree_path_experiment(vc, args.depth, args.repeat))
    report["experiments"].append(run_multiple_openings_experiment(vc, args.openings, args.repeat))
    report["experiments"].append(run_merkle_baseline(args.merkle_leaf_count, args.repeat))

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prototype experiments for the lattice-based vector commitment paper.")
    parser.add_argument("--q", type=int, default=12289)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--m", type=int, default=48)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--openings", type=int, default=8)
    parser.add_argument("--merkle-leaf-count", type=int, default=256)
    parser.add_argument("--r-bound", type=int, default=2)
    parser.add_argument("--trapdoor-bound", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--platform-note", type=str, default="Fill in CPU, RAM, OS, and Python version before reporting.")
    parser.add_argument("--output", type=str, default="experiment_report.json")
    args = parser.parse_args()

    report = run_all(args)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
