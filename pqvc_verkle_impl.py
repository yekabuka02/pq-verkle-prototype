
from __future__ import annotations

import hashlib
import math
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def hash_to_scalar(message: bytes, q: int) -> int:
    digest = hashlib.sha256(message).digest()
    return int.from_bytes(digest, "big") % q


def field_element_num_bytes(q: int) -> int:
    return max(1, math.ceil(q.bit_length() / 8))


def serialize_vector(vec: np.ndarray, q: int) -> bytes:
    width = field_element_num_bytes(q)
    out = bytearray()
    for x in vec.tolist():
        out.extend(int(x % q).to_bytes(width, "big", signed=False))
    return bytes(out)


def centered_linf_norm(vec: np.ndarray, q: int) -> int:
    centered = ((vec.astype(np.int64) + q // 2) % q) - q // 2
    return int(np.max(np.abs(centered)))


def random_short_vector(length: int, bound: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(-bound, bound + 1, size=length, dtype=np.int64)


@dataclass
class VCState:
    messages: List[bytes]
    coeffs: List[int]
    r: np.ndarray
    commitment: np.ndarray


@dataclass
class VCOpenProof:
    index: int
    proof: np.ndarray
    message: bytes


class LatticeVectorCommitment:
    """
    Executable prototype for experiments.

    This prototype uses one global public matrix A and precomputed short vectors R_j such that
    U_j = A R_j (mod q). It preserves algebraic correctness and supports update / proof-update
    experiments, while remaining lightweight enough for benchmarking in Python.
    """

    def __init__(
        self,
        q: int,
        n: int,
        m: int,
        k: int,
        *,
        r_bound: int = 2,
        trapdoor_bound: int = 2,
        gamma: Optional[int] = None,
        seed: int = 12345,
    ) -> None:
        self.q = int(q)
        self.n = int(n)
        self.m = int(m)
        self.k = int(k)
        self.l = math.ceil(math.log2(self.q))
        self.N = self.m + self.n * self.l
        self.r_bound = int(r_bound)
        self.trapdoor_bound = int(trapdoor_bound)
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)

        self.A = self.rng.integers(0, self.q, size=(self.n, self.N), dtype=np.int64)
        self.R = [random_short_vector(self.N, self.trapdoor_bound, self.rng) for _ in range(self.k)]
        self.U = [self._mul_A(rj) for rj in self.R]

    def _mod_q(self, vec: np.ndarray) -> np.ndarray:
        return np.mod(vec, self.q).astype(np.int64)

    def _mul_A(self, vec: np.ndarray) -> np.ndarray:
        return self._mod_q(self.A @ np.mod(vec, self.q))

    def size_report(self) -> Dict[str, Any]:
        width = field_element_num_bytes(self.q)
        public_parameter_field_elements = self.n * self.N + self.k * self.n
        public_parameter_bytes = public_parameter_field_elements * width
        return {
            "q": self.q,
            "n": self.n,
            "m": self.m,
            "k": self.k,
            "l": self.l,
            "N": self.N,
            "field_element_bytes": width,
            "public_parameter_field_elements": public_parameter_field_elements,
            "public_parameter_bytes": public_parameter_bytes,
            "commitment_field_elements": self.n,
            "commitment_bytes": self.n * width,
            "proof_field_elements": self.N,
            "proof_bytes": self.N * width,
            "auxiliary_preimages_count": self.k,
            "auxiliary_preimages_field_elements": self.k * self.N,
            "auxiliary_preimages_bytes": self.k * self.N * width,
        }

    def commit(self, messages: Sequence[bytes]) -> VCState:
        if len(messages) != self.k:
            raise ValueError(f"Expected exactly {self.k} messages, got {len(messages)}")

        coeffs = [hash_to_scalar(m, self.q) for m in messages]
        r = random_short_vector(self.N, self.r_bound, self.rng)

        total = np.zeros(self.n, dtype=np.int64)
        for a_j, U_j in zip(coeffs, self.U):
            total = self._mod_q(total + (a_j * U_j))
        commitment = self._mod_q(total + self._mul_A(r))

        return VCState(messages=list(messages), coeffs=coeffs, r=r, commitment=commitment)

    def open(self, state: VCState, index: int) -> VCOpenProof:
        if not (0 <= index < self.k):
            raise IndexError("index out of range")

        proof = np.array(state.r, copy=True, dtype=np.int64)
        for j, a_j in enumerate(state.coeffs):
            if j == index:
                continue
            proof = self._mod_q(proof + (a_j * self.R[j]))

        return VCOpenProof(index=index, proof=proof, message=state.messages[index])

    def verify(self, commitment: np.ndarray, index: int, message: bytes, proof: VCOpenProof) -> bool:
        if proof.index != index:
            return False
        a_i = hash_to_scalar(message, self.q)
        lhs = self._mod_q(commitment)
        rhs = self._mod_q(self._mul_A(proof.proof) + (a_i * self.U[index]))

        if self.gamma is not None:
            if centered_linf_norm(proof.proof, self.q) > self.gamma:
                return False
        return np.array_equal(lhs, rhs)

    def update(self, state: VCState, index: int, new_message: bytes) -> Tuple[VCState, int]:
        if not (0 <= index < self.k):
            raise IndexError("index out of range")

        new_coeff = hash_to_scalar(new_message, self.q)
        old_coeff = state.coeffs[index]
        delta = (new_coeff - old_coeff) % self.q

        new_commitment = self._mod_q(state.commitment + delta * self.U[index])
        new_messages = list(state.messages)
        new_messages[index] = new_message
        new_coeffs = list(state.coeffs)
        new_coeffs[index] = new_coeff

        new_state = VCState(
            messages=new_messages,
            coeffs=new_coeffs,
            r=np.array(state.r, copy=True, dtype=np.int64),
            commitment=new_commitment,
        )
        return new_state, delta

    def proof_update(self, proof: VCOpenProof, opened_index: int, updated_index: int, delta: int) -> VCOpenProof:
        if opened_index != proof.index:
            raise ValueError("opened_index must match proof.index")

        if opened_index == updated_index:
            return VCOpenProof(index=proof.index, proof=np.array(proof.proof, copy=True), message=proof.message)

        updated = self._mod_q(proof.proof + (delta * self.R[updated_index]))
        return VCOpenProof(index=proof.index, proof=updated, message=proof.message)

    def serialize_commitment(self, commitment: np.ndarray) -> bytes:
        return serialize_vector(commitment, self.q)


def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


@dataclass
class MerkleProof:
    index: int
    leaf: bytes
    siblings: List[Tuple[bytes, str]]


class BinaryMerkleTree:
    def __init__(self, leaves: Sequence[bytes]) -> None:
        if len(leaves) == 0:
            raise ValueError("At least one leaf is required")
        self.leaves = list(leaves)
        self.levels: List[List[bytes]] = []
        self._build()

    def _build(self) -> None:
        current = [_sha256(leaf) for leaf in self.leaves]
        self.levels = [current]
        while len(current) > 1:
            nxt = []
            for i in range(0, len(current), 2):
                left = current[i]
                right = current[i + 1] if i + 1 < len(current) else current[i]
                nxt.append(_sha256(left + right))
            self.levels.append(nxt)
            current = nxt

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    def open(self, index: int) -> MerkleProof:
        if not (0 <= index < len(self.leaves)):
            raise IndexError("index out of range")

        siblings: List[Tuple[bytes, str]] = []
        idx = index
        for level in self.levels[:-1]:
            if idx % 2 == 0:
                sib_idx = idx + 1 if idx + 1 < len(level) else idx
                siblings.append((level[sib_idx], "R"))
            else:
                sib_idx = idx - 1
                siblings.append((level[sib_idx], "L"))
            idx //= 2
        return MerkleProof(index=index, leaf=self.leaves[index], siblings=siblings)

    @staticmethod
    def verify(root: bytes, proof: MerkleProof) -> bool:
        h = _sha256(proof.leaf)
        for sib, side in proof.siblings:
            if side == "R":
                h = _sha256(h + sib)
            else:
                h = _sha256(sib + h)
        return h == root

    def update(self, index: int, new_leaf: bytes) -> None:
        if not (0 <= index < len(self.leaves)):
            raise IndexError("index out of range")
        self.leaves[index] = new_leaf
        self._build()

    def proof_size_bytes(self, proof: MerkleProof) -> int:
        return len(proof.siblings) * 32


@dataclass
class TreeNode:
    state: VCState
    children: List[Any]
    level: int
    parent: Optional["TreeNode"] = None
    index_in_parent: Optional[int] = None


class VerkleTreePrototype:
    """
    Fixed-arity tree prototype.
    Parent nodes commit to serialized child commitments.
    Leaf-parent nodes commit directly to leaf messages.
    """

    def __init__(self, vc: LatticeVectorCommitment, depth: int, leaves: Sequence[bytes]) -> None:
        self.vc = vc
        self.arity = vc.k
        self.depth = int(depth)
        expected_leaves = self.arity ** self.depth
        if len(leaves) != expected_leaves:
            raise ValueError(f"Expected {expected_leaves} leaves for depth={depth} and arity={self.arity}, got {len(leaves)}")
        self.leaves = list(leaves)
        self.root = self._build_tree()

    def _build_tree(self) -> TreeNode:
        items: List[Any] = list(self.leaves)
        level = self.depth
        while level > 0:
            parent_nodes: List[TreeNode] = []
            for start in range(0, len(items), self.arity):
                chunk = items[start:start + self.arity]
                if level == self.depth:
                    messages = [bytes(x) for x in chunk]
                else:
                    messages = [self.vc.serialize_commitment(node.state.commitment) for node in chunk]
                state = self.vc.commit(messages)
                node = TreeNode(state=state, children=list(chunk), level=level - 1)
                for idx, child in enumerate(chunk):
                    if isinstance(child, TreeNode):
                        child.parent = node
                        child.index_in_parent = idx
                parent_nodes.append(node)
            items = parent_nodes
            level -= 1
        if len(items) != 1:
            raise RuntimeError("Tree construction failed")
        return items[0]

    def open_leaf_path(self, leaf_index: int) -> List[Tuple[np.ndarray, int, bytes, VCOpenProof]]:
        if not (0 <= leaf_index < len(self.leaves)):
            raise IndexError("leaf_index out of range")

        digits: List[int] = []
        idx = leaf_index
        for _ in range(self.depth):
            digits.append(idx % self.arity)
            idx //= self.arity
        digits = list(reversed(digits))

        proofs: List[Tuple[np.ndarray, int, bytes, VCOpenProof]] = []
        node = self.root
        for level_idx, child_pos in enumerate(digits):
            child = node.children[child_pos]
            if isinstance(child, TreeNode):
                msg = self.vc.serialize_commitment(child.state.commitment)
            else:
                msg = bytes(child)
            proof = self.vc.open(node.state, child_pos)
            proofs.append((np.array(node.state.commitment, copy=True), child_pos, msg, proof))
            if isinstance(child, TreeNode):
                node = child
        return proofs

    def verify_leaf_path(self, path_proofs: Sequence[Tuple[np.ndarray, int, bytes, VCOpenProof]]) -> bool:
        return all(self.vc.verify(commitment, index, message, proof) for commitment, index, message, proof in path_proofs)

    def update_leaf(self, leaf_index: int, new_leaf: bytes) -> None:
        if not (0 <= leaf_index < len(self.leaves)):
            raise IndexError("leaf_index out of range")
        self.leaves[leaf_index] = new_leaf
        # Rebuild for robustness. This keeps the prototype simple and deterministic.
        self.root = self._build_tree()


def benchmark_call(func, *args, repeat: int = 5, **kwargs) -> Dict[str, float]:
    times = []
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)
    return {
        "mean_seconds": float(np.mean(times)),
        "min_seconds": float(np.min(times)),
        "max_seconds": float(np.max(times)),
        "result": result,
    }


def measure_peak_memory_bytes(func, *args, **kwargs) -> Tuple[int, Any]:
    tracemalloc.start()
    try:
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return int(peak), result


def paper_profile(q: int = 2**23 + 9, n: int = 512, m: int = 1024, k: int = 256) -> Dict[str, Any]:
    l = math.ceil(math.log2(q))
    N = m + n * l
    width = field_element_num_bytes(q)
    matrix_elems = n * N
    all_A_elems = k * matrix_elems
    all_U_elems = k * n
    all_R_elems = k * (k - 1) * N
    return {
        "q": q,
        "n": n,
        "m": m,
        "k": k,
        "l": l,
        "N": N,
        "field_element_bytes": width,
        "single_A_elements": matrix_elems,
        "all_A_elements": all_A_elems,
        "all_U_elements": all_U_elems,
        "all_R_elements_if_pairwise_precomputed": all_R_elems,
        "commitment_elements": n,
        "proof_elements": N,
        "single_A_bytes": matrix_elems * width,
        "all_A_bytes": all_A_elems * width,
        "commitment_bytes": n * width,
        "proof_bytes": N * width,
        "all_R_bytes_if_pairwise_precomputed": all_R_elems * width,
    }
