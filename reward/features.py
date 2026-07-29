"""Turn the argument graph into six numbers (0..1 each).

All six are fractions on purpose, not yes/no. GRPO compares several attempts per
question; if they all score the same it learns nothing, and yes/no scores tie
constantly. So a binary measure would break training silently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import networkx as nx

from reward.ari import RelationResult
from reward.xaif_build import Trace

MAX_DEPTH = 5      # cap on chain length (see support_depth)


@dataclass
class Features:
    support_density: float      # how much of the trace is inferential
    attachment: float           # how many steps connect to anything
    connectivity: float         # how many steps reach the conclusion
    support_depth: float        # how long the reasoning chain is
    conflict_rate: float        # self-contradiction
    restatement_rate: float     # repetition / padding

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def build_graph(trace: Trace, relations: RelationResult) -> nx.DiGraph:
    # node per step, edge per relation
    graph = nx.DiGraph()
    for idx, text in enumerate(trace.steps):
        graph.add_node(idx, text=text, is_conclusion=(idx == trace.conclusion_index))
    for rel in relations.relations:
        graph.add_edge(rel.source, rel.target, kind=rel.kind, conf=rel.confidence)
    return graph


def _support_graph(graph: nx.DiGraph) -> nx.DiGraph:
    # just the support (RA) links - conflict and restatement aren't inference
    keep = [(u, v) for u, v, d in graph.edges(data=True) if d["kind"] == "RA"]
    sub = nx.DiGraph()
    sub.add_nodes_from(graph.nodes)
    sub.add_edges_from(keep)
    return sub


def compute(trace: Trace, relations: RelationResult) -> Features:
    if trace.is_degenerate:
        return Features(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    graph = build_graph(trace, relations)
    n = trace.n_steps
    support = _support_graph(graph)
    conclusion = trace.conclusion_index

    n_support = support.number_of_edges()
    n_conflict = sum(1 for *_, d in graph.edges(data=True) if d["kind"] == "CA")
    n_restate = sum(1 for *_, d in graph.edges(data=True) if d["kind"] == "MA")

    # normalise by n-1 (a tree over n nodes has n-1 edges) so a clean chain
    # scores near 1, not near 0
    support_density = min(1.0, n_support / max(1, n - 1))

    # steps that are in any support link - catches floating assertions
    attached = {u for u, v in support.edges()} | {v for u, v in support.edges()}
    attachment = len(attached) / n

    # steps with a support path to the conclusion
    reaching = sum(
        1
        for node in support.nodes
        if node != conclusion and nx.has_path(support, node, conclusion)
    )
    connectivity = reaching / max(1, n - 1)

    # longest support chain ending at the conclusion, capped
    support_depth = _longest_chain_to(support, conclusion) / MAX_DEPTH

    # conflict and restatement: reported as-is, the scorer flips their sign
    conflict_rate = min(1.0, n_conflict / max(1, n - 1))
    restatement_rate = min(1.0, n_restate / max(1, n - 1))

    return Features(
        support_density=support_density,
        attachment=attachment,
        connectivity=connectivity,
        support_depth=min(1.0, support_depth),
        conflict_rate=conflict_rate,
        restatement_rate=restatement_rate,
    )


def _longest_chain_to(support: nx.DiGraph, conclusion: int) -> int:
    # walk backwards from the conclusion. use a visited set in case the model
    # produced a cycle, so it doesn't loop forever.
    if conclusion is None or conclusion not in support:
        return 0

    best = 0
    stack = [(conclusion, 0, {conclusion})]
    while stack:
        node, depth, seen = stack.pop()
        best = max(best, depth)
        if depth >= MAX_DEPTH:
            continue
        for predecessor in support.predecessors(node):
            if predecessor not in seen:
                stack.append((predecessor, depth + 1, seen | {predecessor}))
    return best
