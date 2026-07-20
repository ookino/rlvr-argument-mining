"""Measuring the argument graph.

Each function turns a graph into one number between 0 and 1. Those numbers are
what the scorer combines into the training signal.

WHY EVERY MEASUREMENT IS A FRACTION, NOT A YES/NO
-------------------------------------------------
The training method works by having the model produce several attempts at the
same question and comparing their scores. If every attempt scores identically,
there is nothing to compare and the model learns nothing from that batch. A
yes/no measurement makes ties very likely. Fractions make them rare.

This is not a style preference. A binary measurement here would quietly break
training in a way that looks like "the idea did not work".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import networkx as nx

from reward.ari import RelationResult
from reward.xaif_build import Trace

MAX_DEPTH = 5      # chains longer than this are capped; see support_depth


@dataclass
class Features:
    support_density: float      # how much of the trace is doing inferential work
    attachment: float           # how many steps are connected to anything at all
    connectivity: float         # how many steps reach the conclusion
    support_depth: float        # how long the reasoning chain is
    conflict_rate: float        # self-contradiction within the trace
    restatement_rate: float     # padding: the same point made twice

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def build_graph(trace: Trace, relations: RelationResult) -> nx.DiGraph:
    """One node per step, one edge per detected relation."""
    graph = nx.DiGraph()
    for idx, text in enumerate(trace.steps):
        graph.add_node(idx, text=text, is_conclusion=(idx == trace.conclusion_index))
    for rel in relations.relations:
        graph.add_edge(rel.source, rel.target, kind=rel.kind, conf=rel.confidence)
    return graph


def _support_graph(graph: nx.DiGraph) -> nx.DiGraph:
    """Just the support links. Contradiction and restatement are not inference."""
    keep = [(u, v) for u, v, d in graph.edges(data=True) if d["kind"] == "RA"]
    sub = nx.DiGraph()
    sub.add_nodes_from(graph.nodes)
    sub.add_edges_from(keep)
    return sub


def compute(trace: Trace, relations: RelationResult) -> Features:
    """All six measurements. Returns zeros for a trace too small to have structure."""
    if trace.is_degenerate:
        return Features(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    graph = build_graph(trace, relations)
    n = trace.n_steps
    support = _support_graph(graph)
    conclusion = trace.conclusion_index

    n_support = support.number_of_edges()
    n_conflict = sum(1 for *_, d in graph.edges(data=True) if d["kind"] == "CA")
    n_restate = sum(1 for *_, d in graph.edges(data=True) if d["kind"] == "MA")

    # How much inferential linking there is, relative to how much there could
    # be. Normalised by n-1 (a tree over n nodes) rather than by all possible
    # pairs, so a well-formed chain scores near 1 instead of near zero.
    support_density = min(1.0, n_support / max(1, n - 1))

    # Steps that participate in any support link at all. Catches floating
    # assertions: sentences that assert something and connect to nothing.
    attached = {u for u, v in support.edges()} | {v for u, v in support.edges()}
    attachment = len(attached) / n

    # Steps with a support path to the conclusion. This is the measurement
    # closest to "does the reasoning actually lead anywhere".
    reaching = sum(
        1
        for node in support.nodes
        if node != conclusion and nx.has_path(support, node, conclusion)
    )
    connectivity = reaching / max(1, n - 1)

    # Longest chain of support ending at the conclusion, capped. Rewards
    # reasoning that builds on itself over a flat list of assertions.
    support_depth = _longest_chain_to(support, conclusion) / MAX_DEPTH

    # Contradiction and restatement are normalised against the same n-1 scale.
    # Both are reported as-is here; the scorer decides their sign. Restatement
    # in particular is the padding signal: saying the same thing twice to
    # inflate the graph is the most obvious way to game this reward.
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
    """Length of the longest support chain terminating at the conclusion.

    Cycles should not occur (support runs forward through a trace) but the
    model can produce them, so this walks with a visited set rather than
    assuming acyclicity and crashing on a real trace at 2am.
    """
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
