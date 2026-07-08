from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import Document, Entity


@dataclass(slots=True)
class GraphNode:
    node_id: str
    node_type: str
    text: str
    start: int | None = None
    end: int | None = None
    attributes: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class GraphEdge:
    source_id: str
    relation: str
    target_id: str
    attributes: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class PatientGraph:
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    def add_node(self, node: GraphNode) -> None:
        self.nodes.append(node)

    def add_edge(self, edge: GraphEdge) -> None:
        self.edges.append(edge)


def build_patient_graph(document: Document, entities: list[Entity]) -> PatientGraph:
    graph = PatientGraph()

    for index, section in enumerate(document.sections):
        graph.add_node(
            GraphNode(
                node_id=f"section:{index}",
                node_type="section",
                text=section.header or document.text[section.start:section.end].strip()[:80],
                start=section.start,
                end=section.end,
                attributes={
                    "kind": section.kind,
                    "default_assertions": list(section.default_assertions),
                    "expected_types": list(section.expected_types),
                },
            )
        )

    for index, clause in enumerate(document.clauses):
        clause_id = f"clause:{index}"
        graph.add_node(
            GraphNode(
                node_id=clause_id,
                node_type="clause",
                text=clause.text,
                start=clause.start,
                end=clause.end,
                attributes={
                    "section_kind": clause.section_kind,
                    "anchor_text": clause.anchor_text,
                    "anchor_kind": clause.anchor_kind,
                    "default_assertions": list(clause.default_assertions),
                    "expected_types": list(clause.expected_types),
                },
            )
        )
        for section_index, section in enumerate(document.sections):
            if section.start <= clause.start and clause.end <= section.end:
                graph.add_edge(GraphEdge(source_id=f"section:{section_index}", relation="contains_clause", target_id=clause_id))
                break

    for index, anchor in enumerate(document.anchors):
        anchor_id = f"anchor:{index}"
        graph.add_node(
            GraphNode(
                node_id=anchor_id,
                node_type="anchor",
                text=anchor.text,
                start=anchor.start,
                end=anchor.end,
                attributes={
                    "kind": anchor.kind,
                    "expected_types": list(anchor.expected_types),
                },
            )
        )
        for clause_index, clause in enumerate(document.clauses):
            if clause.start <= anchor.start and anchor.end <= clause.end:
                graph.add_edge(GraphEdge(source_id=f"clause:{clause_index}", relation="has_anchor", target_id=anchor_id))
                break

    for index, entity in enumerate(entities):
        entity_id = f"entity:{index}"
        graph.add_node(
            GraphNode(
                node_id=entity_id,
                node_type="entity",
                text=entity.text,
                start=entity.start,
                end=entity.end,
                attributes={
                    "entity_type": entity.entity_type,
                    "assertions": list(entity.assertions),
                    "candidates": list(entity.candidates),
                },
            )
        )
        for clause_index, clause in enumerate(document.clauses):
            if clause.start <= entity.start and entity.end <= clause.end:
                graph.add_edge(GraphEdge(source_id=f"clause:{clause_index}", relation="mentions_entity", target_id=entity_id))
                break

    return graph
