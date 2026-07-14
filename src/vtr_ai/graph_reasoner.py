from __future__ import annotations

from .graph_retriever import GraphRetriever
from .patient_graph import PatientGraph
from .schemas import Document, Entity
from .section_parser import find_enclosing_clause


class GraphReasoner:
    def __init__(self, retriever: GraphRetriever) -> None:
        self.retriever = retriever

    def apply(self, entities: list[Entity], document: Document, patient_graph: PatientGraph) -> list[Entity]:
        graph_clause_kinds = self._build_graph_clause_kinds(patient_graph)
        for index, entity in enumerate(entities):
            if entity.entity_type not in {"CHẨN_ĐOÁN", "THUỐC"}:
                continue
            clause = find_enclosing_clause(document.clauses, entity.start, entity.end)
            clause_kind = graph_clause_kinds.get(index)
            if clause_kind is None:
                clause_kind = clause.anchor_kind if clause is not None else None
            retrieval = self.retriever.retrieve(entity, clause_kind)
            if not retrieval.candidate_codes:
                continue
            if not entity.candidates:
                entity.candidates = retrieval.candidate_codes[:3]
                continue
            base_rank = {code: index for index, code in enumerate(entity.candidates)}
            combined = []
            seen: set[str] = set()
            for code in entity.candidates + retrieval.candidate_codes:
                if code in seen:
                    continue
                seen.add(code)
                retrieval_score = retrieval.support.get(code, 0.0)
                rank_bonus = max(0.0, 0.3 - 0.1 * base_rank.get(code, 3))
                combined.append((code, retrieval_score + rank_bonus))
            combined.sort(key=lambda item: (-item[1], item[0]))
            if combined and combined[0][1] < 0.45:
                entity.candidates = []
            else:
                entity.candidates = [code for code, _ in combined[:3]]
        return entities

    @staticmethod
    def _build_graph_clause_kinds(patient_graph: PatientGraph) -> dict[int, str]:
        clause_kind_by_id = {
            node.node_id: str(node.attributes.get("anchor_kind", ""))
            for node in patient_graph.nodes
            if node.node_type == "clause"
        }
        result: dict[int, str] = {}
        for edge in patient_graph.edges:
            if edge.relation != "mentions_entity" or not edge.source_id.startswith("clause:"):
                continue
            try:
                entity_index = int(edge.target_id.split(":", 1)[1])
            except (IndexError, ValueError):
                continue
            clause_kind = clause_kind_by_id.get(edge.source_id, "")
            if clause_kind:
                result[entity_index] = clause_kind
        return result
