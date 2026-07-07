from __future__ import annotations

from .schemas import PatientCase


class RuleEngine:
    def apply(self, patient_case: PatientCase) -> PatientCase:
        for entity in patient_case.entities:
            if entity.entity_type not in {"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"}:
                entity.assertions = []
            if entity.entity_type not in {"CHẨN_ĐOÁN", "THUỐC"}:
                entity.candidates = []
        return patient_case
