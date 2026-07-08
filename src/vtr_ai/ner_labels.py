from __future__ import annotations

ENTITY_LABELS = [
    "TRIỆU_CHỨNG",
    "CHẨN_ĐOÁN",
    "THUỐC",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
]


def build_bio_labels() -> list[str]:
    labels = ["O"]
    for label in ENTITY_LABELS:
        labels.append(f"B-{label}")
        labels.append(f"I-{label}")
    return labels


BIO_LABELS = build_bio_labels()
LABEL_TO_ID = {label: index for index, label in enumerate(BIO_LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
