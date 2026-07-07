# Luồng Hybrid Ontology + AI Y Khoa

Tài liệu này chuyển pipeline hiện tại sang cách nhìn của bài báo về `ontology + knowledge base + inference engine`, nhưng vẫn giữ các bước NLP/AI cần thiết cho bài toán.

## 1. Luồng tổng thể

```mermaid
flowchart TD
    A["Văn bản y khoa tự do<br/>ghi chú bác sĩ, EHR, xét nghiệm"] --> B["Tiền xử lý văn bản"]
    B --> C["Trích xuất khái niệm y khoa"]
    C --> C1["TRIỆU_CHỨNG"]
    C --> C2["CHẨN_ĐOÁN"]
    C --> C3["THUỐC"]
    C --> C4["TÊN_XÉT_NGHIỆM"]
    C --> C5["KẾT_QUẢ_XÉT_NGHIỆM"]

    C --> D["Biểu diễn tri thức theo ontology"]
    D --> D1["Concepts"]
    D --> D2["Attributes"]
    D --> D3["Relations"]
    D --> D4["Context assertions"]

    D --> E["Ánh xạ chuẩn y khoa"]
    E --> E1["ICD-10 cho CHẨN_ĐOÁN"]
    E --> E2["RxNorm cho THUỐC"]

    E --> F["Tổ chức cơ sở tri thức"]
    F --> F1["Fact: bệnh nhân có triệu chứng gì"]
    F --> F2["Fact: được chẩn đoán gì"]
    F --> F3["Fact: đang hay đã dùng thuốc gì"]
    F --> F4["Fact: xét nghiệm nào có kết quả gì"]
    F --> F5["Rule: phủ định, tiền sử, người nhà"]

    F --> G["Bộ suy diễn"]
    G --> G1["Suy diễn assertion"]
    G --> G2["Suy diễn quan hệ khái niệm"]
    G --> G3["Loại bỏ diễn giải sai theo ngữ cảnh"]

    G --> H["Hậu xử lý và chuẩn hóa output"]
    H --> I["JSON đầu ra theo format đề bài"]
```

## 2. Ánh xạ từ bài báo sang hệ thống này

| Theo bài báo | Trong hệ thống hiện tại |
| --- | --- |
| Ontology | Schema y khoa cho thực thể, thuộc tính, quan hệ, assertion |
| Knowledge base | Fact trích từ văn bản + ICD-10 + RxNorm + từ điển alias |
| Inference engine | Assertion detection + relation reasoning + hậu kiểm ngữ cảnh |
| Mô hình hóa bài toán | Chuyển đoạn văn bản thành ca lâm sàng có cấu trúc |
| Ứng dụng | Xuất JSON chuẩn hóa phục vụ liên thông dữ liệu và AI downstream |

## 3. Luồng kỹ thuật sát với code baseline

```mermaid
flowchart LR
    A["Input .txt"] --> B["cli.py"]
    B --> C["preprocess.py"]
    C --> D["ner.py"]
    D --> E["pipeline.py"]
    E --> F["candidate_generation.py"]
    E --> G["assertion.py"]
    F --> H["postprocess.py"]
    G --> H
    H --> I["io_utils.py -> .json"]
```

## 4. Khung ontology đề xuất

- `PatientCase`
- `Symptom`
- `Diagnosis`
- `Drug`
- `LabTest`
- `LabResult`
- `AssertionContext`

Các quan hệ nên có:

- `has_symptom(PatientCase, Symptom)`
- `has_diagnosis(PatientCase, Diagnosis)`
- `uses_drug(PatientCase, Drug)`
- `has_lab_test(PatientCase, LabTest)`
- `has_lab_result(LabTest, LabResult)`
- `has_assertion(Entity, AssertionContext)`
- `mapped_to_icd10(Diagnosis, Code)`
- `mapped_to_rxnorm(Drug, Code)`

## 5. Bản ngắn để đưa vào slide

```mermaid
flowchart LR
    A["Free-form clinical text"] --> B["Entity extraction"]
    B --> C["Ontology representation"]
    C --> D["Knowledge base"]
    D --> E["Inference engine"]
    E --> F["Standardized JSON output"]
```

## 6. Câu mô tả ngắn có thể dùng trong báo cáo

Hệ thống được thiết kế theo hướng lai giữa AI xử lý ngôn ngữ và hệ cơ sở tri thức dựa trên ontology. Văn bản y khoa tự do trước hết được tiền xử lý và trích xuất các khái niệm lâm sàng, sau đó các khái niệm này được biểu diễn dưới dạng ontology, ánh xạ với các chuẩn ICD-10 và RxNorm, nạp vào cơ sở tri thức, rồi đưa qua bộ suy diễn để xác định ngữ cảnh và quan hệ trước khi sinh đầu ra JSON chuẩn hóa.
