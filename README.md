# VTR AI Baseline

Baseline end-to-end pipeline cho bài toán chuẩn hóa khái niệm y khoa từ văn bản tự do.

## Cách chạy

```bash
PYTHONPATH=src python -m vtr_ai.cli --input_dir test/input --output_dir test/output --config config.yaml
```

Hoặc sau khi cài editable:

```bash
pip install -e .
run_infer --input_dir test/input --output_dir test/output --config config.yaml
```

## Thành phần hiện có

- Tiền xử lý văn bản và ánh xạ offset normalized -> raw
- Rule-based entity extraction cho 5 nhãn bắt buộc
- Candidate mapping ICD-10/RxNorm bằng exact + fuzzy scoring
- Assertion detection cho `isNegated`, `isFamily`, `isHistorical`
- Hậu xử lý span và xuất JSON theo format đề bài
- Bộ test smoke/unit cho các case chính

## Ghi chú

- Baseline này chưa gắn mô hình fine-tuned. Các hook mở rộng đã được chừa sẵn trong pipeline để thay thế bộ nhận diện rule-based bằng model/reranker ở bước sau.
- `position` được lưu nội bộ theo quy ước `[start, end)` và xuất ra JSON theo cùng quy ước đó để đảm bảo `text == raw_text[start:end]`.

