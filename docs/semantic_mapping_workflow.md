# Semantic ICD/RxNorm Workflow

Mục tiêu: dùng bộ `gold_dir` đã audit để chuẩn bị:

- `zero_shot_eval.jsonl`
- `few_shot_eval.jsonl`
- `sft_train.jsonl`
- `sft_dev.jsonl`

cho bài semantic mapping `CHẨN_ĐOÁN -> ICD-10` và `THUỐC -> RxNorm`.

## Input kỳ vọng

- `input_dir`: thư mục `.txt`
- `gold_dir`: thư mục `.json` theo schema Viettel
- `config`: config trỏ tới KB ICD/RxNorm

## CLI

```bash
PYTHONPATH=src python -m vtr_ai.prepare_semantic_datasets \
  --input_dir /Users/springwang/Documents/VTR/review_packet_68_100_split/txt \
  --gold_dir /Users/springwang/Documents/VTR/review_packet_68_100_split/json \
  --config /Users/springwang/Documents/VTR/tmp/config.standard.yaml \
  --output_dir /Users/springwang/Documents/VTR/artifacts/semantic_68_100 \
  --shortlist_size 10 \
  --shortlist_min_confidence 0.1 \
  --support_size_per_type 6 \
  --shots_per_query 4
```

## Output chính

- `semantic_examples.jsonl`: toàn bộ example semantic có gold code
- `semantic_unmapped.jsonl`: các entity `CHẨN_ĐOÁN/THUỐC` chưa có gold candidate
- `train.jsonl`, `dev.jsonl`, `test.jsonl`: split nền
- `support.jsonl`: pool ví dụ few-shot
- `zero_shot_eval.jsonl`: prompt-ready eval không có ví dụ
- `few_shot_eval.jsonl`: prompt-ready eval có ví dụ support
- `sft_train.jsonl`, `sft_dev.jsonl`: chat-format cho fine-tune `<9B`
- `summary.json`: thống kê số lượng

## Khuyến nghị dùng với model <9B

- Fine-tune trên `sft_train.jsonl`
- Theo dõi nhanh trên `sft_dev.jsonl`
- Đánh giá zero-shot bằng `zero_shot_eval.jsonl`
- Đánh giá few-shot bằng `few_shot_eval.jsonl`
- Giữ `shortlist` trong prompt để model chỉ chọn trong candidate đã retrieve, tránh hallucinate mã
