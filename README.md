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

## Tự động hoá end-to-end

Để chạy từ đầu đến cuối theo đúng định dạng nộp Viettel:

```bash
PYTHONPATH=src python package_submission.py \
  --input_dir /Users/springwang/Downloads/input \
  --output_dir /Users/springwang/Documents/VTR/output \
  --zip_path /Users/springwang/Documents/VTR/output.zip \
  --config config.yaml
```

Hoặc sau khi cài editable:

```bash
run_submit \
  --input_dir /Users/springwang/Downloads/input \
  --output_dir /Users/springwang/Documents/VTR/output \
  --zip_path /Users/springwang/Documents/VTR/output.zip \
  --config config.yaml
```

Luồng này sẽ:

- xóa `output/` cũ nếu có
- chạy inference cho toàn bộ `*.txt`
- kiểm tra số lượng file, schema JSON, offset `position`, và ràng buộc `assertions/candidates`
- nén thành `output.zip` với cấu trúc `output/*.json`

## Thành phần hiện có

- Tiền xử lý văn bản và ánh xạ offset normalized -> raw
- Rule-based entity extraction cho 5 nhãn bắt buộc
- Tầng mở rộng viết tắt y khoa tiếng Việt trước NER
- Interface backend cho `rule`, `model`, `hybrid` NER
- Loader checkpoint/runtime cho NER ở `model_ner.py`
- Candidate mapping ICD-10/RxNorm bằng exact + fuzzy scoring
- Assertion detection cho `isNegated`, `isFamily`, `isHistorical`
- Validation output theo format nộp Viettel
- Packaging tự động thành `output.zip`
- Hậu xử lý span và xuất JSON theo format đề bài
- Bộ test smoke/unit cho các case chính

## Ghi chú

- Baseline này chưa gắn mô hình fine-tuned. Các hook mở rộng đã được chừa sẵn trong pipeline để thay thế bộ nhận diện rule-based bằng model/reranker ở bước sau.
- `model` hiện hỗ trợ 2 provider:
  - `lexical`: runtime nhẹ, nạp từ checkpoint JSON local
  - `transformers`: loader cho local Hugging Face token-classification checkpoint
- `transformers` có metadata label map riêng ở `transformers_ner_metadata.json` để map nhãn checkpoint về đúng 5 nhãn của bài Viettel.
- Nếu `metadata_path` không khai báo, runtime sẽ tự tìm `transformers_ner_metadata.json` trong thư mục checkpoint.
- Khi đã có model thật, chỉ cần đổi `ner.provider`, `ner.checkpoint_path`, và giữ nguyên pipeline còn lại.
- `hybrid` sẽ tự fallback về rules nếu model backend không load được và `fallback_to_rules: true`.
- `position` được lưu nội bộ theo quy ước `[start, end)` và xuất ra JSON theo cùng quy ước đó để đảm bảo `text == raw_text[start:end]`.

## Huấn luyện NER

Format dữ liệu train mẫu:

```json
{"text":"Bệnh nhân có tiền sử THA và rung nhĩ.","entities":[{"position":[21,24],"type":"CHẨN_ĐOÁN"},{"position":[29,37],"type":"CHẨN_ĐOÁN"}]}
```

Mỗi dòng là một JSON object trong file `.jsonl`. Ví dụ mẫu có ở:

- [ner_train_sample.jsonl](/Users/springwang/Documents/VTR/src/vtr_ai/data/examples/ner_train_sample.jsonl)

Huấn luyện token-classification checkpoint:

```bash
train_ner \
  --train_jsonl /Users/springwang/Documents/VTR/src/vtr_ai/data/examples/ner_train_sample.jsonl \
  --model_name vinai/phobert-base-v2 \
  --output_dir /Users/springwang/Documents/VTR/artifacts/ner-phobert \
  --num_train_epochs 3
```

Nếu chưa có dev set riêng, có thể tách trực tiếp từ train:

```bash
train_ner \
  --train_jsonl /Users/springwang/Documents/VTR/src/vtr_ai/data/examples/ner_train_sample.jsonl \
  --model_name vinai/phobert-base-v2 \
  --output_dir /Users/springwang/Documents/VTR/artifacts/ner-phobert \
  --validation_ratio 0.2 \
  --seed 42
```

Nếu đã có dev set riêng:

```bash
train_ner \
  --train_jsonl /Users/springwang/Documents/VTR/data/ner_train.jsonl \
  --eval_jsonl /Users/springwang/Documents/VTR/data/ner_dev.jsonl \
  --model_name vinai/phobert-base-v2 \
  --output_dir /Users/springwang/Documents/VTR/artifacts/ner-phobert
```

Khi có tập eval, script sẽ evaluate theo từng epoch và lưu `eval_metrics.json` trong thư mục checkpoint.

Sau khi train xong, cấu hình:

```yaml
ner:
  backend: model
  provider: transformers
  checkpoint_path: /Users/springwang/Documents/VTR/artifacts/ner-phobert
  metadata_path: /Users/springwang/Documents/VTR/artifacts/ner-phobert/transformers_ner_metadata.json
```

Suy luận NER riêng lẻ:

```bash
infer_ner --text "Bệnh nhân khó thở và dùng aspirin 325mg." --config config.yaml
```

## Build KB Chuẩn ICD-10 / RxNorm

CLI dựng knowledge base chuẩn:

```bash
build_standard_kb \
  --icd10_zip /path/to/icd10cm-Code\ Descriptions-2026.zip \
  --icd10_seed_aliases /Users/springwang/Documents/VTR/src/vtr_ai/data/icd10_sample.json \
  --icd10_output /Users/springwang/Documents/VTR/src/vtr_ai/data/icd10_standard.json
```

Nếu có sẵn gói RxNorm Prescribable chính thức:

```bash
build_standard_kb \
  --rxnorm_zip /path/to/RxNorm_full_prescribe_current.zip \
  --rxnorm_seed_aliases /Users/springwang/Documents/VTR/src/vtr_ai/data/rxnorm_sample.json \
  --rxnorm_output /Users/springwang/Documents/VTR/src/vtr_ai/data/rxnorm_standard.json
```

Nguồn chính thống dùng để build:

- ICD-10-CM code descriptions từ CDC/NCHS
- RxNorm Prescribable release từ NLM

Builder sẽ:

- parse release zip chính thức
- chuẩn hoá format code
- merge thêm alias seed tiếng Việt hiện có theo `code`
- xuất về đúng schema JSON mà pipeline đang dùng

Kiểm tra nhanh checkpoint/runtime trước khi chạy full pipeline:

```bash
check_ner_runtime \
  --config /Users/springwang/Documents/VTR/config.yaml \
  --text "Bệnh nhân khó thở và rung nhĩ."
```

Lệnh này hữu ích để xác nhận:

- checkpoint có tồn tại và đủ file bắt buộc hay không
- runtime thực tế đang load class nào
- metadata label map có được dò đúng từ checkpoint không
- prediction smoke-test có ra đúng 5 nhãn chuẩn hoá hay không

## Đánh giá offline

Khi đã có thư mục nhãn chuẩn `gold_dir` theo đúng format Viettel (`*.json` song song với `input/*.txt`), có thể đo nhanh các chỉ số chính:

```bash
eval_outputs \
  --input_dir /Users/springwang/Downloads/input \
  --pred_dir /Users/springwang/Documents/VTR/output \
  --gold_dir /Users/springwang/Documents/VTR/gold
```

CLI này hiện trả về JSON summary cho:

- `entity_span`: đúng span
- `entity_span_type`: đúng span + type
- `entity_record`: đúng full record gồm text, span, type, assertions, candidates
- `assertions`: micro-F1 trên từng nhãn `isNegated/isFamily/isHistorical`
- `candidates`: micro-F1 trên từng mã ICD-10/RxNorm gắn vào đúng thực thể

## Phân tích lỗi chưa map mã

Khi cần soi các thực thể `CHẨN_ĐOÁN` hoặc `THUỐC` còn `candidates: []`, có thể dùng:

```bash
analyze_outputs \
  --input_dir /Users/springwang/Downloads/input \
  --output_dir /Users/springwang/Documents/VTR/output \
  --entity_type CHẨN_ĐOÁN \
  --limit 20 \
  --output_path /Users/springwang/Documents/VTR/reports/unmatched_diagnosis.json
```

Report sẽ chứa:

- `file_name`
- `text`
- `position`
- `assertions`
- `context` lấy trực tiếp từ raw text quanh thực thể

Workflow này hữu ích để rà nhanh các case còn mơ hồ trước khi quyết định:

- thêm alias vào KB
- sửa rule extractor
- hoặc chủ động giữ `candidates: []` để bảo toàn precision

## Audit Sau Mỗi Lần Chạy Full Input

Để validate output và sinh luôn các report audit trong một lệnh:

```bash
audit_outputs \
  --input_dir /Users/springwang/Downloads/input \
  --output_dir /Users/springwang/Documents/VTR/output \
  --report_dir /Users/springwang/Documents/VTR/reports
```

Lệnh này sẽ tạo:

- `summary.json`: số file và số entity theo từng type
- `unmatched_diagnosis.json`: các `CHẨN_ĐOÁN` chưa map mã
- `unmatched_drugs.json`: các `THUỐC` chưa map mã

## Workflow Một Lệnh

Để chạy toàn bộ luồng `infer -> validate -> zip -> audit` trong một lệnh:

```bash
run_workflow \
  --input_dir /Users/springwang/Downloads/input \
  --output_dir /Users/springwang/Documents/VTR/output \
  --zip_path /Users/springwang/Documents/VTR/output.zip \
  --report_dir /Users/springwang/Documents/VTR/reports \
  --config /Users/springwang/Documents/VTR/config.yaml
```

## Bootstrap Dev Set Từ Output Hiện Tại

Khi muốn lấy prediction hiện tại làm điểm khởi đầu để chỉnh tay thành dev/gold set:

```bash
bootstrap_annotations \
  --input_dir /Users/springwang/Downloads/input \
  --output_dir /Users/springwang/Documents/VTR/output \
  --output_path /Users/springwang/Documents/VTR/reports/bootstrap_annotations.jsonl
```

Nếu muốn reviewer sửa trực tiếp trên trường đích `gold_entities` nhưng vẫn giữ `predicted_entities` để đối chiếu:

```bash
bootstrap_annotations \
  --input_dir /Users/springwang/Downloads/input \
  --output_dir /Users/springwang/Documents/VTR/output \
  --output_path /Users/springwang/Documents/VTR/reports/bootstrap_annotations.jsonl \
  --seed_gold_entities
```

Trước khi convert review JSONL sang `gold_dir` hoặc `train_jsonl`, có thể kiểm tra nhanh offset/schema:

```bash
validate_review_jsonl \
  --review_jsonl /Users/springwang/Documents/VTR/reports/dev_subset.jsonl
```

Mỗi dòng JSONL sẽ gồm:

- `file_name`
- `text`
- `predicted_entities`

Workflow này phù hợp để:

- mở trong editor và sửa tay dần
- chuyển thành tập `gold_dir` nhỏ
- hoặc convert tiếp sang format train cho NER/assertion model

Sau khi sửa tay file JSONL, có thể convert lại thành `gold_dir` để dùng cho `eval_outputs`:

```bash
review_to_gold \
  --review_jsonl /Users/springwang/Documents/VTR/reports/bootstrap_annotations.jsonl \
  --output_dir /Users/springwang/Documents/VTR/gold
```

Tool này ưu tiên đọc `gold_entities` nếu anh thêm field đó khi review; nếu không có thì nó dùng `predicted_entities`.

Khi đã có `gold_dir`, có thể convert sang JSONL để huấn luyện NER:

```bash
gold_to_ner_jsonl \
  --input_dir /Users/springwang/Downloads/input \
  --gold_dir /Users/springwang/Documents/VTR/gold \
  --output_path /Users/springwang/Documents/VTR/reports/ner_train_from_gold.jsonl
```

File JSONL này dùng trực tiếp cho `train_ner`.

## Chuẩn bị semantic ICD/RxNorm cho zero-shot, few-shot, và SFT

Khi đã có `input_dir` + `gold_dir` đã audit, có thể dựng bộ semantic mapping cho `CHẨN_ĐOÁN` và `THUỐC`:

```bash
prepare_semantic_datasets \
  --input_dir /Users/springwang/Documents/VTR/review_packet_68_100_split/txt \
  --gold_dir /Users/springwang/Documents/VTR/review_packet_68_100_split/json \
  --config /Users/springwang/Documents/VTR/tmp/config.standard.yaml \
  --output_dir /Users/springwang/Documents/VTR/artifacts/semantic_68_100 \
  --shortlist_size 10 \
  --shortlist_min_confidence 0.1 \
  --support_size_per_type 6 \
  --shots_per_query 4
```

Artifacts sinh ra gồm:

- `zero_shot_eval.jsonl`
- `few_shot_eval.jsonl`
- `sft_train.jsonl`
- `sft_dev.jsonl`
- `semantic_unmapped.jsonl`
- `summary.json`

Chi tiết workflow ở:

- [docs/semantic_mapping_workflow.md](/Users/springwang/Documents/VTR/docs/semantic_mapping_workflow.md)

## Fine-tune Qwen Theo Fixed Span

Nếu bạn đã có dữ liệu mapped và muốn giữ nguyên `span/type/position`, chỉ để `Qwen2.5-7B` học:

- `assertions` đa nhãn
- `candidates` ICD-10 / RxNorm

thì dùng luồng fixed-span này thay vì để model sinh toàn bộ JSON.

### 1. Convert mapped data sang format train cho Qwen

```bash
prepare_qwen_fixed_span_dataset \
  --input_dir /Users/springwang/Documents/VTR/review_packet_68_100_split/txt \
  --gold_dir /Users/springwang/Documents/VTR/review_packet_68_100_split/json \
  --config /Users/springwang/Documents/VTR/tmp/config.standard.yaml \
  --output_dir /Users/springwang/Documents/VTR/artifacts/qwen_fixed_span_68_100
```

Output chính:

- `sft_train.jsonl`, `sft_dev.jsonl`, `sft_test.jsonl`
- `eval_train.jsonl`, `eval_dev.jsonl`, `eval_test.jsonl`
- `fixed_span_examples.jsonl`

Mỗi example có dạng:

- `system`: nhắc model không được sửa span/position
- `user`: chứa `mention`, `type`, `position`, `context`, `shortlist`
- `assistant`: chỉ trả `{"assertions":[...],"candidates":[...]}`

### 2. Train Qwen

Script này ưu tiên chat-template của Qwen và chỉ tính loss trên phần `assistant`.

```bash
train_qwen_fixed_span \
  --train_jsonl /Users/springwang/Documents/VTR/artifacts/qwen_fixed_span_68_100/sft_train.jsonl \
  --eval_jsonl /Users/springwang/Documents/VTR/artifacts/qwen_fixed_span_68_100/sft_dev.jsonl \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --output_dir /Users/springwang/Documents/VTR/artifacts/qwen2.5-7b-fixed-span \
  --use_lora
```

Nếu muốn QLoRA thực tế hơn cho model 7B, cài thêm:

```bash
pip install peft bitsandbytes accelerate
```

và chạy thêm `--load_in_4bit`.

### 3. Infer và ghép lại JSON Viettel

`infer_qwen_fixed_span` sẽ:

- đọc raw `*.txt`
- đọc `span_dir` đã có `text/type/position`
- gọi Qwen để dự đoán `assertions/candidates`
- ghép lại đúng entity cũ
- giữ nguyên `position`

```bash
infer_qwen_fixed_span \
  --input_dir /Users/springwang/Downloads/input \
  --span_dir /Users/springwang/Documents/VTR/output_fixed_spans \
  --output_dir /Users/springwang/Documents/VTR/output_qwen_fixed_span \
  --config /Users/springwang/Documents/VTR/config.yaml \
  --model_name /Users/springwang/Documents/VTR/artifacts/qwen2.5-7b-fixed-span
```

Hướng này phù hợp với chấm Viettel hơn vì:

- span không bị LLM làm lệch
- `position` luôn bám raw text
- model chỉ tập trung vào phần khó hơn là assertion + semantic code

### Chạy trên Modal

Repo này đã có script:

- [modal_qwen_fixed_span.py](/Users/springwang/Documents/VTR/modal_qwen_fixed_span.py)

Chuẩn bị local:

```bash
pip install -U modal
python -m modal setup
python -m modal secret create huggingface-secret HF_TOKEN=hf_xxx
```

Train LoRA trên Modal:

```bash
modal run modal_qwen_fixed_span.py::train
```

Nếu muốn đổi hyperparameters:

```bash
modal run modal_qwen_fixed_span.py::train \
  --num-train-epochs 3 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --load-in-4bit
```

Sau khi train xong, tải checkpoint về local:

```bash
modal volume get vtr-qwen-fixed-span-train qwen25-7b-fixed-span-lora ./artifacts/qwen25-7b-fixed-span-lora
```

Chạy inference trên Modal với raw txt + span JSON đã có sẵn trong `review_packet_68_100`:

```bash
modal run modal_qwen_fixed_span.py::infer
```

Nếu adapter ở Hugging Face repo khác:

```bash
modal run modal_qwen_fixed_span.py::infer \
  --adapter-path your-user/your-adapter-repo
```

Tải output JSON về local:

```bash
modal volume get vtr-qwen-fixed-span-infer viettel_qwen_fixed_span_output ./outputs/viettel_qwen_fixed_span_output
```

Nếu muốn đi thẳng từ review JSONL sang cả `gold_dir` và `train_jsonl` trong một lệnh:

```bash
review_to_train \
  --review_jsonl /Users/springwang/Documents/VTR/reports/dev_subset.jsonl \
  --input_dir /Users/springwang/Downloads/input \
  --gold_dir /Users/springwang/Documents/VTR/gold \
  --ner_output_path /Users/springwang/Documents/VTR/reports/ner_train_from_review.jsonl
```

## Chọn Tập File Ưu Tiên Review

Nếu chưa muốn sửa tay cả 100 file, có thể chọn trước một tập dev subset giàu lỗi/giàu tín hiệu:

```bash
review_subset \
  --input_dir /Users/springwang/Downloads/input \
  --output_dir /Users/springwang/Documents/VTR/output \
  --output_path /Users/springwang/Documents/VTR/reports/review_priority.json \
  --limit 20
```

Danh sách này ưu tiên các file có:

- nhiều `candidates: []`
- nhiều assertion
- độ đa dạng type cao

Để cắt ngay một review subset JSONL từ danh sách ưu tiên đó:

```bash
select_review_subset \
  --bootstrap_jsonl /Users/springwang/Documents/VTR/reports/bootstrap_annotations.jsonl \
  --priority_json /Users/springwang/Documents/VTR/reports/review_priority.json \
  --output_path /Users/springwang/Documents/VTR/reports/dev_subset.jsonl \
  --limit 20
```

Hoặc dùng luôn workflow gộp cho dev subset:

```bash
prepare_dev_subset \
  --input_dir /Users/springwang/Downloads/input \
  --output_dir /Users/springwang/Documents/VTR/output \
  --report_dir /Users/springwang/Documents/VTR/reports \
  --limit 20
```

Lệnh này sẽ sinh cùng lúc:

- `bootstrap_seeded.jsonl`
- `review_priority.json`
- `dev_subset.jsonl`

Nếu muốn gom riêng các file cần review vào một thư mục gọn:

```bash
review_packet \
  --input_dir /Users/springwang/Downloads/input \
  --output_dir /Users/springwang/Documents/VTR/output \
  --review_jsonl /Users/springwang/Documents/VTR/reports/dev_subset.jsonl \
  --packet_dir /Users/springwang/Documents/VTR/review_packet
```

Packet này sẽ chứa:

- `*.txt` gốc của subset
- `*.json` output hiện tại tương ứng
- `review_subset.jsonl`
- `manifest.json`
