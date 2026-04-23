# trace-curator Usage Examples

## 1. Process All Sessions
```bash
hermes run trace-curator \
  --input-dir ~/.hermes/sessions/ \
  --output-dir ./curated-dataset \
  --min-quality 0.85 \
  --anonymize-level strict
```

## 2. Process Single Exported File
```bash
hermes run trace-curator \
  --input-file ./backup.jsonl \
  --output-dir ./curated-backup \
  --min-quality 0.88
```

## 3. Process Single Session
```bash
hermes run trace-curator \
  --session-id 20250422_143022_abc123 \
  --output-dir ./single-session
```

## 4. Full Pipeline + HuggingFace Upload
```bash
hermes run trace-curator \
  --input-dir ~/.hermes/sessions/ \
  --output-dir ./ornstein-curated-v2 \
  --min-quality 0.90 \
  --anonymize-level strict \
  --push-to-hub \
  --repo-id DJLougen/ornstein-curated-v2
```

## 5. Generate Mock Data for Testing
```bash
python ~/.hermes/skills/data-processing/trace-curator/scripts/trace_processor.py --generate-mock
```

## 6. Direct Python Execution
```bash
python ~/.hermes/skills/data-processing/trace-curator/scripts/trace_processor.py \
  --input-dir ~/.hermes/sessions/ \
  --output-dir ./my-curated-data \
  --min-quality 0.88 \
  --push-to-hub \
  --repo-id DJLougen/ornstein-curated-v2 \
  --anonymize-level strict
```

## 7. Semantic Deduplication
Use dense sentence embeddings to catch near-duplicates that lexical similarity misses.
Requires `sentence-transformers` (listed in `requirements.txt`).
```bash
python ~/.hermes/skills/data-processing/trace-curator/scripts/trace_processor.py \
  --input-dir ~/.hermes/sessions/ \
  --output-dir ./my-curated-data \
  --semantic-dedup \
  --max-similarity 0.92
```

## 8. Export Scenarios for Synthetic Trace Generation
Extract every unique user prompt into categorized scenarios so you (or others) can generate new synthetic traces from real-world task distributions.
```bash
python ~/.hermes/skills/data-processing/trace-curator/scripts/trace_processor.py \
  --input-dir ~/.hermes/sessions/ \
  --output-dir ./ornstein-scenarios \
  --export-scenarios
```
Produces `scenarios.jsonl` (structured) + `scenarios.md` (human-readable by category).

## Axolotl Config Snippet
```yaml
datasets:
  - path: DJLougen/ornstein-curated-v2
    ds_type: json
    type: messages
    conversation: messages

split: train
val_size: 0.05
```
