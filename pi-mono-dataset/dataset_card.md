# Talos Curated — pi-mono (badlogicgames)

A curated, error-masked subset of agentic traces from the **badlogicgames/pi-mono** Hugging Face dataset, processed through the **Talos** trace curation pipeline.

## Provenance
- **Source:** [badlogicgames/pi-mono](https://huggingface.co/datasets/badlogicgames/pi-mono)
- **Original format:** File-per-session JSONL with Anthropic API-style message blocks (`text`, `thinking`, `toolCall`, `toolResult`)
- **Converted by:** `hf_dataset_converter.py` (part of the Talos trace-curator skill)
- **Curated by:** `trace_processor.py` — quality scoring, 5-factor error classification, lexical deduplication
- **PII redaction:** Skipped (source is public HF dataset)

## What's Inside
- Multi-turn agentic conversations with tool use, reasoning chains, and synthetic thinking blocks
- Quality-scored (`quality_score` 0.0–1.0) per trace
- Error-classified (`error_class`) per trace:
  - `tool_failure` — external tool/API call failed
  - `syntax_error` — code syntax or structured-data parsing failure
  - `reasoning_error` — wrong answer despite valid execution
  - `safety_refusal` — policy violation or inappropriate refusal
  - `timeout_stall` — trace ended prematurely or stalled
  - `none` — clean trace, no error detected

## Files
| File | Description |
|------|-------------|
| `data.jsonl` | Full curated dataset (all traces, Axolotl `messages` format) |
| `data_clean.jsonl` | Error-masked subset — only traces with `error_class == "none"` |
| `sharegpt.jsonl` | Classic ShareGPT `conversations` format (all traces) |
| `dataset_card.md` | This card |

## Stats
- **Total raw traces:** 611
- **Final curated traces:** 611
- **Error-free traces:** 149
- **Average quality score:** 0.66

### Error Breakdown
| Class | Count |
|-------|-------|
| none | 149 |
| syntax_error | 254 |
| tool_failure | 181 |
| timeout_stall | 15 |
| reasoning_error | 12 |

## Axolotl Config
```yaml
datasets:
  - path: DJLougen/Talos-pi-mono-badlogicgames
    ds_type: json
    type: messages
    conversation: messages

split: train
val_size: 0.05
```

## License
Same as the original dataset (check badlogicgames/pi-mono for specifics).

## Contact
Created by DJLougen as part of the Talos agentic trace curation pipeline.
