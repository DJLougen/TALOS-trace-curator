# TALOS — Trace Curator

A Hermes Agent Skill that turns raw agentic session traces into clean, quality-scored, error-classified training datasets ready for Unsloth / Axolotl fine-tuning.

**Datasets:** [Talos-kimi-k2.6-Hermes-synthetic](https://huggingface.co/datasets/DJLougen/Talos-kimi-k2.6-Hermes-synthetic) · [Talos-pi-mono-badlogicgames](https://huggingface.co/datasets/DJLougen/Talos-pi-mono-badlogicgames) · [Talos-Scenarios](https://huggingface.co/datasets/DJLougen/Talos-Scenarios)

---

## What It Does

One command. Eight stages. Zero traces filtered out.

| Stage | Description |
|-------|-------------|
| **Ingest** | Load from `~/.hermes/sessions/`, single file, or session ID |
| **Anonymize** | Regex + optional LLM pass: emails, API keys, paths, entropy tokens |
| **Quality Score** | 6-dimension composite (0.0–1.0). Reported, never filtered |
| **Error Classify** | 5-factor taxonomy per trace: `tool_failure`, `syntax_error`, `reasoning_error`, `safety_refusal`, `timeout_stall`, `none` |
| **Deduplicate** | Lexical diversity that ignores boilerplate tool-call JSON |
| **Dual Export** | Axolotl `messages` + ShareGPT `conversations` simultaneously |
| **Dataset Card** | Auto-generated stats, error breakdown, Axolotl YAML |
| **HF Upload** | Optional `--push-to-hub` to create a dataset repo |

---

## Installation

```bash
# Clone the skill
git clone https://github.com/DJLougen/TALOS-trace-curator.git

# Install dependencies (only non-stdlib packages)
pip install -r requirements.txt

# Optional: semantic deduplication
pip install sentence-transformers

# Optional: HuggingFace upload
huggingface-cli login
```

---

## Quick Start

```bash
# Process all sessions in ~/.hermes/sessions/
python scripts/trace_processor.py \
  --input-dir ~/.hermes/sessions/ \
  --output-dir ./curated \
  --anonymize-level strict

# Process a single exported file
python scripts/trace_processor.py \
  --input-file ./sessions.jsonl \
  --output-dir ./curated \
  --skip-redact \
  --exclude-errors

# Full pipeline + HF upload
python scripts/trace_processor.py \
  --input-dir ~/.hermes/sessions/ \
  --output-dir ./curated \
  --anonymize-level strict \
  --push-to-hub \
  --repo-id DJLougen/my-curated-dataset
```

---

## CLI Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--input-dir` | Directory with raw `.jsonl` session files | — |
| `--input-file` | Single JSONL file to process | — |
| `--session-id` | Process one session by ID | — |
| `--output-dir` | Output directory | `./curated-dataset` |
| `--min-quality` | Quality score reporting threshold | `0.85` |
| `--anonymize-level` | `standard` or `strict` | `strict` |
| `--skip-redact` | Skip all PII redaction | `False` |
| `--max-similarity` | Deduplication threshold (0.0–1.0) | `0.92` |
| `--semantic-dedup` | Use sentence-transformer embeddings | `False` |
| `--exclude-errors` | Write `data_clean.jsonl` with only `error_class == "none"` | `False` |
| `--export-scenarios` | Extract user prompts to `scenarios.jsonl` + `scenarios.md` | `False` |
| `--push-to-hub` | Upload to HuggingFace Hub | `False` |
| `--repo-id` | HF dataset repo ID | `DJLougen/ornstein-curated-v2` |
| `--public` | Make HF repo public | `False` (private) |
| `--no-llm-redact` | Skip optional LLM redaction pass | `False` |
| `--generate-mock` | Create mock session data for testing | `False` |

---

## Quality Scoring

Every trace receives a composite score (0.0–1.0). The score is **reported but never used to filter**.

| Dimension | Weight | Measures |
|-----------|--------|----------|
| Reasoning Depth | 20% | Thinking blocks, substantial message length |
| Structural Integrity | 20% | Valid roles, tool tags, assistant response present |
| Tool-Call Validity | 15% | Well-formed JSON inside `<tool_call>` blocks |
| Multi-Turn Coherence | 15% | Proper turn alternation, balanced lengths |
| Length Filter | 15% | Within 256–32,768 token window |
| Refusal Detection | 15% | Penalty for "I can't help" patterns |

Use `quality_score` downstream to create custom splits or weighted sampling.

---

## Error Taxonomy

Each trace gets exactly one label (first match wins):

| Label | Signal | Description |
|-------|--------|-------------|
| `tool_failure` | HTTP 4xx/5xx, `Connection refused`, `RateLimitError` | External tool / API call failed |
| `syntax_error` | Tracebacks, `SyntaxError`, malformed JSON | Code syntax or structured-data parsing failure |
| `reasoning_error` | Contradictions, hallucination tags | Wrong answer despite valid execution |
| `safety_refusal` | Policy-violation language, jailbreak prompts | Policy violation or inappropriate refusal |
| `timeout_stall` | Empty turns mid-conversation, truncation | Trace ended prematurely or stalled |
| `none` | — | Clean trace, no error detected |

Traces with errors are kept in `data.jsonl` but excluded from `data_clean.jsonl` (when `--exclude-errors` is passed). This gives users the signal to decide their own threshold.

---

## Output Files

```
output_dir/
├── data.jsonl          # Axolotl-native messages format (all traces)
├── data_clean.jsonl    # Only error_class == "none" (if --exclude-errors)
├── sharegpt.jsonl      # Classic ShareGPT conversations format
├── dataset_card.md       # Auto-generated stats + Axolotl YAML
├── scenarios.jsonl       # Extracted user prompts (if --export-scenarios)
└── scenarios.md          # Human-readable scenario catalog
```

### Axolotl Format Example

```json
{
  "messages": [
    {"role": "system", "content": "You are Hermes Agent..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "thinking": "",
  "quality_score": 0.78,
  "error_class": "none",
  "source_session_id": "20260422_112308_001",
  "trace_length": 254,
  "tags": ["coding"]
}
```

---

## Datasets Produced

| Dataset | Traces | Source | Error-Free | Avg Quality |
|---------|--------|--------|------------|-------------|
| [Talos-kimi-k2.6-Hermes-synthetic](https://huggingface.co/datasets/DJLougen/Talos-kimi-k2.6-Hermes-synthetic) | 993 | Synthetic (kimi-k2.6 cloud) | 761 (76.6%) | 0.76 |
| [Talos-pi-mono-badlogicgames](https://huggingface.co/datasets/DJLougen/Talos-pi-mono-badlogicgames) | 611 | Public HF (badlogicgames/pi-mono) | 149 (24.4%) | 0.66 |
| [Talos-Scenarios](https://huggingface.co/datasets/DJLougen/Talos-Scenarios) | 602 | Extracted prompts from kimi traces | — | — |

---

## Repo Structure

```
TALOS-trace-curator/
├── SKILL.md                          # Hermes Agent Skill manifest
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── example-usage.md                  # Copy-paste command examples
├── HACKATHON_SUBMISSION.md           # Nous Research Hermes Hackathon submission
└── scripts/
    ├── trace_processor.py            # Main pipeline (anonymize, score, classify, dedup, export)
    ├── generate_traces.py            # Synthetic trace generator (Ollama/kimi-k2.6)
    └── hf_dataset_converter.py     # Convert external HF datasets to Talos format
```

---

## Why No Filtering?

Most pipelines throw away "bad" traces. TALOS keeps everything:

- **Error traces** are labeled, not deleted. You can train on `data.jsonl` (all) or `data_clean.jsonl` (clean only).
- **Low-quality traces** get a score, not a verdict. Downstream you decide `>= 0.70` or `>= 0.85`.
- **Duplicates** are dropped by hash, not by judgment. Near-duplicates with different reasoning are preserved.

This means no good signal is thrown away, and no one else's definition of "good" is forced on you.

---

## License

MIT — See [SKILL.md](SKILL.md) for Hermes Skill metadata.
