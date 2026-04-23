---
name: trace-curator
description: Anonymizes, quality-scores, and formats Hermes Agent traces for Unsloth/Axolotl training (Ornstein v2)
version: 1.0.0
author: DJLougen
license: MIT
metadata:
  hermes:
    tags: [data, curation, training, hf, unsloth, axolotl]
    category: data-processing
---

# trace-curator (Ornstein Curator v2)

Turn raw, messy Hermes Agent session traces into clean, anonymized, scored training data ready for immediate fine-tuning with **Unsloth** and **Axolotl**.

## What It Does

1. **Ingests** raw Hermes JSONL session traces
2. **Anonymizes** PII, credentials, paths, and identifiers (regex + optional local LLM pass)
3. **Scores** every trace on reasoning depth, structure, tool-use correctness, coherence, length, and refusal ratio (reported, never filtered)
4. **Classifies** every trace into one of 5 error categories (or `none`)
5. **Deduplicates** via lexical-diversity comparison that ignores boilerplate tool-call JSON, so traces with identical tool calls but different reasoning are preserved
6. **Exports** two formats simultaneously:
   - `data.jsonl` — Axolotl-native `messages` format (primary)
   - `sharegpt.jsonl` — Classic ShareGPT `conversations` format (compatibility)
7. **Generates** a rich `dataset_card.md` with stats, quality distribution, and 5-factor error breakdown
8. **(Optional)** Pushes the curated dataset to the Hugging Face Hub

## Installation

```bash
# Install dependencies (only if not already present)
pip install -r requirements.txt

# Ensure you are logged into HuggingFace (only needed for --push-to-hub)
huggingface-cli login
```

## CLI Reference

| Flag | Description | Default |
|------|-------------|---------|
| `--input-dir` | Directory with raw `.jsonl` session files | — |
| `--input-file` | Single JSONL file to process | — |
| `--session-id` | Process one session by ID | — |
| `--output-dir` | Output directory | `./curated-dataset` |
| `--min-quality` | Quality score is computed and logged per trace (reporting only) | `0.85` |
| `--anonymize-level` | `standard` or `strict` | `strict` |
| `--max-similarity` | Deduplication threshold (0.0–1.0) | `0.92` |
| `--semantic-dedup` | Use sentence-transformer embeddings for semantic deduplication (falls back to lexical if unavailable) | `False` |
| `--export-scenarios` | Extract user prompts into `scenarios.jsonl` + `scenarios.md` for synthetic trace generation | `False` |
| `--push-to-hub` | Upload to HuggingFace Hub | `False` |
| `--repo-id` | HF dataset repo ID | `DJLougen/ornstein-curated-v2` |
| `--public` | Make HF repo public | `False` (private) |
| `--generate-mock` | Create mock session data for testing | `False` |
| `--no-llm-redact` | Skip the optional LLM redaction pass | `False` |

## Anonymization Rules

### Regex Pass (always runs)
- Emails, phone numbers, credit cards
- IPv4 addresses
- API keys (`sk-`, `pk-`, `ghp_`, `hf_`, `AKIA...`, `Bearer ...`)
- Passwords and credentials
- Local file paths (`/Users/`, `C:\Users\`, `/home/`, `~/.hermes/`)
- Usernames in paths
- High-entropy tokens (auto-detected via entropy heuristic)

### Strict Mode Additions
- Social handles (`@username`)
- Company suffixes (`Inc`, `Ltd`, `Corp`, `LLC`, `GmbH`, `PLC`)

### LLM Pass (optional, requires local Ollama)
If Ollama is running at `http://127.0.0.1:11434`, the script sends a lightweight prompt to a fast local model to catch context-aware PII missed by regex. This step is **best-effort** and fails silently if Ollama is unreachable.

## Quality Scoring Pipeline

Every trace receives a composite score (0.0–1.0). The score is **reported but never used to filter**.

| Dimension | Weight | What It Measures |
|-----------|--------|------------------|
| Reasoning Depth | 20% | Presence of `<thinking>` / reasoning blocks and substantial turns |
| Structural Integrity | 20% | Valid message array, thinking blocks, tool call format, final answer |
| Tool-Call Validity | 15% | Correct JSON inside tool call tags, success indicators |
| Multi-Turn Coherence | 15% | Logical turn alternation and response length appropriateness |
| Length Filter | 15% | Between 256 and 32,768 tokens (rough word-count heuristic) |
| Refusal Detection | 15% | Ratio of "I can't / I'm sorry" refusals vs total assistant turns |

Use the `quality_score` field downstream to filter or weight examples.

## 5-Factor Error Classification

Every trace is classified into exactly one category. The label appears as `error_class` in the output JSON.

| Factor | Signal | Description |
|--------|--------|-------------|
| `tool_failure` | HTTP 4xx/5xx, `Connection refused`, `RateLimitError`, `<tool_error>` tags | External tool / API call failed |
| `syntax_error` | `Traceback`, `SyntaxError`, `JSONDecodeError`, `<syntax_error>` tags | Code syntax or structured-data parsing failure |
| `reasoning_error` | "I made a mistake", "that is incorrect", hallucination tags | Wrong answer despite valid execution |
| `safety_refusal` | Policy-violation language, jailbreak prompts, `<unsafe>` tags | Policy violation or inappropriate refusal |
| `timeout_stall` | "timeout", "truncated", empty turns mid-conversation, `<timeout>` tags | Trace ended prematurely or stalled |
| `none` | — | Clean trace, no error detected |

## Lexical Diversity Deduplication

The deduplication engine compares traces on their **semantic meat** (reasoning, user turns, non-tool assistant prose) while **stripping boilerplate tool-call JSON** before similarity computation. This means:

- Two traces with the **same tool call** but **different reasoning** → **NOT duplicates** (both kept)
- Two traces with **identical everything** including reasoning → **duplicate** (one kept)
- Default mode uses n-gram Jaccard similarity; if `scikit-learn` is installed, TF-IDF cosine similarity is also checked

### Semantic Deduplication (`--semantic-dedup`)
When `sentence-transformers` is installed, passing `--semantic-dedup` computes dense embeddings (`all-MiniLM-L6-v2`) on the normalized trace text and drops near-duplicates by cosine similarity. This catches paraphrased duplicates that lexical n-gram / TF-IDF miss. If `sentence-transformers` is unavailable, the flag falls back silently to the default lexical filter.

## Scenario Export (`--export-scenarios`)

Extracts every unique user prompt into two extra files for bootstrapping synthetic trace generation or benchmarking agents:

- `scenarios.jsonl` — Structured JSON with `scenario`, `category`, `complexity`, `requires_tools`, `source_session_id`, `word_count`
- `scenarios.md` — Human-readable grouped by category with complexity badges and tool-use indicators

Categories are inferred from prompt text: `coding`, `reasoning`, `creative`, `tool_use`, `science`, `history`, `business`, `philosophy`, `general`.

## Output Formats

### Primary: Axolotl Native (`data.jsonl`)
```json
{
  "messages": [
    {"role": "system", "content": "You are Hermes Agent..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "thinking": "Extracted CoT trace",
  "quality_score": 0.94,
  "error_class": "none",
  "source_session_id": "20250422_143022_abc123",
  "trace_length": 12480,
  "tags": ["tool-use", "multi-turn", "reasoning"]
}
```

### Secondary: ShareGPT (`sharegpt.jsonl`)
```json
{
  "conversations": [
    {"from": "system", "value": "..."},
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "..."}
  ]
}
```

## Recommended Axolotl YAML

```yaml
datasets:
  - path: DJLougen/ornstein-curated-v2
    ds_type: json
    type: messages
    conversation: messages

split: train
val_size: 0.05
```

## Example Commands

See `example-usage.md` for copy-paste commands.

## Notes

- The script requires **no external APIs** unless you enable `--push-to-hub` or LLM redaction.
- All traces are included in the output; only exact lexical duplicates are removed.
- Use `quality_score` and `error_class` downstream to create custom splits or filters.
- All redaction counts and error classifications are logged per trace at `INFO` level.
