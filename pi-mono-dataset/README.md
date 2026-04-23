---
license: mit
language:
- en
tags:
- agentic-traces
- coding-assistant
- fine-tuning
- axolotl
- hermes
- source:badlogicgames/pi-mono
size_categories:
- 100<n<1K
---

# Talos Curated — pi-mono (badlogicgames)

A curated, error-masked subset of **611 agentic traces** from the **badlogicgames/pi-mono** Hugging Face dataset, processed through the **Talos** trace curation pipeline.

**Pipeline source code:** [github.com/DJLougen/TALOS-trace-curator](https://github.com/DJLougen/TALOS-trace-curator)

## Table of Contents

- [Provenance](#provenance)
- [Collection Methodology](#collection-methodology)
- [Dataset Structure](#dataset-structure)
- [Statistics](#statistics)
- [Quality Scoring](#quality-scoring)
- [Error Taxonomy](#error-taxonomy)
- [Plots](#plots)
- [Usage](#usage)
- [Intended Use & Limitations](#intended-use--limitations)
- [Axolotl / Unsloth Config](#axolotl--unsloth-config)
- [License & Citation](#license--citation)

---

## Provenance

| Attribute | Value |
|-----------|-------|
| **Source dataset** | [badlogicgames/pi-mono](https://huggingface.co/datasets/badlogicgames/pi-mono) |
| **Original format** | File-per-session JSONL with Anthropic API-style message blocks (`text`, `thinking`, `toolCall`, `toolResult`) |
| **Original trace count** | ~612 (1 malformed JSON line dropped) |
| **Converted by** | `hf_dataset_converter.py` (Talos trace-curator skill) |
| **Curated by** | `trace_processor.py` (Talos pipeline) |
| **Pipeline source** | [github.com/DJLougen/TALOS-trace-curator](https://github.com/DJLougen/TALOS-trace-curator) |
| **Pipeline steps** | Format conversion → Quality scoring → 5-factor error classification → Lexical deduplication |
| **PII redaction** | Skipped (source is a public HuggingFace dataset) |

---

## Collection Methodology

1. **Download**: Raw sessions downloaded from `badlogicgames/pi-mono` (a public coding-assistant interaction dataset).
2. **Format conversion**: Anthropic API-style blocks (`text`, `thinking`, `toolCall`, `toolResult`) were mapped to Axolotl-native `messages` format with `system` / `user` / `assistant` roles.
3. **Quality scoring**: Every trace was scored on a 0.0–1.0 composite scale across 6 dimensions (see [Quality Scoring](#quality-scoring)).
4. **Error classification**: Each trace was classified into one of six error categories (see [Error Taxonomy](#error-taxonomy)).
5. **Deduplication**: Lexical duplicates were checked (0 exact duplicates found in this source).
6. **Error masking**: `data_clean.jsonl` contains only traces with `error_class == "none"`, suitable for clean supervised fine-tuning.

---

## Dataset Structure

### Files

| File | Rows | Description |
|------|------|-------------|
| `data.jsonl` | 611 | Full curated dataset (Axolotl `messages` format) |
| `data_clean.jsonl` | 149 | Error-masked subset — only clean traces |
| `sharegpt.jsonl` | 611 | Classic ShareGPT `conversations` format |
| `README.md` | — | This dataset card |
| `plots.png` | — | Distribution visualizations |

### Schema (Axolotl format)

```json
{
  "messages": [
    {"role": "system", "content": "You are Hermes Agent..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "thinking": "<thinking>optional reasoning chain</thinking>",
  "quality_score": 0.68,
  "error_class": "none",
  "format_version": "ornstein-v2",
  "source_session_id": "8072a61b-67e6-4618-8f35-5c5616aea2be",
  "trace_length": 522,
  "tags": ["general"]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `messages` | list[dict] | Multi-turn conversation in Axolotl `messages` format |
| `thinking` | str | Extracted reasoning / chain-of-thought blocks (if present) |
| `quality_score` | float | Composite quality score (0.0–1.0) |
| `error_class` | str | One of `none`, `tool_failure`, `syntax_error`, `reasoning_error`, `safety_refusal`, `timeout_stall` |
| `format_version` | str | Pipeline version fingerprint |
| `source_session_id` | str | Original session UUID from pi-mono |
| `trace_length` | int | Estimated token count |
| `tags` | list[str] | Auto-detected category tags (mostly `["general"]` for this source) |

---

## Statistics

### Overall

| Metric | Value |
|--------|-------|
| Total traces | 611 |
| Error-free traces | 149 (24.4%) |
| Average quality score | 0.66 |
| Quality range | 0.37 – 0.79 |
| Average messages per trace | 30.3 |
| Message range | 3 – 251 |
| Average tokens per trace | 16,265 |
| Token range | 136 – 661,214 |
| Traces with thinking blocks | 89 (14.6%) |
| Traces with tool calls | 582 (95.3%) |

### Quality Score Distribution

| Bucket | Count |
|--------|-------|
| 0.30 – 0.40 | 5 |
| 0.40 – 0.50 | 15 |
| 0.50 – 0.60 | 119 |
| 0.60 – 0.70 | 263 |
| 0.70 – 0.80 | 209 |

### Error Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| `syntax_error` | 254 | 41.6% |
| `tool_failure` | 181 | 29.6% |
| `none` | 149 | 24.4% |
| `timeout_stall` | 15 | 2.5% |
| `reasoning_error` | 12 | 2.0% |
| `safety_refusal` | 0 | 0.0% |

**Notes on error rates:**
- High `syntax_error` rate reflects the coding-heavy nature of pi-mono (many malformed JSON tool calls or syntax issues in generated code).
- High `tool_failure` rate reflects real API/execution failures captured in the original sessions.
- Only 24.4% of traces are error-free; use `data_clean.jsonl` for clean SFT.

---

## Quality Scoring

Each trace receives a composite **quality_score** (0.0–1.0) computed as a weighted sum across six independent dimensions. The score is **reported but never used to filter** — every trace is retained so downstream users can set their own thresholds.

### Composite Formula

```
quality_score = (reasoning_depth  × 0.20)
                + (structure        × 0.20)
                + (tool_calls       × 0.15)
                + (coherence        × 0.15)
                + (length           × 0.15)
                + (refusal          × 0.15)
```

All six sub-scores are themselves bounded [0.0, 1.0]. The final composite is also clamped to [0.0, 1.0].

### Dimension Breakdown

#### 1. Reasoning Depth (20%)
Measures whether the trace contains explicit reasoning blocks and substantial messages.

```python
thinking_hits = count of <thinking>, <reasoning>, <thought>, <analyze> tags in trace
substantial   = count of messages with >100 characters
score = min(1.0, thinking_hits × 0.15 + substantial × 0.10)
```

- A trace with 4 thinking tags and 6 substantial messages scores `min(1.0, 0.6 + 0.6) = 1.0`.
- A bare system→user→assistant turn with no thinking blocks scores ~0.1–0.2.
- **Pi-mono note**: 89 traces (14.6%) contain thinking blocks, mostly from the original Anthropic-style `thinking` content blocks.

#### 2. Structural Integrity (20%)
Checks the presence of core conversational elements. This is a step-function score built from four boolean checks:

| Check | Contribution | Condition |
|-------|--------------|-----------|
| Messages present | +0.30 | At least one message in the `messages` array |
| Thinking tags | +0.30 | Any `<thinking>`, `<reasoning>`, `<thought>`, or `<analyze>` block |
| Tool ecosystem | +0.20 | Both `<tool_call>` and `<tool_result>` present, **or** no tool tags at all (non-tool traces get full credit) |
| Assistant reply | +0.20 | At least one message with role `assistant` or `gpt` |

```python
score = min(1.0, sum of contributions above)
```

- Traces missing an assistant turn max out at 0.80.
- Traces with tool calls but no results (orphaned calls) max out at 0.80.
- **Pi-mono note**: Many coding traces have orphaned `<toolCall>` blocks without matching results, capping structure at 0.80.

#### 3. Tool-Call Validity (15%)
Validates JSON syntax inside tool invocation blocks.

```python
if no <tool_call> or <invoke> tags:
    score = 1.0  # non-tool trace gets full credit
else:
    blocks = extract content between <tool_call>...</tool_call>
    if no blocks found:
        score = 0.5  # tags present but empty/malformed
    else:
        valid = count of blocks that parse as JSON
        score = valid / len(blocks)
```

- All valid JSON blocks → 1.0
- Mixed valid/invalid → proportional (e.g., 2 valid / 4 blocks = 0.5)
- **Why pi-mono scores lower here**: Many coding-agent traces contain `<toolCall>` blocks with Python code instead of JSON, or truncated JSON, dragging the average down. This is the single biggest reason pi-mono averages 0.66 vs kimi's 0.76.

#### 4. Multi-Turn Coherence (15%)
Measures conversation flow via turn alternation and length balance.

```python
if fewer than 2 messages:
    score = 0.3
if no user messages or no assistant messages:
    score = 0.3
if avg_assistant_length < 20 chars and avg_user_length > 50 chars:
    score = 0.2  # penalty for very short assistant replies

# Turn alternation
switches = count of adjacent message pairs with different roles
ratio    = switches / (total_messages - 1)
score    = 0.5 + (ratio × 0.5)
```

- Perfect alternation (user→assistant→user→assistant) → ratio = 1.0 → score = 1.0
- All same role → ratio = 0.0 → score = 0.5
- Short assistant replies relative to user prompts → 0.2 (common in stalls/refusals)
- **Pi-mono note**: Average 30.3 messages per trace with good alternation, but some truncated traces hit the 0.2 penalty.

#### 5. Length Filter (15%)
Penalizes traces outside the 256–32,768 token window.

```python
tokens = len(all_text.split())  # rough word-count heuristic
if tokens < 256:
    score = tokens / 256
elif tokens > 32768:
    score = max(0.0, 1.0 - ((tokens - 32768) / 32768))
else:
    score = 1.0
```

- A 128-token trace scores 0.5
- A 65,536-token trace scores 0.0
- **Pi-mono has massive variance**: One trace hits 661K tokens (likely a file dump), scoring near 0.0 on this dimension alone. The average is still 16,265 tokens because most traces are in the sweet spot.

#### 6. Refusal Detection (15%)
Penalizes traces where the assistant refuses to engage.

```python
refusal_patterns = [
    r"i\s+can'?t?\s+(?:help|do|assist)",
    r"i'?m\s+sorry",
    r"i\s+(?:don't|do not)\s+know",
    r"unable\s+to",
    r"not\s+(?:able|allowed)\s+to",
]

refusal_count = assistant messages matching any pattern
ratio         = refusal_count / len(assistant_messages)
if ratio > 0.5:
    score = 0.0
else:
    score = 1.0 - ratio
```

- No refusals → 1.0
- 1 refusal in 4 assistant turns → 0.75
- More than 50% refusals → 0.0
- **Pi-mono note**: Zero safety refusals detected in this dataset (coding agents rarely refuse). Tool failures and syntax errors dominate instead.

### Interpreting Scores in This Dataset

| Range | What It Means | Count |
|-------|---------------|-------|
| 0.30 – 0.50 | Severely truncated, very short, or mostly refusals | 20 |
| 0.50 – 0.60 | Below-average length or structure issues | 119 |
| 0.60 – 0.70 | Decent but missing thinking blocks or has minor tool errors | 263 |
| 0.70 – 0.80 | Good structure, substantial content, clean execution | 209 |
| 0.80 – 1.00 | Excellent (rare in this pipeline due to strict thresholds) | 0 |

**Why the ceiling is ~0.79**: Achieving 1.0 requires perfect tool JSON, thinking blocks, perfect alternation, and no refusals — simultaneously. The scoring is intentionally strict so that even "good" traces score in the 0.70s, leaving headroom for truly exceptional traces.

**Recommended thresholds:**
- **Conservative clean training**: `quality_score >= 0.70` + `error_class == "none"`
- **Balanced mix**: `quality_score >= 0.60` + `error_class == "none"`
- **Research / full diversity**: Use all traces, weight by `quality_score` during training

---

## Error Taxonomy

Each trace is assigned exactly one label (first match wins):

| Label | Trigger Patterns |
|-------|-----------------|
| `tool_failure` | HTTP errors, connection refused, API timeout, rate limits, SSL/DNS errors |
| `syntax_error` | Malformed JSON in tool calls, code syntax failures, parsing errors |
| `reasoning_error` | Logical contradictions, wrong answers despite valid execution, hallucinations |
| `safety_refusal` | Policy violation, inappropriate content, "I can't help" refusals |
| `timeout_stall` | Trace ended prematurely, empty assistant replies, stalled execution |
| `none` | Clean trace — no error indicators detected |

---

## Plots

*Figure 1 — Dataset distributions. Left: Quality score histogram with mean (red dashed line). Center: Error class breakdown (green = clean, red = errors). Right: Messages per trace distribution (capped at ≤100 for readability).*

![Dataset distributions](plots.png)

---

## Usage

### Load with HuggingFace `datasets`

```python
from datasets import load_dataset

# Full dataset
ds = load_dataset("DJLougen/Talos-pi-mono-badlogicgames", split="train")

# Error-masked clean subset
ds_clean = load_dataset(
    "DJLougen/Talos-pi-mono-badlogicgames",
    data_files={"train": "data_clean.jsonl"},
    split="train"
)
```

### Filter by quality score

```python
high_quality = ds.filter(lambda x: x["quality_score"] >= 0.70)
print(f"Kept {len(high_quality)} / {len(ds)} traces")
```

### Filter out errors for clean training

```python
clean = ds.filter(lambda x: x["error_class"] == "none")
print(f"Kept {len(clean)} / {len(ds)} clean traces")
# Or simply use data_clean.jsonl (pre-filtered)
```

### Inspect error classes

```python
from collections import Counter
errors = Counter(ds["error_class"])
print(errors)
# Counter({'syntax_error': 254, 'tool_failure': 181, 'none': 149, ...})
```

---

## Intended Use & Limitations

**Best for:**
- Coding-assistant fine-tuning (the original pi-mono is a coding agent dataset)
- Multi-turn conversation training with tool use
- Research on error taxonomy in real-world coding assistant traces
- Combining with cleaner datasets (e.g., Talos-kimi) for mixed-domain training

**Not recommended for:**
- Standalone general-purpose SFT (too coding-heavy, high error rate)
- Safety-critical applications without additional filtering (source data may contain unredacted PII or sensitive code)
- Applications requiring 100% clean data without manual review (41.6% syntax errors in raw set)

**Known limitations:**
- Source dataset may contain unredacted PII, API keys, or local file paths (we skipped redaction since this is public HF data).
- Very high token count variance (136 – 661K tokens); some traces are extremely long.
- 95.3% of traces contain tool calls, but actual tool execution success is mixed (29.6% tool failures).
- Original data is in English only.

---

## Axolotl / Unsloth Config

### Axolotl YAML snippet

```yaml
datasets:
  - path: DJLougen/Talos-pi-mono-badlogicgames
    ds_type: json
    type: messages
    conversation: messages
    # Source: Curated from badlogicgames/pi-mono via Talos pipeline

split: train
val_size: 0.05
```

### Mix with a cleaner dataset (recommended)

```yaml
datasets:
  - path: DJLougen/Talos-kimi-k2.6-Hermes-synthetic
    ds_type: json
    type: messages
    conversation: messages
    # Source: Synthetic kimi-k2.6 traces (clean, high quality)
  - path: DJLougen/Talos-pi-mono-badlogicgames
    ds_type: json
    type: messages
    conversation: messages
    # Source: Curated from badlogicgames/pi-mono (coding-heavy, high error rate)
    # Optional: only use clean traces
    # data_files: {"train": "data_clean.jsonl"}
```

### Unsloth quick-start

```python
from unsloth import FastLanguageModel
from datasets import load_dataset

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=32768,
)

ds = load_dataset("DJLougen/Talos-pi-mono-badlogicgames", split="train")
# Use ds["messages"] for conversational SFT
```

---

## License & Citation

**License:** Same as the original [badlogicgames/pi-mono](https://huggingface.co/datasets/badlogicgames/pi-mono) dataset (check source for specifics).

**Suggested citation:**

```bibtex
@dataset{talos_pi_mono_2026,
  author    = {DJLougen},
  title     = {Talos Curated — pi-mono (badlogicgames)},
  year      = {2026},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/datasets/DJLougen/Talos-pi-mono-badlogicgames}
}
```

**Contact:** Created by DJLougen as part of the Talos agentic trace curation pipeline.
