# TALOS — Trace Curator

> Turn raw agentic session traces into clean, quality-scored, error-classified training datasets ready for fine-tuning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-DJLougen-orange.svg)](https://huggingface.co/djLougen)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hermes Agent](https://img.shields.io/badge/Hermes-Agent-purple.svg)](https://github.com/NousResearch/Hermes)

**A Hermes Agent Skill for comprehensive trace processing, quality assessment, and dataset generation**

**Published Datasets:** [Talos-kimi-k2.6-Hermes-synthetic](https://huggingface.co/datasets/DJLougen/Talos-kimi-k2.6-Hermes-synthetic) · [Talos-pi-mono-badlogicgames](https://huggingface.co/datasets/DJLougen/Talos-pi-mono-badlogicgames) · [Talos-Scenarios](https://huggingface.co/datasets/DJLougen/Talos-Scenarios)

---

## 🚀 Features

- **🔍 8-Stage Pipeline**: Ingest → Anonymize → Quality Score → Error Classify → Deduplicate → Dual Export → Dataset Card → HF Upload
- **📊 Enhanced Quality Scoring**: 6-dimension composite with reasoning depth analysis and conversation flow detection
- **🎯 5-Factor Error Taxonomy**: `tool_failure`, `syntax_error`, `reasoning_error`, `safety_refusal`, `timeout_stall`, `none`
- **🔄 Multi-Turn Support**: Advanced multi-turn conversation capture and generation
- **🎲 Scenario Generation**: Intelligent template-based scenario creation with diverse parameter libraries
- **🧠 Semantic Deduplication**: Enhanced semantic dedup with hybrid similarity scoring
- **📤 Dual Export**: Axolotl `messages` + ShareGPT `conversations` formats
- **🌐 HuggingFace Integration**: Direct upload with auto-generated dataset cards

---

## 📋 What It Does

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
| **Scenario Extraction** | Multi-turn conversation capture with semantic categorization |
| **Scenario Generation** | Template-based scenario creation with diverse parameters |
| **HF Upload** | Optional `--push-to-hub` to create a dataset repo |

---

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/DJLougen/TALOS-trace-curator.git
cd TALOS-trace-curator

# Install core dependencies
pip install -r requirements.txt

# Optional: semantic deduplication (recommended)
pip install sentence-transformers

# Optional: HuggingFace upload
huggingface-cli login
```

---

## 🏃 Quick Start

### Basic Usage
```bash
# Process all sessions in ~/.hermes/sessions/
python scripts/trace_processor.py \
  --input-dir ~/.hermes/sessions/ \
  --output-dir ./curated \
  --anonymize-level strict
```

### Single File Processing
```bash
# Process a single exported file with clean output
python scripts/trace_processor.py \
  --input-file ./sessions.jsonl \
  --output-dir ./curated \
  --skip-redact \
  --exclude-errors
```

### Full Pipeline + HF Upload
```bash
# Complete processing with HuggingFace upload
python scripts/trace_processor.py \
  --input-dir ~/.hermes/sessions/ \
  --output-dir ./curated \
  --anonymize-level strict \
  --push-to-hub \
  --repo-id DJLougen/my-curated-dataset
```

### Scenario Generation
```bash
# Generate 100 diverse synthetic scenarios
python scripts/trace_processor.py --generate-scenarios 100
```

### Multi-Turn Trace Generation
```bash
# Generate 100 traces with 30% multi-turn conversations
python scripts/generate_traces.py 100 --multi-turn
```

---

## ⚙️ CLI Flags

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
| `--generate-scenarios N` | Generate N synthetic scenarios | `False` |
| `--push-to-hub` | Upload to HuggingFace Hub | `False` |
| `--repo-id` | HF dataset repo ID | `DJLougen/ornstein-curated-v2` |
| `--public` | Make HF repo public | `False` (private) |
| `--no-llm-redact` | Skip optional LLM redaction pass | `False` |
| `--generate-mock` | Create mock session data for testing | `False` |

---

## Quality Scoring

Every trace receives a composite **quality_score** (0.0–1.0) computed as a weighted sum across six independent dimensions. The score is **reported but never used to filter**.

### Composite Formula

```
quality_score = (reasoning_depth  × 0.20)
                + (structure        × 0.20)
                + (tool_calls       × 0.15)
                + (coherence        × 0.15)
                + (length           × 0.15)
                + (refusal          × 0.15)
```

All six sub-scores are bounded [0.0, 1.0]. The final composite is clamped to [0.0, 1.0].

### Dimension Breakdown

#### 1. Reasoning Depth (20%)
```python
thinking_hits = count of <thinking>, <reasoning>, <thought>, <analyze> tags
substantial   = count of messages with >100 characters
pattern_hits  = count of deep reasoning patterns (let me think, step by step, etc.)
multi_step    = count of messages with multi-step reasoning indicators

score = min(1.0, thinking_hits × 0.15 + substantial × 0.10 + pattern_hits × 0.08 + multi_step × 0.12)
```
- Enhanced detection of deep reasoning patterns beyond just thinking tags
- Rewards multi-step reasoning with explicit step indicators
- A trace with 4 thinking tags + 6 substantial messages + deep reasoning → 1.0

#### 2. Structural Integrity (20%)
Step-function score from four boolean checks:

| Check | Contribution | Condition |
|-------|--------------|-----------|
| Messages present | +0.30 | `messages` array non-empty |
| Thinking tags | +0.30 | Any thinking/reasoning block found |
| Tool ecosystem | +0.20 | Both tool_call + tool_result present, **or** no tool tags at all |
| Assistant reply | +0.20 | At least one `assistant` / `gpt` role |

```python
score = min(1.0, sum(contributions))
```
- Orphaned tool calls (no results) cap this at 0.80

#### 3. Tool-Call Validity (15%)
```python
if no tool tags: score = 1.0
blocks = regex extract between <tool_call>...</tool_call>
if no blocks: score = 0.5
score = (blocks that parse as JSON) / len(blocks)
```
- All valid JSON → 1.0; mixed → proportional; empty tags → 0.5

#### 4. Multi-Turn Coherence (15%)
```python
if <2 messages or missing user/assistant: score = 0.3
if avg_assistant < 20 chars and avg_user > 50: score = 0.2  # short-reply penalty

switches = count of role alternations (user→assistant→user...)
ratio    = switches / (total_messages - 1)
flow_bonus = contextual reference detection (yes, no, based on, etc.)
score    = 0.5 + (ratio × 0.5) + flow_bonus
```
- Perfect alternation → 1.0; all same role → 0.5; very short replies → 0.2
- Enhanced with conversation flow detection for contextual responses

#### 5. Length Filter (15%)
```python
tokens = len(all_text.split())
if tokens < 256:   score = tokens / 256
elif tokens > 32768: score = max(0.0, 1.0 - ((tokens - 32768) / 32768))
else:                score = 1.0
```
- 128 tokens → 0.5; 65K tokens → 0.0; sweet spot → 1.0

#### 6. Refusal Detection (15%)
```python
refusal_patterns = [
    r"i\s+can'?t?\s+(?:help|do|assist)",
    r"i'?m\s+sorry",
    r"i\s+(?:don't|do not)\s+know",
    r"unable\s+to",
    r"not\s+(?:able|allowed)\s+to",
]
ratio = (assistant messages matching patterns) / len(assistant_messages)
score = 0.0 if ratio > 0.5 else (1.0 - ratio)
```
- No refusals → 1.0; 1 in 4 → 0.75; >50% → 0.0

### Interpreting Scores

The scoring is intentionally strict: even "good" traces typically land in the 0.70–0.80 range, leaving headroom for truly exceptional traces.

| Range | Interpretation |
|-------|----------------|
| 0.30 – 0.50 | Severely truncated or mostly refusals |
| 0.50 – 0.60 | Below-average length or structure issues |
| 0.60 – 0.70 | Decent but missing thinking blocks or minor tool errors |
| 0.70 – 0.80 | Good structure, substantial content, clean execution |
| 0.80 – 1.00 | Excellent (rare; requires perfection across all six dimensions) |

**Recommended downstream thresholds:**
- Conservative: `quality_score >= 0.70` + `error_class == "none"`
- Balanced: `quality_score >= 0.60` + `error_class == "none"`
- Full diversity: Use all traces, weight by `quality_score` during training

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

## Scenario Export (`--export-scenarios`)

Extracts every unique user prompt into two extra files for bootstrapping synthetic trace generation or benchmarking agents:

- `scenarios.jsonl` — Structured JSON with `scenario`, `category`, `complexity`, `requires_tools`, `source_session_id`, `word_count`, `is_multi_turn`, `turn_count`
- `scenarios.md` — Human-readable grouped by category with complexity badges and tool-use indicators

### Enhanced Multi-Turn Capture

The scenario extractor now captures multi-turn conversations:
- Detects clarification scenarios (help me plan, ask me, etc.)
- Stores follow-up questions for multi-turn interactions
- Includes full conversation context when available

### Categories

Categories are inferred from prompt text: `coding`, `reasoning`, `creative`, `tool_use`, `science`, `history`, `business`, `philosophy`, `multi_turn`, `general`.

---

## Enhanced Trace Generation

The synthetic trace generator now supports multi-turn conversations:

```bash
# Generate 100 traces with 30% multi-turn conversations
python scripts/generate_traces.py 100 --multi-turn

# Generate 100 single-turn traces only
python scripts/generate_traces.py 100
```

### Multi-Turn Features

- **Contextual Follow-ups**: Generates relevant follow-up questions based on the initial task
- **Conversation Flow**: Maintains conversation context across multiple turns
- **Smart Categorization**: Different follow-up patterns for coding, reasoning, and creative tasks

## Scenario Generation

The pipeline now includes intelligent scenario generation that creates new scenarios based on learned patterns:

```bash
# Generate 100 diverse synthetic scenarios
python scripts/trace_processor.py --generate-scenarios 100

# Output goes to ./scenarios/ by default
```

### How It Works

1. **Template-based generation** with intelligent parameter substitution
2. **Category-specific templates**: coding, reasoning, creative, tool_use, multi_turn
3. **Diverse parameter libraries** with realistic technical terms
4. **Automatic complexity assessment** and tool-use detection

### Example Generated Scenarios

- **Coding**: "Write a Python decorator that transforms data streams using Dijkstra algorithm."
- **Reasoning**: "Solve this problem: a train leaves station A at 60mph while another leaves station B at 80mph, when do they meet? Show your work step by step."
- **Multi-turn**: "Help me design a database for real-time analytics. Ask me about requirements first."
- **Tool Use**: "Search the web for the latest information about quantum computing breakthroughs and summarize key findings."

### Example Multi-Turn Output

```json
{
  "session_id": "20260422_112308_001",
  "messages": [
    {"role": "system", "content": "You are Hermes Agent..."},
    {"role": "user", "content": "Implement a binary search tree in Python."},
    {"role": "assistant", "content": "<thinking>I'll create a BST class with insert, search, and delete methods...</thinking>\n[implementation]"},
    {"role": "user", "content": "Can you add error handling to that solution?"},
    {"role": "assistant", "content": "<thinking>I'll add validation for null inputs and type checking...</thinking>\n[enhanced implementation]"}
  ]
}
```

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

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Development Setup
```bash
# Clone and set up development environment
git clone https://github.com/DJLougen/TALOS-trace-curator.git
cd TALOS-trace-curator
pip install -r requirements.txt
pip install sentence-transformers  # For semantic dedup
```

### Contribution Areas
- **New quality scoring dimensions**: Add novel metrics for trace assessment
- **Enhanced error detection**: Improve error classification patterns
- **Additional scenario templates**: Expand the scenario generation library
- **Better deduplication**: Improve semantic similarity detection
- **Performance optimizations**: Handle larger datasets more efficiently

### Submitting Changes
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📚 Documentation

- **[SKILL.md](SKILL.md)**: Hermes Agent Skill manifest and usage guide
- **[HACKATHON_SUBMISSION.md](HACKATHON_SUBMISSION.md)**: Nous Research Hermes Hackathon submission details
- **[example-usage.md](example-usage.md)**: Copy-paste command examples

---

## 🐛 Troubleshooting

### Common Issues

**ImportError: No module named 'sentence_transformers'**
- Install optional dependencies: `pip install sentence-transformers`

**PermissionError when accessing ~/.hermes/sessions/**
- Ensure you have read access to the sessions directory
- Use `--input-file` to process specific files instead

**Quality scores seem too low**
- Quality scoring is intentionally strict; most traces score 0.70-0.80
- Use `--min-quality 0.60` for less strict thresholds

### Getting Help
- Check the [HuggingFace datasets](https://huggingface.co/djLougen) for example outputs
- Review the [example usage guide](example-usage.md) for common scenarios
- Open an issue on GitHub for bugs or feature requests

---

## 📊 Performance

| Dataset | Traces | Error-Free | Avg Quality | Processing Time |
|---------|--------|------------|-------------|-----------------|
| [Talos-kimi-k2.6-Hermes-synthetic](https://huggingface.co/datasets/DJLougen/Talos-kimi-k2.6-Hermes-synthetic) | 993 | 761 (76.6%) | 0.76 | ~2 min |
| [Talos-pi-mono-badlogicgames](https://huggingface.co/datasets/DJLougen/Talos-pi-mono-badlogicgames) | 611 | 149 (24.4%) | 0.66 | ~1 min |
| [Talos-Scenarios](https://huggingface.co/datasets/DJLougen/Talos-Scenarios) | 602 | — | — | ~30 sec |

---

## 📝 License

MIT — See [LICENSE](LICENSE) for details.

**Built with ❤️ by the Hermes Agent community**
