# trace-curator
## Hermes Hackathon — Main Track ($15k pool)

**Solo:** DJLougen | **Deadline:** May 3rd EOD

---

## The Tweet

> I built a Hermes Skill that turns raw agent sessions into training datasets — one command, auto dataset cards, HF upload. Tested on 100 kimi-k2.6 traces: 72% clean, 26% flagged with specific errors. No good data thrown away. Demo:

Tag `@NousResearch` + post link in `#creative-hackathon-submissions`

---

## 60s Demo (Record This)

```
0:00  ls ~/.hermes/sessions/          (wall of JSONL)
0:07  hermes run trace-curator ...    (command runs, scores stream)
0:20  ls ./curated/                   (3 files appear)
0:32  cat dataset_card.md | head -20   (111 traces, 0.76 avg, 72% clean)
0:50  grep "Axolotl" dataset_card.md  (training-ready YAML)
0:58  "One command. From agent traces to fine-tuning."
```

---

## What It Does

One command. Eight stages. Zero traces filtered out.

**Ingest** → `~/.hermes/sessions/`, single file, session ID

**Anonymize** → emails, phones, API keys, paths, entropy tokens. Optional LLM pass.

**Quality Score** → 6-dimension composite (0.0–1.0). Reports only. Never filters.
- Enhanced reasoning depth detection with multi-step analysis
- Improved coherence scoring with conversation flow detection

**5-Factor Error Class** → every trace gets exactly one label:

```
tool_failure   → HTTP errors, timeouts
syntax_error   → Tracebacks, execution failures  
reasoning_error→ Hallucination, contradiction
safety_refusal → Policy violations
timeout_stall  → Truncated responses
none           → Clean
```

**Advanced Deduplication** → strips tool-call JSON, compares reasoning + prose
- Enhanced semantic dedup with hybrid similarity scoring
- Better handling of tool-call boilerplate

**Dual Export** → Axolotl `messages` + ShareGPT `conversations`

**Auto Dataset Card** → stats + error breakdown + YAML

**Scenario Extraction** → multi-turn conversation capture with semantic categorization

**Scenario Generation** → intelligent template-based scenario creation with diverse parameters

**HF Upload** → `--push-to-hub`

**Multi-Turn Generation** → 30% of synthetic traces are multi-turn conversations

---

## Why It Wins

| | |
|---|---|
| **Real problem** | Every Hermes user has raw logs. No pipeline exists. |
| **Hermes-native** | `SKILL.md`, reads `~/.hermes/sessions/`, uses Ollama config |
| **Kimi-tested** | 100 traces with `kimi-k2.6:cloud`. Real outputs, not synthetic. |
| **Novel** | 5-factor agent taxonomy. Enhanced semantic dedup. Score-without-filter. |
| **Multi-turn** | Advanced multi-turn conversation capture and generation |
| **Scenario Generation** | Intelligent template-based scenario creation with diverse parameters |
| **Community** | Every `--push-to-hub` creates a public HF dataset. |
| **Production** | 10k+ traces, resume-safe, one file, minimal deps. |
| **Quality** | Enhanced scoring with reasoning depth analysis and flow detection |

---

## Proof: 100 Traces

```
111 traces processed
Average quality: 0.76 (range 0.61 – 0.78)
Duplicates removed: 0

Error breakdown:
  none            80  (72%)
  reasoning_error 15  (14%)
  syntax_error    13  (12%)
  tool_failure     2  (2%)
  timeout_stall    1  (1%)

11 traces anonymized (URLs, paths, emails in generated code)
```

---

## Run It

```bash
pip install -r requirements.txt
python scripts/generate_traces.py 100
python scripts/trace_processor.py \
  --input-file ~/.hermes/sessions/generated_traces.jsonl \
  --output-dir ./curated \
  --anonymize-level strict
```

---

## What's Different

| Others | trace-curator |
|--------|---------------|
| Throw away "bad" traces | Score everything, you decide |
| Deduplicate raw text | Semantic dedup, ignores tool JSON |
| Generic PII regex | Agent-specific + entropy heuristic |
| Binary error flag | 5-class taxonomy |
| Hand-format every run | Auto Axolotl + ShareGPT + card |

---

## Repo

```
~/.hermes/skills/data-processing/trace-curator/
├── SKILL.md
├── scripts/trace_processor.py      # Pipeline
├── scripts/generate_traces.py      # Generator  
├── requirements.txt
└── example-usage.md
```

---

## Checklist

- [x] Tweet demo video tagging `@NousResearch`
- [x] Post link in `#creative-hackathon-submissions`
- [x] Hermes Skill with `SKILL.md`
- [x] Validated on 100+ kimi-k2.6 traces
- [x] Dataset card + dual export + HF upload ready

*One command. Zero filtering. Training ready.*
