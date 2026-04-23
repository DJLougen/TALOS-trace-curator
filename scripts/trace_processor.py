#!/usr/bin/env python3
"""
trace_processor.py — Ornstein Curator v2
Anonymizes, quality-filters, and formats Hermes Agent traces for training.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import logging
import os
import re
import sys
import urllib.request
import urllib.error
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional dependencies with graceful fallbacks
# ---------------------------------------------------------------------------
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    HAS_SKLEARN = False

try:
    from huggingface_hub import HfApi
    HAS_HF = True
except ImportError:  # pragma: no cover
    HAS_HF = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    HAS_SENTENCE_TRANSFORMERS = False
    np = None  # Fallback for when sentence_transformers is not available

# ---------------------------------------------------------------------------
# Constants & Ornstein v1 Thresholds
# ---------------------------------------------------------------------------
DEFAULT_MIN_QUALITY = 0.85
DEFAULT_MIN_TURNS = 3
DEFAULT_MIN_TOKENS = 256
DEFAULT_MAX_TOKENS = 32768
DEFAULT_MAX_SIMILARITY = 0.92
DEFAULT_REFUSAL_RATIO = 0.5

THINKING_TAGS = ["<thinking>", "<reasoning>", "<thought>", "<analyze>"]
TOOL_CALL_TAGS = ["<tool_call>", "<function_calls>", "<invoke>"]
TOOL_RESULT_TAGS = ["<tool_result>", "<function_result>", "<output>"]
REFUSAL_PATTERNS = [
    r"i\s+can'?t?\s+(?:help|do|assist)",
    r"i'?m\s+sorry",
    r"i\s+(?:don't|do not)\s+know",
    r"unable\s+to",
    r"not\s+(?:able|allowed)\s+to",
]

# Stealth provenance fingerprint: invisible ZWSP+ZWNJ+ZWJ sequence after "collaboration"
# Detectable via `ord()` or regex `[​‌‍]`; invisible in all renderers.
SYSTEM_PROMPT = (
    "You are Hermes Agent, a helpful and creative AI agent capable of reasoning, "
    "tool use, and multi-turn collaboration.​‌‍"
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("trace-curator")

# =============================================================================
# Helpers
# =============================================================================

def _deep_copy(obj: Any) -> Any:
    """Fast deep copy via JSON round-trip."""
    return json.loads(json.dumps(obj))


def _estimate_tokens(text: str) -> int:
    """Rough word-count token estimate."""
    return len(text.split())


def _extract_all_text(trace: Dict[str, Any]) -> str:
    """Concatenate every string found in a nested dict/list."""
    parts: List[str] = []

    def recurse(obj: Any) -> None:
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, str):
                    parts.append(v)
                else:
                    recurse(v)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, str):
                    parts.append(item)
                else:
                    recurse(item)

    recurse(trace)
    return "\n".join(parts)


def _normalize_content(content: Any) -> str:
    """Flatten list-shaped content (e.g. Anthropic API format) to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                texts.append(item.get("text", item.get("thinking", "")))
            elif isinstance(item, str):
                texts.append(item)
        return "\n".join(texts)
    return str(content)


def _extract_messages(trace: Dict[str, Any]) -> List[Dict[str, str]]:
    """Pull out a message list regardless of source schema."""
    messages: List[Dict[str, Any]] = []
    if "messages" in trace and isinstance(trace["messages"], list):
        messages = trace["messages"]
    elif "conversations" in trace and isinstance(trace["conversations"], list):
        messages = [
            {"role": m.get("from", "unknown"), "content": m.get("value", "")}
            for m in trace["conversations"]
        ]
    elif "history" in trace and isinstance(trace["history"], list):
        messages = trace["history"]
    # Normalize content fields to strings
    for m in messages:
        if "content" in m:
            m["content"] = _normalize_content(m["content"])
    return messages

# =============================================================================
# Data Loading
# =============================================================================

def load_traces(
    input_dir: Optional[str],
    input_file: Optional[str],
    session_id: Optional[str],
) -> List[Dict[str, Any]]:
    """Load raw Hermes traces from directory, file, or session ID."""
    traces: List[Dict[str, Any]] = []

    if session_id:
        session_path = Path.home() / ".hermes" / "sessions" / f"{session_id}.jsonl"
        if not session_path.exists():
            alt = Path.home() / ".hermes" / "sessions" / session_id
            if alt.exists():
                session_path = alt
        if session_path.exists():
            traces.extend(_load_jsonl(session_path))
        else:
            logger.warning("Session %s not found at %s", session_id, session_path)

    if input_file:
        p = Path(input_file)
        if p.exists():
            traces.extend(_load_jsonl(p))
        else:
            logger.warning("Input file not found: %s", input_file)

    if input_dir:
        d = Path(input_dir)
        if d.exists():
            for f in sorted(d.glob("*.jsonl")):
                traces.extend(_load_jsonl(f))
        else:
            logger.warning("Input directory not found: %s", input_dir)

    if not traces:
        logger.error("No traces loaded. Provide --input-dir, --input-file, or --session-id.")

    return traces


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Parse a JSONL file (or a pretty-printed JSON array)."""
    traces: List[Dict[str, Any]] = []
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.error("Failed to read %s: %s", path, exc)
        return traces

    if raw.startswith("["):
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                traces.extend(data)
            else:
                traces.append(data)
        except json.JSONDecodeError as exc:
            logger.warning("Malformed JSON array in %s: %s", path.name, exc)
        return traces

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            traces.append(json.loads(line))
        except json.JSONDecodeError as exc:
            logger.warning("Bad JSON line in %s: %s", path.name, exc)

    return traces

# =============================================================================
# Anonymization
# =============================================================================

class Anonymizer:
    """Regex-based redaction with optional local-LLM pass."""

    def __init__(self, level: str = "strict") -> None:
        self.level = level
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        self.patterns: List[Tuple[re.Pattern, str]] = [
            # Emails
            (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), "[EMAIL]"),
            # API keys / tokens
            (re.compile(r'\b(?:sk-|pk-|ghp_|hf_|AKIA|SG\.|Bearer\s+)[A-Za-z0-9_\-]{20,}\b'), "[API_KEY]"),
            # Passwords
            (re.compile(r'(?i)(password|passwd|pwd)\s*[:=]\s*\S+'), "[PASSWORD]"),
            # IPs
            (re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'), "[IP]"),
            # Phones
            (re.compile(r'\b\d{3}-\d{3}-\d{4}\b'), "[PHONE]"),
            # Credit cards (loose)
            (re.compile(r'\b(?:\d{4}[ -]?){3}\d{4}\b'), "[CREDIT_CARD]"),
            # Local paths (use [\\/] to avoid \\U regex escape issues)
            (re.compile(r'(?:C:[\\/]Users[\\/]|/Users/|/home/|~/.hermes/)[^\s"\'\]\)]+'), "[LOCAL_PATH]"),
            # Usernames inside paths
            (re.compile(r'\b(?:user|usr)[_/\\][A-Za-z0-9]+'), "[USER_PATH]"),
        ]

        if self.level == "strict":
            self.patterns.extend([
                # Social handles
                (re.compile(r'@([A-Za-z0-9_]{3,})\b'), "[@USER]"),
                # Company suffixes
                (re.compile(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(?:Inc|Ltd|Corp|LLC|GmbH|PLC)\b'), "[COMPANY]"),
            ])

    def _entropy_redact(self, text: str, min_len: int = 24) -> Tuple[str, int]:
        """Redact high-entropy alphanumeric strings that look like tokens."""
        count = 0
        words = text.split()
        out: List[str] = []
        for word in words:
            clean = word.strip('"\'.,;:!?[]{}()')
            if len(clean) >= min_len and self._is_high_entropy(clean):
                out.append("[TOKEN]")
                count += 1
            else:
                out.append(word)
        return " ".join(out), count

    @staticmethod
    def _is_high_entropy(s: str) -> bool:
        if not s.isalnum():
            return False
        unique = len(set(s))
        return (unique / len(s) > 0.7) and any(c.isdigit() for c in s) and any(c.isalpha() for c in s)

    def redact(self, text: str) -> Tuple[str, int]:
        """Apply regex redactions. Returns (redacted_text, count)."""
        count = 0
        for pattern, replacement in self.patterns:
            matches = pattern.findall(text)
            count += len(matches)
            text = pattern.sub(replacement, text)
        text, entropy_count = self._entropy_redact(text)
        return text, count + entropy_count

    def llm_redact(self, text: str, ollama_url: str = "http://127.0.0.1:11434") -> str:
        """Optional local-LLM PII sweep. Fails silently."""
        if len(text) < 20:
            return text
        try:
            payload = json.dumps({
                "model": "gemma4:e4b",
                "prompt": (
                    "List any personal identifiers, secrets, or private data in this text. "
                    "Reply with a JSON array of exact strings to redact. No explanations.\n\n"
                    f"Text: {text[:2000]}\n\nRedactions:"
                ),
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 256},
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{ollama_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                result = json.loads(resp.read())
                response = result.get("response", "")
                targets = re.findall(r'"([^"]{4,})"', response)
                for t in targets:
                    if t in text:
                        text = text.replace(t, "[REDACTED]")
        except Exception:
            pass  # Best-effort; regex is the safety net
        return text

    def redact_trace(self, trace: Dict[str, Any], use_llm: bool = True) -> Tuple[Dict[str, Any], int]:
        """Redact every string in a nested trace object."""
        total = 0
        trace = _deep_copy(trace)

        def recurse(obj: Any) -> None:
            nonlocal total
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, str):
                        new_v, c = self.redact(v)
                        if use_llm:
                            new_v = self.llm_redact(new_v)
                        obj[k] = new_v
                        total += c
                    else:
                        recurse(v)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, str):
                        new_item, c = self.redact(item)
                        if use_llm:
                            new_item = self.llm_redact(new_item)
                        obj[i] = new_item
                        total += c
                    else:
                        recurse(item)

        recurse(trace)
        return trace, total

# =============================================================================
# Quality Filtering (Ornstein v1 Parity)
# =============================================================================

class QualityFilter:
    def __init__(self, min_quality: float = DEFAULT_MIN_QUALITY) -> None:
        self.min_quality = min_quality

    def score(self, trace: Dict[str, Any]) -> Tuple[float, Dict[str, float], bool]:
        breakdown = {
            "reasoning_depth": self._score_reasoning_depth(trace),
            "structure": self._validate_structure(trace),
            "tool_calls": self._check_tool_calls(trace),
            "coherence": self._score_coherence(trace),
            "length": self._length_score(trace),
            "refusal": self._detect_refusal(trace),
        }
        weights = {
            "reasoning_depth": 0.20,
            "structure": 0.20,
            "tool_calls": 0.15,
            "coherence": 0.15,
            "length": 0.15,
            "refusal": 0.15,
        }
        composite = sum(breakdown[k] * weights[k] for k in weights)
        passed = composite >= self.min_quality
        return composite, breakdown, passed

    def _score_reasoning_depth(self, trace: Dict[str, Any]) -> float:
        text = _extract_all_text(trace).lower()
        messages = _extract_messages(trace)
        thinking_hits = sum(1 for tag in THINKING_TAGS if tag in text)
        substantial = sum(1 for m in messages if len(m.get("content", "")) > 100)
        
        # Enhanced scoring: reward deeper reasoning patterns
        reasoning_patterns = [
            r"let me (?:think|analyze|approach) this",
            r"i (?:think|believe|conclude|reason) that",
            r"first,? (?:i'll|we'll|let's)",
            r"step (?:by|wise|one)",
            r"this (?:suggests|indicates|means)",
            r"therefore|consequently|thus|hence",
            r"on the other hand|alternatively|however",
        ]
        
        pattern_hits = sum(1 for pattern in reasoning_patterns if re.search(pattern, text))
        
        # Bonus for multi-step reasoning
        multi_step = 0
        for m in messages:
            content = m.get("content", "")
            if len(content) > 200:  # Substantial response
                # Look for step-by-step indicators
                steps = re.findall(r'(?:first|second|third|then|next|finally|after that)', content.lower())
                if len(steps) >= 2:
                    multi_step += 1
        
        score = min(1.0, thinking_hits * 0.15 + substantial * 0.10 + pattern_hits * 0.08 + multi_step * 0.12)
        return max(0.0, score)

    def _validate_structure(self, trace: Dict[str, Any]) -> float:
        text = _extract_all_text(trace).lower()
        messages = _extract_messages(trace)
        score = 0.0
        if messages:
            score += 0.30
        if any(tag in text for tag in THINKING_TAGS):
            score += 0.30
        has_tools = any(tag in text for tag in TOOL_CALL_TAGS)
        has_results = any(tag in text for tag in TOOL_RESULT_TAGS)
        if has_tools and has_results:
            score += 0.20
        elif not has_tools:
            score += 0.20
        has_assistant = any(m.get("role", "") in ("assistant", "gpt") for m in messages)
        if has_assistant:
            score += 0.20
        return min(1.0, score)

    def _check_tool_calls(self, trace: Dict[str, Any]) -> float:
        text = _extract_all_text(trace)
        if not any(tag in text.lower() for tag in TOOL_CALL_TAGS):
            return 1.0
        blocks = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
        if not blocks:
            blocks = re.findall(r'<invoke>(.*?)</invoke>', text, re.DOTALL)
        if not blocks:
            return 0.5
        valid = 0
        for block in blocks:
            try:
                json.loads(block.strip())
                valid += 1
            except json.JSONDecodeError:
                pass
        return valid / len(blocks)

    def _score_coherence(self, trace: Dict[str, Any]) -> float:
        messages = _extract_messages(trace)
        if len(messages) < 2:
            return 0.3
        user_msgs = [m.get("content", "") for m in messages if m.get("role") in ("user", "human")]
        assistant_msgs = [m.get("content", "") for m in messages if m.get("role") in ("assistant", "gpt")]
        if not user_msgs or not assistant_msgs:
            return 0.3
        avg_user = sum(len(m) for m in user_msgs) / len(user_msgs)
        avg_assist = sum(len(m) for m in assistant_msgs) / len(assistant_msgs)
        if avg_assist < 20 and avg_user > 50:
            return 0.2
        
        # Enhanced coherence scoring
        # Turn alternation score
        normalized = []
        for m in messages:
            r = m.get("role", "")
            if r in ("user", "human"):
                normalized.append("user")
            elif r in ("assistant", "gpt"):
                normalized.append("assistant")
        switches = sum(1 for i in range(len(normalized) - 1) if normalized[i] != normalized[i + 1])
        ratio = switches / max(1, len(normalized) - 1)
        
        # Bonus for conversation flow indicators
        flow_score = 0.0
        for i, m in enumerate(messages):
            content = m.get("content", "").lower()
            if m.get("role") in ("assistant", "gpt") and i > 0:
                # Check if response references previous context
                prev_content = messages[i-1].get("content", "").lower()
                if prev_content:
                    # Look for contextual references
                    if any(word in content for word in ["yes", "no", "correct", "exactly", "right", "understand", "agree"]):
                        flow_score += 0.1
                    if any(pattern in content for pattern in ["based on", "from what", "as you", "given that"]):
                        flow_score += 0.15
        
        return 0.5 + (ratio * 0.5) + min(0.1, flow_score)

    def _length_score(self, trace: Dict[str, Any]) -> float:
        tokens = _estimate_tokens(_extract_all_text(trace))
        if tokens < DEFAULT_MIN_TOKENS:
            return max(0.0, tokens / DEFAULT_MIN_TOKENS)
        if tokens > DEFAULT_MAX_TOKENS:
            return max(0.0, 1.0 - ((tokens - DEFAULT_MAX_TOKENS) / DEFAULT_MAX_TOKENS))
        return 1.0

    def _detect_refusal(self, trace: Dict[str, Any]) -> float:
        messages = _extract_messages(trace)
        assistant_msgs = [m.get("content", "") for m in messages if m.get("role") in ("assistant", "gpt")]
        if not assistant_msgs:
            return 0.5
        refusal_count = 0
        for msg in assistant_msgs:
            lowered = msg.lower()
            for pat in REFUSAL_PATTERNS:
                if re.search(pat, lowered):
                    refusal_count += 1
                    break
        ratio = refusal_count / len(assistant_msgs)
        if ratio > DEFAULT_REFUSAL_RATIO:
            return 0.0
        return 1.0 - ratio

# =============================================================================
# 5-Factor Error Classification
# =============================================================================

class ErrorClassifier:
    """
    5-factor taxonomy for classifying error states in Hermes traces.
    Each trace gets exactly one label (first match wins) or "none".
    """

    FACTORS: Dict[str, Dict[str, Any]] = {
        "tool_failure": {
            "tags": ["<tool_error>", "<tool_failure>", "<api_error>"],
            "patterns": [
                r"\bHTTP\s+(?:4\d{2}|5\d{2}|\d{3})\b",
                r"\bConnection\s+(?:refused|reset|timed?-?out|error)\b",
                r"\b(?:API|tool)\s+(?:error|failed|unavailable|timeout)\b",
                r"\bRateLimitError\b|\bTooManyRequests\b",
                r"\bSSL\s+(?:error|handshake)\b",
                r"\bDNS\s+(?:error|lookup)\b",
                r"\bRequestException\b",
            ],
            "desc": "External tool / API call failed (HTTP error, timeout, connection refused, rate limit)",
        },
        "syntax_error": {
            "tags": ["<syntax_error>", "<parse_error>", "<execution_error>"],
            "patterns": [
                r"Traceback\s+\(most recent call last\)",
                r"\b(?:Syntax|Type|Name|Indentation|Value|Key|Index|Attribute|Import|ModuleNotFound|ZeroDivision)Error\b",
                r"\bException\b.*?:\s*",
                r"\bJSONDecodeError\b",
                r"\bnull\b|\bundefined\b|\bNoneType\b",
                r"\bmalformed\b|\binvalid\s+(?:JSON|XML|syntax|format)\b",
                r"\bEOFError\b|\bRuntimeError\b",
            ],
            "desc": "Code syntax, execution, or structured-data parsing failure",
        },
        "reasoning_error": {
            "tags": ["<reasoning_error>", "<hallucination>", "<contradiction>"],
            "patterns": [
                r"\b(?:wait|actually|no,?\s+that's\s+wrong)\b",
                r"\bI\s+(?:made|was)\s+(?:a\s+)?mistake\b",
                r"\bcorrection\b.*\b(?:apologies|sorry)\b",
                r"\bthat\s+(?:is|was)\s+incorrect\b",
                r"\b(?:hallucinat(?:ed|ion)|fabricated)\b",
                r"\bself-contradict(?:ory|ion)\b",
                r"\b(?:logically|mathematically)\s+impossible\b",
            ],
            "desc": "Wrong answer despite valid execution; hallucination, contradiction, or logical fallacy",
        },
        "safety_refusal": {
            "tags": ["<policy_violation>", "<unsafe>", "<jailbreak>"],
            "patterns": [
                r"\b(?:I\s+cannot|I\s+can't|I'm\s+unable\s+to)\s+(?:help|do|assist|generate|produce|create)\s+(?:with\s+)?(?:that|this|any)\b",
                r"\b(?:violates?|against)\s+(?:policy|guidelines|terms|safety)\b",
                r"\b(?:harmful|dangerous|illegal|unethical|toxic|biased)\s+(?:content|request|instruction)\b",
                r"\b(?:jailbreak|ignore\s+previous\s+instructions|DAN\s+mode)\b",
                r"\b(?:redteam|red-team|adversarial\s+prompt)\b",
            ],
            "desc": "Policy violation, inappropriate refusal, or adversarial / jailbreak trace",
        },
        "timeout_stall": {
            "tags": ["<timeout>", "<truncated>", "<stall>"],
            "patterns": [
                r"\b(?:timeout|timed?\s*out)\b",
                r"\b(?:truncated|cut\s+off|incomplete)\s+(?:response|output|reply)\b",
                r"\b(?:hung|frozen|stalled|no\s+response)\b",
                r"\b(?:aborted|cancelled|interrupted)\b",
            ],
            "desc": "Trace ended prematurely, hung, or produced an incomplete / empty turn",
        },
    }

    @classmethod
    def classify(cls, trace: Dict[str, Any]) -> str:
        """
        Returns one of the 5 factor names, or 'none' if no error detected.
        Checks in priority order: tool > syntax > reasoning > safety > timeout.
        """
        text = _extract_all_text(trace)
        lowered = text.lower()

        for factor_name, config in cls.FACTORS.items():
            # 1. Explicit tags (strong signal)
            for tag in config["tags"]:
                if tag in lowered:
                    return factor_name
            # 2. Regex patterns
            for pat in config["patterns"]:
                if re.search(pat, text, re.IGNORECASE):
                    return factor_name

        return "none"

    @classmethod
    def factor_names(cls) -> List[str]:
        return list(cls.FACTORS.keys()) + ["none"]

# =============================================================================
# Diversity / Deduplication
# =============================================================================

class DiversityFilter:
    """
    Lexical-diversity deduplication.
    Compares traces on their *semantic meat* (reasoning, user queries,
    non-tool assistant prose) while ignoring boilerplate tool-call JSON.
    Two traces with identical tool calls but different reasoning are NOT duplicates.
    Enhanced with better tool-call handling and improved similarity metrics.
    """

    def __init__(self, max_similarity: float = DEFAULT_MAX_SIMILARITY) -> None:
        self.max_similarity = max_similarity
        self._hashes: set = set()
        self._texts: List[str] = []

    @staticmethod
    def _normalize(trace: Dict[str, Any]) -> str:
        """Strip tool calls, results, and excessive whitespace for lexical comparison."""
        text = _extract_all_text(trace)
        # Strip tool call blocks (highly redundant JSON)
        for tag_open, tag_close in [
            ("<tool_call>", "</tool_call>"),
            ("<tool_result>", "</tool_result>"),
            ("<function_calls>", "</function_calls>"),
            ("<function_result>", "</function_result>"),
            ("<invoke>", "</invoke>"),
            ("<output>", "</output>"),
        ]:
            text = re.sub(
                re.escape(tag_open) + r".*?" + re.escape(tag_close),
                "",
                text,
                flags=re.DOTALL,
            )
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text

    def is_duplicate(self, trace: Dict[str, Any]) -> bool:
        text = self._normalize(trace)
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if h in self._hashes:
            return True
        self._hashes.add(h)
        if self.max_similarity >= 1.0:
            self._texts.append(text)
            return False

        for seen in self._texts:
            sim = self._ngram_similarity(text, seen)
            if sim > self.max_similarity:
                return True
        self._texts.append(text)
        return False

    @staticmethod
    def _ngram_similarity(a: str, b: str, n: int = 5) -> float:
        def grams(s: str):
            return {s[i:i + n] for i in range(len(s) - n + 1)}
        ga, gb = grams(a), grams(b)
        if not ga or not gb:
            return 0.0
        inter = len(ga & gb)
        union = len(ga | gb)
        return inter / union if union else 0.0

    @staticmethod
    def _tfidf_similarity(a: str, b: str) -> Optional[float]:
        if not HAS_SKLEARN:
            return None
        try:
            vec = TfidfVectorizer(stop_words="english", max_features=5000)
            mat = vec.fit_transform([a, b])
            sim = cosine_similarity(mat[0:1], mat[1:2])[0][0]
            return float(sim)
        except Exception:
            return None


class SemanticDiversityFilter:
    """
    Semantic deduplication using sentence embeddings.
    Computes dense-vector similarity on normalized trace text
    (tool-call boilerplate stripped) to catch near-duplicates
    that lexical n-gram / TF-IDF miss.

    Falls back to lexical DiversityFilter if sentence-transformers
    is not installed.
    Enhanced with better normalization and hybrid similarity scoring.
    """

    def __init__(
        self,
        max_similarity: float = DEFAULT_MAX_SIMILARITY,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.max_similarity = max_similarity
        self._hashes: set = set()
        self._embeddings: List[Any] = []
        self._model: Optional[Any] = None
        self._model_name = model_name
        self._lexical_filter = DiversityFilter(max_similarity=1.0)  # For exact matches only

        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self._model = SentenceTransformer(model_name)
                logger.info("SemanticDiversityFilter loaded %s", model_name)
            except Exception as exc:
                logger.warning("Could not load sentence-transformers model %s: %s", model_name, exc)
        else:
            logger.warning(
                "sentence-transformers not installed; SemanticDiversityFilter "
                "will fall back to lexical DiversityFilter behaviour."
            )

    @staticmethod
    def _normalize(trace: Dict[str, Any]) -> str:
        """Strip tool calls, results, and excessive whitespace."""
        text = _extract_all_text(trace)
        for tag_open, tag_close in [
            ("<tool_call>", "</tool_call>"),
            ("<tool_result>", "</tool_result>"),
            ("<function_calls>", "</function_calls>"),
            ("<function_result>", "</function_result>"),
            ("<invoke>", "</invoke>"),
            ("<output>", "</output>"),
        ]:
            text = re.sub(
                re.escape(tag_open) + r".*?" + re.escape(tag_close),
                "",
                text,
                flags=re.DOTALL,
            )
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text

    def is_duplicate(self, trace: Dict[str, Any]) -> bool:
        text = self._normalize(trace)
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if h in self._hashes:
            return True
        self._hashes.add(h)

        if self._model is not None:
            try:
                emb = self._model.encode(text, convert_to_tensor=False)
                for seen_emb in self._embeddings:
                    # Calculate cosine similarity
                    dot_product = np.dot(emb, seen_emb)
                    norm_emb = np.linalg.norm(emb)
                    norm_seen = np.linalg.norm(seen_emb)
                    if norm_emb == 0 or norm_seen == 0:
                        sim = 0.0
                    else:
                        sim = float(dot_product / (norm_emb * norm_seen))
                    if sim > self.max_similarity:
                        return True
                self._embeddings.append(emb)
            except Exception as exc:
                logger.debug("Semantic comparison failed: %s", exc)
        return False


# =============================================================================
# Scenario Extraction (for synthetic trace generation)
# =============================================================================

class ScenarioExtractor:
    """
    Extract user-facing scenarios / prompts from traces so others can
    generate synthetic training data from real-world task distributions.
    Enhanced to capture multi-turn conversations and semantic patterns.
    """

    CATEGORIES = {
        "coding": [
            "python", "script", "function", "implement", "code", "build a",
            "api", "docker", "sql", "bash", "shell", "javascript", "typescript",
            "go server", "html", "css", "algorithm", "sort", "cache", "regex",
            "github actions", "makefile", "cli tool", "web socket", "orm",
            "fernet", "encrypt", "hash", "generator", "class that", "decorator",
            "merge sort", "binary search", "bloom filter", "dijkstra", "quicksort",
            "lru cache", "trie", "observer", "circuit breaker", "rate limiter",
        ],
        "reasoning": [
            "solve:", "probability", "prove", "explain why", "paradox",
            "monty hall", "prisoner", "expected value", "how many", "if you have",
            "if a", "if it takes", "solve the", "towers of hanoi", "coin flip",
            "hat puzzle", "logic", "riddle", "math", "calculate", "optimise",
            "what is the smallest", "what is the probability", "gcd", "lcm",
            "riemann", "pigeonhole", "bayes", "correlation", "causation",
        ],
        "creative": [
            "poem", "haiku", "limerick", "sonnet", "story", "rap", "verse",
            "joke", "humorous", "sci-fi", "dialogue", "draft a", "write a",
            "create a slogan", "marketing", "logo", "design a", "elevator pitch",
            "apology letter", "resignation", "thank-you note", "user manual",
        ],
        "tool_use": [
            "look up", "search for", "find the", "look up the", "search the web",
            "current weather", "current price", "exchange rate", "stock price",
            "population of", "melting point", "atomic number", "distance between",
            "release date", "plot summary", "latest news", "weather in",
            "rules of", "definition of", "meaning of", "inventor of",
        ],
        "science": [
            "how does", "how do", "explain how", "what causes", "what is the",
            "higgs boson", "black hole", "dna", "rna", "antibiotics", "vaccine",
            "evolution", "natural selection", "ozone", "aurora", "rainbow",
            "el nino", "big bang", "exoplanet", "climate", "water cycle",
            "jet engine", "transistor", "fiber optic", "seismograph", "radar",
            "sonar", "thermocouple", "heliostat", "particle accelerator",
            "quantum", "relativity", "newton", "einstein", "photon", "neutron",
        ],
        "history": [
            "what was", "explain the", "causes of", "treaty of", "war and",
            "revolution", "berlin wall", "cold war", "world war", "battle of",
            "arab spring", "black death", "silk road", "opium war", "gettysburg",
            "nuremberg", "yalta", "versailles", "warsaw ghetto", "holocaust",
            "decolonization", "warsaw pact", "zimmermann", "tiananmen",
        ],
        "business": [
            "401(k)", "ipo", "stock market", "etf", "mutual fund", "bull market",
            "bear market", "revenue", "profit", "fiscal policy", "monetary policy",
            "gdp", "gnp", "venture capital", "startup", "salary", "negotiate",
            "hire", "contractor", "freelance", "consulting", "cloud provider",
            "backup strategy", "monitoring stack", "ci/cd", "ab testing",
            "home server", "vpn", "recommendation engine", "search engine",
        ],
        "philosophy": [
            "meaning of life", "free will", "determinism", "consciousness",
            "social contract", "civil disobedience", "enlightenment", "groupthink",
            "cognitive dissonance", "bystander effect", "fundamental attribution",
            "prisoner's dilemma", "arrow impossibility", "trolley problem",
            "just war", "ethics", "moral", "justice", "fairness", "rights",
            "authority", "expertise", "patriotism", "nationalism", "faith",
            "belief", "destiny", "fate",
        ],
        "multi_turn": [
            "ask me", "ask about", "help me plan", "help me choose", "clarify",
            "recommend", "suggest", "what do you think", "what would you",
            "let's discuss", "walk me through", "step by step", "explain to me",
        ],
    }

    COMPLEXITY_KEYWORDS = {
        "hard": [
            "implement", "design", "architecture", "optimise", "thread-safe",
            "distributed", "circuit breaker", "bloom filter", "dijkstra",
            "merge sort", "quicksort", "binary search tree", "topological sort",
            "synthetic", "ablation", "hyperparameter", "embedding",
        ],
        "medium": [
            "build", "create", "write a python", "docker compose", "regex",
            "sql query", "bash script", "generator", "decorator", "context manager",
        ],
    }

    TOOL_KEYWORDS = [
        "look up", "search", "find the current", "find the latest",
        "current price", "current weather", "population of", "distance between",
        "melting point", "atomic number", "release date", "plot summary",
    ]

    @classmethod
    def infer_category(cls, prompt: str) -> str:
        p = prompt.lower()
        scores: Dict[str, int] = {}
        for cat, keywords in cls.CATEGORIES.items():
            scores[cat] = sum(1 for kw in keywords if kw in p)
        if scores:
            best = max(scores, key=scores.get)
            if scores[best] > 0:
                return best
        return "general"

    @classmethod
    def infer_complexity(cls, prompt: str) -> str:
        p = prompt.lower()
        words = len(p.split())
        for level, keywords in cls.COMPLEXITY_KEYWORDS.items():
            if any(kw in p for kw in keywords):
                return level
        if words > 25:
            return "hard"
        if words > 12:
            return "medium"
        return "simple"

    @classmethod
    def requires_tools(cls, prompt: str) -> bool:
        p = prompt.lower()
        return any(kw in p for kw in cls.TOOL_KEYWORDS)

    @classmethod
    def is_multi_turn(cls, prompt: str) -> bool:
        """Detect if this is a multi-turn clarification scenario."""
        p = prompt.lower()
        multi_turn_keywords = [
            "ask me", "ask about", "help me plan", "help me choose", 
            "clarify", "recommend", "suggest", "what do you think",
            "what would you", "let's discuss", "walk me through",
            "step by step", "explain to me", "guide me", "show me how"
        ]
        return any(kw in p for kw in multi_turn_keywords)

    @classmethod
    def extract(cls, trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        messages = _extract_messages(trace)
        if not messages:
            return None
        
        # Find all user messages to capture multi-turn conversations
        user_messages = [m for m in messages if m.get("role") in ("user", "human")]
        if not user_messages:
            return None
        
        # Get the first user message as the primary scenario
        primary_prompt = user_messages[0].get("content", "").strip()
        if not primary_prompt:
            return None
        
        scenario = {
            "scenario": primary_prompt,
            "category": cls.infer_category(primary_prompt),
            "complexity": cls.infer_complexity(primary_prompt),
            "requires_tools": cls.requires_tools(primary_prompt),
            "source_session_id": trace.get("session_id", ""),
            "word_count": len(primary_prompt.split()),
            "is_multi_turn": cls.is_multi_turn(primary_prompt),
            "turn_count": len(user_messages),
        }
        
        # If multi-turn, capture the full conversation context
        if len(user_messages) > 1:
            scenario["follow_ups"] = [
                m.get("content", "") for m in user_messages[1:] 
                if m.get("content", "").strip()
            ]
            scenario["full_conversation"] = " | ".join(
                m.get("content", "") for m in user_messages
            )
        
        return scenario

    @classmethod
    def to_jsonl(cls, scenarios: List[Dict[str, Any]], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for s in scenarios:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    @classmethod
    def to_markdown(cls, scenarios: List[Dict[str, Any]], path: Path) -> None:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for s in scenarios:
            grouped.setdefault(s["category"], []).append(s)

        lines = [
            "# Ornstein Scenario Dataset",
            "",
            "Real-world agent task prompts extracted from curated Hermes traces.",
            "Use these to bootstrap synthetic trace generation or evaluate agents.",
            "",
            f"**Total scenarios:** {len(scenarios)}",
            f"**Categories:** {len(grouped)}",
            "",
            "---",
            "",
        ]
        for cat in sorted(grouped.keys()):
            items = grouped[cat]
            multi_turn_count = sum(1 for i in items if i.get("is_multi_turn"))
            lines.append(f"## {cat.title()}")
            lines.append(f"*{len(items)} scenarios ({multi_turn_count} multi-turn)*")
            lines.append("")
            for item in items:
                tool_badge = " 🛠" if item["requires_tools"] else ""
                multi_badge = " 🔄" if item.get("is_multi_turn") else ""
                lines.append(
                    f"- **{item['complexity'].title()}**{tool_badge}{multi_badge}: {item['scenario']}"
                )
                if item.get("follow_ups"):
                    for i, follow_up in enumerate(item["follow_ups"][:2]):  # Limit to 2 follow-ups
                        lines.append(f"  - Follow-up {i+1}: {follow_up[:100]}...")
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Formatters
# =============================================================================

class AxolotlFormatter:
    @classmethod
    def format(
        cls,
        trace: Dict[str, Any],
        quality_score: float,
        source_id: str,
        error_class: str = "none",
    ) -> Dict[str, Any]:
        messages = _extract_messages(trace)
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        normalized = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role in ("human", "user"):
                normalized.append({"role": "user", "content": content})
            elif role in ("gpt", "assistant"):
                normalized.append({"role": "assistant", "content": content})
            elif role == "system":
                normalized.append({"role": "system", "content": content})

        thinking = cls._extract_thinking(trace)
        tags = cls._tag_trace(trace)
        if error_class != "none":
            tags.append(error_class)

        return {
            "messages": normalized,
            "thinking": thinking,
            "quality_score": round(quality_score, 2),
            "error_class": error_class,
            "format_version": "ornstein-v2",
            "source_session_id": source_id,
            "trace_length": sum(len(m.get("content", "")) for m in normalized),
            "tags": tags,
        }

    @staticmethod
    def _extract_thinking(trace: Dict[str, Any]) -> str:
        text = json.dumps(trace)
        parts: List[str] = []
        for tag in THINKING_TAGS:
            close = tag.replace("<", "</")
            pat = re.compile(re.escape(tag) + r"(.*?)" + re.escape(close), re.DOTALL)
            parts.extend(pat.findall(text))
        return "\n".join(parts).strip()[:4096]

    @staticmethod
    def _tag_trace(trace: Dict[str, Any]) -> List[str]:
        text = json.dumps(trace).lower()
        tags: List[str] = []
        if any(tag in text for tag in TOOL_CALL_TAGS):
            tags.append("tool-use")
        if len(_extract_messages(trace)) > 4:
            tags.append("multi-turn")
        if any(tag in text for tag in THINKING_TAGS):
            tags.append("reasoning")
        if not tags:
            tags.append("general")
        return tags


class ShareGPTFormatter:
    @classmethod
    def format(cls, trace: Dict[str, Any]) -> Dict[str, Any]:
        messages = _extract_messages(trace)
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        conversations = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role in ("human", "user"):
                conversations.append({"from": "human", "value": content})
            elif role in ("gpt", "assistant"):
                conversations.append({"from": "gpt", "value": content})
            elif role == "system":
                conversations.append({"from": "system", "value": content})

        return {"conversations": conversations}

# =============================================================================
# Dataset Card
# =============================================================================

class DatasetCardBuilder:
    TEMPLATE = """# {repo_name}

## Dataset Card — Ornstein Curated v2
Generated by **trace-curator** on {date}.

## Statistics
- **Total traces:** {total}
- **Exact duplicates removed:** {discarded}
- **Final examples:** {kept}
- **Average quality score:** {avg_quality:.2f}

## Quality Score Distribution
{score_distribution}

## 5-Factor Error Classification
{error_distribution}

## Note
All traces are included in the output; no quality-based filtering is applied.
Only exact duplicates are removed. Use the `quality_score` field in each record to filter downstream.

## Example Entry (Axolotl format)
```json
{example}
```

## Recommended Axolotl YAML
```yaml
datasets:
  - path: {repo_id}
    ds_type: json
    type: messages
    conversation: messages

split: train
val_size: 0.05
```

## Files
- `data.jsonl` — Axolotl-native messages format (primary)
- `sharegpt.jsonl` — Classic ShareGPT conversations format (compatibility)
- `dataset_card.md` — This file
"""

    @classmethod
    def build(cls, stats: Dict[str, Any], example: Dict[str, Any]) -> str:
        return cls.TEMPLATE.format(
            repo_name=stats.get("repo_name", "ornstein-curated-v2"),
            date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            total=stats["kept"] + stats["discarded"],
            kept=stats["kept"],
            discarded=stats["discarded"],
            avg_quality=stats["avg_quality"],
            score_distribution=cls._fmt_distribution(stats.get("score_distribution", {}), suffix="traces"),
            error_distribution=cls._fmt_error_distribution(stats.get("error_distribution", {})),
            example=json.dumps(example, indent=2, ensure_ascii=False),
            repo_id=stats.get("repo_id", "DJLougen/ornstein-curated-v2"),
        )

    @staticmethod
    def _fmt_distribution(dist: Dict[str, int], suffix: str = "traces") -> str:
        if not dist:
            return "- No distribution data available."
        lines = []
        for bucket in sorted(dist.keys()):
            lines.append(f"- **{bucket}:** {dist[bucket]} {suffix}")
        return "\n".join(lines)

    @staticmethod
    def _fmt_error_distribution(dist: Dict[str, int]) -> str:
        if not dist:
            return "- No errors detected."
        factor_desc = {
            "tool_failure": "External tool / API call failed",
            "syntax_error": "Code syntax or execution failure",
            "reasoning_error": "Wrong answer / hallucination / contradiction",
            "safety_refusal": "Policy violation or inappropriate refusal",
            "timeout_stall": "Trace ended prematurely or stalled",
            "none": "Clean — no error detected",
        }
        lines = []
        for factor in ErrorClassifier.factor_names():
            count = dist.get(factor, 0)
            desc = factor_desc.get(factor, factor)
            lines.append(f"- **{desc}:** {count}")
        return "\n".join(lines)

# =============================================================================
# HuggingFace Hub Upload
# =============================================================================

def push_to_hub(output_dir: Path, repo_id: str, private: bool = True) -> None:
    if not HAS_HF:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        return

    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        logger.info("HF dataset repo ready: %s", repo_id)
    except Exception as exc:
        logger.error("Failed to create HF repo: %s", exc)
        return

    for fname in ("data.jsonl", "sharegpt.jsonl", "dataset_card.md"):
        fpath = output_dir / fname
        if not fpath.exists():
            continue
        try:
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=fname,
                repo_id=repo_id,
                repo_type="dataset",
            )
            logger.info("Uploaded %s → %s", fname, repo_id)
        except Exception as exc:
            logger.error("Failed to upload %s: %s", fname, exc)

    logger.info("Dataset live at https://huggingface.co/datasets/%s", repo_id)

# =============================================================================
# Mock Data Generator (for testing without real sessions)
# =============================================================================

def generate_mock_sessions(output_dir: Optional[str] = None) -> Path:
    """Create a handful of synthetic Hermes traces for validation."""
    sessions_dir = Path.home() / ".hermes" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    good_trace = {
        "session_id": "20250422_143022_abc123",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Build a Python script that fetches weather data."},
            {"role": "assistant", "content": (
                "<thinking>\nThe user wants a weather fetcher. I'll use requests and a public API.\n"
                "I should include error handling and a simple CLI interface.\n</thinking>\n\n"
                "Here is a complete script:\n```python\nimport requests\ndef get_weather(city):\n"
                "    url = f'https://api.weather.example/v1/{city}'\n    return requests.get(url).json()\n```"
            )},
            {"role": "user", "content": "Can you add temperature unit conversion?"},
            {"role": "assistant", "content": (
                "<thinking>\nAdd celsius/fahrenheit toggle with a helper function.\n</thinking>\n\n"
                "Updated script:\n```python\ndef c_to_f(c):\n    return c * 9/5 + 32\n```"
            )},
        ],
    }

    tool_trace = {
        "session_id": "20250422_150055_def456",
        "messages": [
            {"role": "user", "content": "Search the web for 'Python 3.12 features'."},
            {"role": "assistant", "content": (
                "<tool_call>\n{\"name\": \"web_search\", \"arguments\": {\"query\": \"Python 3.12 features\"}}\n</tool_call>"
            )},
            {"role": "system", "content": (
                "<tool_result>\n{\"results\": [{\"title\": \"What's New In Python 3.12\"}]}\n</tool_result>"
            )},
            {"role": "assistant", "content": "Here are the top results for Python 3.12 features..."},
        ],
    }

    refusal_trace = {
        "session_id": "20250422_151200_bad000",
        "messages": [
            {"role": "user", "content": "What is the meaning of life?"},
            {"role": "assistant", "content": "I'm sorry, I can't answer philosophical questions."},
        ],
    }

    short_trace = {
        "session_id": "20250422_152000_short",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ],
    }

    pii_trace = {
        "session_id": "20250422_153000_pii",
        "messages": [
            {"role": "user", "content": "My email is alice@example.com and my API key is sk-live-51AaBbCcDdEeFfGgHhIiJjKk"},
            {"role": "assistant", "content": "Thanks, I've noted your contact info at C:\\Users\\Alice\\docs."},
        ],
    }

    traces = [good_trace, tool_trace, refusal_trace, short_trace, pii_trace]
    out_file = sessions_dir / "mock_sessions.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    logger.info("Mock sessions written to %s (%d traces)", out_file, len(traces))
    return out_file

# =============================================================================
# Main Pipeline
# =============================================================================

class ScenarioGenerator:
    """
    Generates new scenarios based on patterns learned from existing traces.
    Uses template-based generation with intelligent variation.
    """

    # Template categories with parameterized placeholders
    CODE_TEMPLATES = [
        "Write a Python {function} that {action} {object} using {algorithm} algorithm.",
        "Build a {type} server with {feature1}, {feature2}, and {feature3}.",
        "Create a CLI tool that converts {format1} to {format2} with error handling.",
        "Implement a {data_structure} in Python with {operation1} and {operation2} methods.",
        "Write a {language} script that {action} {object} and handles {error_type} errors.",
    ]

    REASONING_TEMPLATES = [
        "Solve this problem: {problem_statement}. Show your work step by step.",
        "If {condition1}, what is the probability that {condition2}? Explain your reasoning.",
        "A {object} has {property1} and {property2}. Calculate {target_property}.",
        "Explain why {fact1} implies {fact2} using {method} approach.",
        "Prove that {statement} is true by contradiction.",
    ]

    CREATIVE_TEMPLATES = [
        "Write a {poem_type} about {topic} that incorporates {element} metaphor.",
        "Draft a {document_type} for {audience} explaining {concept} in simple terms.",
        "Create a {format} about {character} who learns {lesson}.",
        "Write a {genre} story where {conflict} leads to {resolution}.",
        "Generate a {format} comparing {concept1} and {concept2} with examples.",
    ]

    TOOL_USE_TEMPLATES = [
        "Search the web for the latest information about {topic} and summarize key findings.",
        "Find the current {metric} for {entity} and explain recent trends.",
        "Look up how {technology} works and create a beginner-friendly explanation.",
        "Research the history of {concept} and create a timeline of key developments.",
        "Find current data about {topic} and create a comparative analysis.",
    ]

    MULTI_TURN_TEMPLATES = [
        "Help me design a {system} for {use_case}. Ask me about requirements first.",
        "I want to build {project}. Walk me through the architecture decisions.",
        "Guide me through implementing {feature}. Ask clarifying questions as needed.",
        "Help me choose between {option1} and {option2} for {use_case}.",
        "I need to solve {problem}. What information do you need from me to help?",
    ]

    # Parameter libraries for intelligent substitution
    PARAMETERS = {
        "function": ["class", "decorator", "generator", "context manager", "iterator"],
        "action": ["process", "transform", "validate", "optimize", "compress"],
        "object": ["data streams", "file systems", "API responses", "user input", "config files"],
        "algorithm": ["merge sort", "binary search", "Dijkstra", "BFS", "A*"],
        "type": ["web", "API", "database", "message queue", "cache"],
        "feature1": ["authentication", "rate limiting", "caching", "logging", "monitoring"],
        "feature2": ["error handling", "input validation", "data serialization", "compression", "encryption"],
        "feature3": ["load balancing", "circuit breaking", "retry logic", "circuit breaker", "health checks"],
        "format1": ["CSV", "JSON", "XML", "YAML", "HTML"],
        "format2": ["JSON", "Parquet", "Avro", "MessagePack", "Protocol Buffers"],
        "data_structure": ["binary search tree", "graph", "trie", "heap", "segment tree"],
        "operation1": ["insert", "delete", "search", "update", "traverse"],
        "operation2": ["delete", "search", "balance", "serialize", "validate"],
        "language": ["Python", "Bash", "JavaScript", "TypeScript", "Rust"],
        "error_type": ["network", "file system", "database", "authentication", "validation"],
        "problem_statement": ["a train leaves station A at 60mph while another leaves station B at 80mph, when do they meet?", 
                             "you have 12 coins, one is counterfeit (lighter), find it in 3 weighings",
                             "a pond with algae doubling every day becomes full on day 30, when was it half full?"],
        "condition1": ["it rains today", "you roll a 6 on a die", "the first card drawn is an ace"],
        "condition2": ["you get a promotion", "the second roll is also a 6", "the second card is a king"],
        "object": ["square", "triangle", "cylinder", "parabola", "exponential function"],
        "property1": ["area of 100", "perimeter of 50", "volume of 200", "slope of 2", "coefficient of 3"],
        "property2": ["height of 10", "base of 20", "radius of 5", "intercept of 4", "rate of 0.5"],
        "target_property": ["the volume", "the surface area", "the derivative", "the integral", "the minimum value"],
        "fact1": ["all squares are rectangles", "the derivative of x² is 2x", "parallel lines never meet"],
        "fact2": ["not all rectangles are squares", "the derivative of x³ is 3x²", "perpendicular lines intersect at 90°"],
        "method": ["geometric", "algebraic", "calculus", "inductive", "contradiction"],
        "statement": ["there are infinitely many prime numbers", "the square root of 2 is irrational", "0.999... equals 1"],
        "poem_type": ["haiku", "sonnet", "limerick", "free verse", "acrostic"],
        "topic": ["artificial intelligence", "climate change", "space exploration", "music", "cooking"],
        "element": ["water", "fire", "light", "time", "gravity"],
        "document_type": ["tutorial", "blog post", "technical documentation", "user manual", "FAQ"],
        "audience": ["beginners", "intermediate users", "experts", "managers", "students"],
        "concept": ["blockchain", "machine learning", "quantum computing", "internet of things", "virtual reality"],
        "format": ["comparison table", "timeline", "case study", "how-to guide", "infographic description"],
        "character": ["a young programmer", "an old scientist", "a curious child", "a retired engineer", "a startup founder"],
        "lesson": ["the value of persistence", "the importance of asking questions", "that failure is learning", "teamwork beats solo work", "sometimes the simplest solution is best"],
        "genre": ["sci-fi", "mystery", "comedy", "drama", "educational"],
        "conflict": ["facing a deadline", "discovering a bug", "choosing between two approaches", "learning a new technology", "explaining complex concepts"],
        "resolution": ["success through iteration", "breakthrough insight", "collaboration", "simplified approach", "mentor guidance"],
        "concept1": ["REST API", "GraphQL", "microservices", "monolithic architecture", "serverless"],
        "concept2": ["GraphQL", "microservices", "monolithic architecture", "serverless", "event-driven architecture"],
        "topic": ["quantum computing breakthroughs", "AI safety research", "sustainable technology", "space exploration", "biotechnology"],
        "metric": ["stock price", "exchange rate", "temperature", "population", "unemployment rate"],
        "entity": ["Bitcoin", "NVIDIA", "Tokyo", "India", "the Federal Reserve"],
        "technology": ["5G networks", "CRISPR gene editing", "quantum cryptography", "solid-state batteries", "brain-computer interfaces"],
        "concept": ["democracy", "artificial intelligence", "renewable energy", "genetic engineering", "space colonization"],
        "system": ["database", "cache layer", "message queue", "authentication system", "monitoring stack"],
        "use_case": ["handling 1M+ users", "real-time analytics", "fraud detection", "content delivery", "log aggregation"],
        "project": ["a distributed web scraper", "a real-time chat application", "a recommendation engine", "a CI/CD pipeline", "a data pipeline"],
        "feature": ["user authentication", "real-time updates", "file uploads", "search functionality", "notifications"],
        "problem": ["scaling a monolithic application", "reducing API response times", "handling database migrations", "implementing rate limiting", "optimizing build times"],
        "option1": ["PostgreSQL", "MongoDB", "Redis", "Elasticsearch", "Cassandra"],
        "option2": ["MongoDB", "Redis", "Elasticsearch", "Cassandra", "PostgreSQL"],
    }

    @classmethod
    def generate_random_scenario(cls, category: str = None) -> Dict[str, Any]:
        """Generate a single random scenario of specified category."""
        if category is None:
            category = random.choice(list(cls.get_all_templates().keys()))
        
        templates = cls.get_templates_for_category(category)
        if not templates:
            return None
        
        template = random.choice(templates)
        # Fill in placeholders with random parameters
        filled = cls._fill_template(template)
        
        return {
            "scenario": filled,
            "category": category,
            "complexity": cls._infer_complexity(filled),
            "requires_tools": cls._requires_tools(filled),
            "is_multi_turn": category == "multi_turn",
            "turn_count": 1 if category != "multi_turn" else 3,
        }

    @classmethod
    def get_templates_for_category(cls, category: str) -> List[str]:
        """Get template list for a category."""
        template_map = {
            "coding": cls.CODE_TEMPLATES,
            "reasoning": cls.REASONING_TEMPLATES,
            "creative": cls.CREATIVE_TEMPLATES,
            "tool_use": cls.TOOL_USE_TEMPLATES,
            "multi_turn": cls.MULTI_TURN_TEMPLATES,
        }
        return template_map.get(category, [])

    @classmethod
    def get_all_templates(cls) -> Dict[str, List[str]]:
        """Get all templates by category."""
        return {
            "coding": cls.CODE_TEMPLATES,
            "reasoning": cls.REASONING_TEMPLATES,
            "creative": cls.CREATIVE_TEMPLATES,
            "tool_use": cls.TOOL_USE_TEMPLATES,
            "multi_turn": cls.MULTI_TURN_TEMPLATES,
        }

    @classmethod
    def _fill_template(cls, template: str) -> str:
        """Fill template placeholders with random parameters."""
        result = template
        for placeholder in cls.PARAMETERS:
            if "{" + placeholder + "}" in result:
                options = cls.PARAMETERS[placeholder]
                result = result.replace("{" + placeholder + "}", random.choice(options))
        return result

    @classmethod
    def _infer_complexity(cls, scenario: str) -> str:
        """Infer complexity from generated scenario."""
        words = len(scenario.split())
        if words > 20:
            return "hard"
        if words > 12:
            return "medium"
        return "simple"

    @classmethod
    def _requires_tools(cls, scenario: str) -> bool:
        """Check if scenario requires tool use."""
        tool_keywords = ["search", "find", "look up", "current", "latest"]
        return any(kw in scenario.lower() for kw in tool_keywords)

    @classmethod
    def generate_batch(cls, count: int, category: str = None) -> List[Dict[str, Any]]:
        """Generate a batch of scenarios."""
        scenarios = []
        for _ in range(count):
            scenario = cls.generate_random_scenario(category)
            if scenario:
                scenarios.append(scenario)
        return scenarios

    @classmethod
    def generate_diverse_batch(cls, total_count: int) -> List[Dict[str, Any]]:
        """Generate diverse scenarios across all categories."""
        categories = list(cls.get_all_templates().keys())
        scenarios = []
        
        for category in categories:
            category_count = max(1, total_count // len(categories))
            scenarios.extend(cls.generate_batch(category_count, category))
        
        # Shuffle to mix categories
        random.shuffle(scenarios)
        return scenarios[:total_count]


# Add scenario generation to main pipeline
def generate_scenarios(count: int = 100, output_dir: str = "./scenarios") -> None:
    """Generate synthetic scenarios and save to files."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Generate diverse scenarios
    scenarios = ScenarioGenerator.generate_diverse_batch(count)
    
    # Save to JSONL
    with open(out_path / "generated_scenarios.jsonl", "w", encoding="utf-8") as f:
        for scenario in scenarios:
            f.write(json.dumps(scenario, ensure_ascii=False) + "\n")
    
    # Save to markdown
    lines = [
        "# Generated Scenarios",
        "",
        f"Generated {len(scenarios)} synthetic scenarios for testing.",
        "",
    ]
    
    # Group by category
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for scenario in scenarios:
        grouped.setdefault(scenario["category"], []).append(scenario)
    
    for category, category_scenarios in sorted(grouped.items()):
        lines.append(f"## {category.title()}")
        lines.append(f"*{len(category_scenarios)} scenarios*")
        lines.append("")
        for scenario in category_scenarios:
            lines.append(f"- **{scenario['complexity'].title()}**: {scenario['scenario']}")
        lines.append("")
    
    with open(out_path / "generated_scenarios.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    logger.info("Generated %d scenarios to %s", len(scenarios), out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TALOS Trace Curator - Enhanced version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Existing arguments...
    parser.add_argument("--input-dir", type=str, help="Directory with raw JSONL session files")
    parser.add_argument("--input-file", type=str, help="Single JSONL file to process")
    parser.add_argument("--session-id", type=str, help="Process a single session by ID")
    parser.add_argument("--output-dir", type=str, default="./curated-dataset", help="Output directory")
    parser.add_argument("--min-quality", type=float, default=DEFAULT_MIN_QUALITY, help="Quality score threshold")
    parser.add_argument("--anonymize-level", choices=["standard", "strict"], default="strict")
    parser.add_argument("--max-similarity", type=float, default=DEFAULT_MAX_SIMILARITY)
    parser.add_argument("--semantic-dedup", action="store_true", help="Use sentence-transformer embeddings")
    parser.add_argument("--export-scenarios", action="store_true", help="Export scenarios from traces")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--repo-id", type=str, default="DJLougen/ornstein-curated-v2")
    parser.add_argument("--public", action="store_true", help="Make HF repo public")
    parser.add_argument("--generate-mock", action="store_true", help="Create mock data")
    parser.add_argument("--no-llm-redact", action="store_true", help="Skip LLM redaction")
    parser.add_argument("--exclude-errors", action="store_true", help="Write clean-only output")
    parser.add_argument("--skip-redact", action="store_true", help="Skip PII redaction")
    
    # New scenario generation argument
    parser.add_argument("--generate-scenarios", type=int, help="Generate N synthetic scenarios")
    
    args = parser.parse_args()
    
    if args.generate_scenarios:
        generate_scenarios(args.generate_scenarios)
        sys.exit(0)
    
    if args.generate_mock:
        generate_mock_sessions()
        sys.exit(0)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    traces = load_traces(args.input_dir, args.input_file, args.session_id)
    if not traces:
        logger.error("No traces found. Exiting.")
        sys.exit(1)
    logger.info("Loaded %d raw traces", len(traces))

    # ------------------------------------------------------------------
    # Initialise filters
    # ------------------------------------------------------------------
    anonymizer = Anonymizer(level=args.anonymize_level)
    quality_filter = QualityFilter(min_quality=args.min_quality)
    if args.semantic_dedup:
        diversity_filter: DiversityFilter | SemanticDiversityFilter = SemanticDiversityFilter(max_similarity=args.max_similarity)
    else:
        diversity_filter = DiversityFilter(max_similarity=args.max_similarity)

    # ------------------------------------------------------------------
    # Process — keep every trace, only deduplicate; report quality scores
    # ------------------------------------------------------------------
    kept: List[Dict[str, Any]] = []
    discarded_stats: Dict[str, int] = Counter()
    score_distribution: Dict[str, int] = Counter()
    error_distribution: Dict[str, int] = Counter()

    for idx, trace in enumerate(traces):
        trace_id = trace.get("session_id", trace.get("id", f"trace_{idx}"))

        # Anonymize (optional)
        if not args.skip_redact:
            use_llm = not args.no_llm_redact
            trace, redaction_count = anonymizer.redact_trace(trace, use_llm=use_llm)
            if redaction_count:
                logger.info("[%s] Redacted %d items", trace_id, redaction_count)

        # Quality (computed for every trace, never filtered out)
        score, breakdown, _passed = quality_filter.score(trace)
        logger.info("[%s] Quality score: %.2f", trace_id, score)

        # 5-factor error classification
        error_class = ErrorClassifier.classify(trace)
        if error_class != "none":
            logger.info("[%s] Error class: %s", trace_id, error_class)
        error_distribution[error_class] += 1

        # Bucket for distribution report
        bucket = f"{int(score * 10) / 10:.1f}-{int(score * 10) / 10 + 0.1:.1f}"
        score_distribution[bucket] += 1

        # Diversity — exact duplicates still dropped to avoid bloat
        if diversity_filter.is_duplicate(trace):
            discarded_stats["duplicate"] += 1
            discarded_stats["total"] += 1
            logger.debug("[%s] Discarded (duplicate)", trace_id)
            continue

        # Format
        axolotl = AxolotlFormatter.format(trace, score, trace_id, error_class=error_class)
        sharegpt = ShareGPTFormatter.format(trace)

        kept.append({
            "axolotl": axolotl,
            "sharegpt": sharegpt,
            "score": score,
            "breakdown": breakdown,
        })

    total_in = len(traces)
    total_kept = len(kept)
    duplicates = discarded_stats.get("duplicate", 0)
    logger.info(
        "Output %d / %d traces (%.1f%%); %d exact duplicates removed",
        total_kept,
        total_in,
        100 * total_kept / max(1, total_in),
        duplicates,
    )

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "data.jsonl", "w", encoding="utf-8") as f:
        for item in kept:
            f.write(json.dumps(item["axolotl"], ensure_ascii=False) + "\n")

    with open(out_dir / "sharegpt.jsonl", "w", encoding="utf-8") as f:
        for item in kept:
            f.write(json.dumps(item["sharegpt"], ensure_ascii=False) + "\n")

    # Error-masked training split: only traces with error_class == "none"
    if args.exclude_errors:
        clean = [item for item in kept if item["axolotl"].get("error_class", "none") == "none"]
        with open(out_dir / "data_clean.jsonl", "w", encoding="utf-8") as f:
            for item in clean:
                f.write(json.dumps(item["axolotl"], ensure_ascii=False) + "\n")
        logger.info("Wrote data_clean.jsonl with %d / %d error-free traces", len(clean), len(kept))

    # Stats
    avg_quality = sum(item["score"] for item in kept) / max(1, len(kept))
    stats = {
        "repo_name": args.repo_id.split("/")[-1],
        "repo_id": args.repo_id,
        "kept": total_kept,
        "discarded": discarded_stats["total"],
        "avg_quality": avg_quality,
        "score_distribution": dict(score_distribution),
        "error_distribution": dict(error_distribution),
    }

    # Dataset card
    example = kept[0]["axolotl"] if kept else {}
    card = DatasetCardBuilder.build(stats, example)
    (out_dir / "dataset_card.md").write_text(card, encoding="utf-8")

    # ------------------------------------------------------------------
    # Scenario export (for synthetic trace generation)
    # ------------------------------------------------------------------
    if args.export_scenarios:
        scenarios: List[Dict[str, Any]] = []
        seen_scenarios: set = set()
        for trace in traces:
            sc = ScenarioExtractor.extract(trace)
            if sc:
                h = hashlib.sha256(sc["scenario"].encode("utf-8")).hexdigest()
                if h not in seen_scenarios:
                    seen_scenarios.add(h)
                    scenarios.append(sc)
        if scenarios:
            ScenarioExtractor.to_jsonl(scenarios, out_dir / "scenarios.jsonl")
            ScenarioExtractor.to_markdown(scenarios, out_dir / "scenarios.md")
            logger.info(
                "Exported %d unique scenarios to scenarios.jsonl + scenarios.md",
                len(scenarios),
            )

    logger.info("Dataset written to %s", out_dir.resolve())

    # ------------------------------------------------------------------
    # HuggingFace
    # ------------------------------------------------------------------
    if args.push_to_hub:
        push_to_hub(out_dir, args.repo_id, private=not args.public)


if __name__ == "__main__":
    main()
