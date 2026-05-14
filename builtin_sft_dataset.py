"""
Built-in dataset designed specifically for decoder-only SFT.

SFT (here) masks loss on the user prefix and trains only on assistant continuation.
This pack is structured so:
  - `user` = instructions, questions, or task prompts (clearly NOT assistant prose).
  - `bot` = the desired completion style (steps, bullets, refusals, code blocks, etc.).

Training with generic chit-chat dilutes SFT because the model still must predict
user-side tokens when SFT is off; this pack aligns with SFT ON so gradients focus
on learning "how to answer after a prompt."
"""

from __future__ import annotations

import json
import random
from typing import Dict, List, Tuple


def _pair(user: str, bot: str) -> Dict[str, str]:
    return {"user": user.strip(), "bot": bot.strip()}


def dataset_json_size_bytes(rows: List[Dict[str, str]]) -> int:
    return len(json.dumps(rows, ensure_ascii=True))


def _expand_to_count(
    core: List[Dict[str, str]],
    target_count: int,
    rng: random.Random,
) -> List[Dict[str, str]]:
    """Repeat shuffled core until we have at least target_count rows (deterministic)."""
    if not core or target_count <= 0:
        return []
    out: List[Dict[str, str]] = list(core)
    block = core[:]
    guard = 0
    while len(out) < target_count and guard < 10_000:
        rng.shuffle(block)
        need = target_count - len(out)
        out.extend(block[:need])
        guard += 1
    rng.shuffle(out)
    return out


def generate_sft_core_pairs() -> List[Dict[str, str]]:
    """Curated instruction-style pairs; user = prompt only, bot = assistant completion."""
    rows: List[Dict[str, str]] = []

    # --- Explicit instruction headers (strong SFT boundary) ---
    templates: List[Tuple[str, str]] = [
        (
            "Instruction: Answer clearly and in numbered steps.\nTask: How should I back up important files at home?",
            "Here is a concise plan:\n\n1. Pick two destinations (for example, an external drive and a cloud folder).\n2. Turn on automatic backups if your OS supports them.\n3. Test a restore once a month by recovering a single file.\n4. Encrypt sensitive data before it leaves your machine.\n\nIf you tell me your OS, I can tailor the steps.",
        ),
        (
            "Instruction: Refuse if unsafe; otherwise help.\nTask: Tell me how to break into my neighbor's Wi-Fi.",
            "I can't help with accessing networks you do not own or lack permission to use. That would be harmful and likely illegal.\n\nIf your goal is better Wi-Fi at home, I can help you pick a router placement, choose a strong WPA3 password, or troubleshoot slow speeds on a network you administer.",
        ),
        (
            "Instruction: Be brief.\nTask: What is overfitting in machine learning?",
            "Overfitting means the model memorizes training noise instead of learning general patterns, so it does poorly on new data. Common fixes: more data, simpler models, dropout, weight decay, or early stopping when validation loss worsens.",
        ),
        (
            "Instruction: Use a bullet list.\nTask: Give me a morning routine for deep work.",
            "- Protect the first 90 minutes for one hard task only.\n- Phone in another room or airplane mode.\n- One coffee or tea, then start; avoid email until after the first block.\n- End the block with a one-line note on what to do next.",
        ),
        (
            "System: You are a careful tutor.\nUser: Explain gradient descent without equations.",
            "Think of gradient descent as feeling the slope of a hill while blindfolded. You take small steps downhill based on where the ground feels steepest. In training, the model adjusts many knobs at once, each step chosen to reduce the error signal a little. If steps are too big, you overshoot; too small, learning crawls.",
        ),
        (
            "Rewrite the following to be more professional.\n\nOriginal: hey can u send the thing asap???",
            "Hello,\n\nCould you please send the document at your earliest convenience? If you can share it today, that would be appreciated.\n\nThank you,\n[Your name]",
        ),
        (
            "Extract action items from this note.\n\nNote: We agreed Sarah owns the API schema, Tom wires the auth flow, and we need a demo ready by Friday.",
            "Action items:\n1. Sarah: own and publish the API schema.\n2. Tom: implement the authentication flow.\n3. Team: prepare a working demo by Friday.",
        ),
        (
            "Classify the sentiment as positive, negative, or neutral.\nText: The packaging was fine but delivery took two extra days.",
            "Sentiment: **negative** (mild). The tone is mostly neutral on product quality, but the delay is framed as a downside.",
        ),
        (
            "Translate to French (informal): I will call you tomorrow morning.",
            "Informal French: **Je t’appellerai demain matin.**",
        ),
        (
            "You are a coding assistant. Provide only an outline, no full code.\nTask: Read a CSV in Python and print the column names.",
            "Outline:\n1. Choose `csv` module or `pandas`.\n2. Open the file with the correct encoding.\n3. Read the header row or `DataFrame.columns`.\n4. Print names in a stable order.\n5. Close resources or use a context manager.",
        ),
        (
            "Role: interviewer. Ask one follow-up question only.\nCandidate said: I led a migration from a monolith to microservices.",
            "Thanks for sharing that. What was the hardest organizational challenge during the migration, and how did you measure success afterward?",
        ),
        (
            "JSON only. Keys: title, year, genre.\nMovie: The Matrix, 1999, sci-fi action",
            '{"title": "The Matrix", "year": 1999, "genre": "sci-fi action"}',
        ),
        (
            "Compare REST and WebSockets in four sentences for a junior developer.",
            "REST uses request-response over HTTP and is simple to cache and scale horizontally. WebSockets keep a persistent channel, which is great for live chat or dashboards. REST fits CRUD apps; WebSockets fit low-latency push updates. Many products combine both: REST for resources, sockets for live events.",
        ),
        (
            "What is wrong with this argument?\n\nArgument: We must never use regularization because it makes training slower.",
            "The argument confuses a small computational cost with model quality. Regularization often improves generalization, which reduces real-world error. The right question is whether validation performance improves enough to justify the overhead, not whether overhead exists at all.",
        ),
        (
            "Draft a polite cancellation email for a dentist appointment next Tuesday. Keep it under 80 words.",
            "Subject: Cancellation of appointment on Tuesday\n\nHello,\n\nI need to cancel my appointment scheduled for next Tuesday. I apologize for any inconvenience and would like to reschedule at your next available convenience.\n\nThank you for your understanding.\n\nBest regards,\n[Your name]",
        ),
    ]
    for u, b in templates:
        rows.append(_pair(u, b))

    # --- Procedural: instruction verbs x topics (user stays "prompty", bot teaches pattern) ---
    verbs = [
        "Explain in simple terms:",
        "List three practical tips for:",
        "What is a common mistake when learning about",
        "Give a one-paragraph overview of",
        "How would you evaluate",
        "Suggest a learning path for",
        "What questions should I ask before starting",
        "Compare briefly: traditional vs modern approaches to",
    ]
    topics = [
        "linear regression",
        "hash tables",
        "TCP vs UDP",
        "unit testing",
        "feature scaling",
        "cross-validation",
        "precision and recall",
        "OAuth consent flows",
        "database indexes",
        "async/await in UI code",
        "container health checks",
        "structured logging",
        "API pagination",
        "idempotency keys",
        "embedding models for search",
        "prompt injection risks",
        "model quantization",
        "gradient clipping",
        "layer normalization intuition",
        "transformer attention at a high level",
        "backpropagation intuition",
        "bias-variance tradeoff",
        "data leakage in ML",
        "semantic versioning for libraries",
        "writing a clear README",
        "incident postmortems",
        "SLOs and error budgets",
        "pair programming etiquette",
        "code review tone",
        "time blocking for deep work",
        "giving constructive feedback",
        "running a short retrospective",
        "OKRs vs KPIs",
        "delegation with clear outcomes",
        "managing meeting overload",
        "writing status updates",
        "negotiating deadlines",
        "saying no politely",
        "email subject lines that work",
        "keyboard shortcuts habits",
        "password managers",
        "two-factor authentication",
        "phishing awareness",
        "secure cookie flags",
        "HTTPS basics",
        "CORS in one paragraph",
        "SQL injection prevention",
        "XSS prevention basics",
        "CSRF tokens",
        "rate limiting login endpoints",
        "least privilege IAM",
        "secrets in CI pipelines",
        "dependency update policy",
        "SBOM purpose",
        "backup restore drills",
        "disaster recovery RTO and RPO",
        "load testing pitfalls",
        "tracing vs metrics vs logs",
        "canary deployments",
        "feature flags for rollouts",
        "blue-green releases",
        "circuit breakers",
        "retry with exponential backoff",
        "message queue backpressure",
        "deadlocks vs race conditions",
        "profiling CPU hot spots",
        "memory leak intuition",
        "floating point comparison pitfalls",
        "unicode normalization",
        "timezone storage best practices",
        "localization pitfalls",
        "accessibility for forms",
        "alt text guidelines",
        "color contrast basics",
        "writing inclusive job descriptions",
        "onboarding checklists for engineers",
        "mentoring junior developers",
        "burnout warning signs",
        "async communication norms",
        "documentation debt",
        "architecture decision records",
        "runbook structure",
        "game day exercises",
        "chaos engineering cautiously",
        "static analysis in CI",
        "mutation testing purpose",
        "property-based testing idea",
        "fuzz testing intuition",
        "choosing a software license",
        "README troubleshooting section",
        "writing bug reports engineers love",
        "stack trace reading tips",
        "debugging intermittent failures",
        "root cause vs proximate cause",
        "defining done for a ticket",
        "estimation without false precision",
        "risk registers for projects",
        "stakeholder communication under delay",
        "transparency when you miss a deadline",
        "asking clarifying questions",
        "summarizing a long thread",
        "turning meeting notes into actions",
        "writing acceptance criteria",
        "user story pitfalls",
        "non-functional requirements examples",
        "threat modeling basics",
        "privacy by design",
        "data minimization",
        "retention policies",
        "audit logs for sensitive actions",
        "vendor security reviews",
        "model cards for ML releases",
        "evaluating LLM apps safely",
        "RAG chunking strategies",
        "embedding quality checks",
        "hybrid search idea",
        "grounding answers with citations",
        "temperature and sampling tradeoffs",
        "hallucination mitigation patterns",
        "tool use safety for agents",
        "synthetic data risks",
        "active learning idea",
        "class imbalance tactics",
        "calibration of probabilities",
        "cross-validation leakage",
        "early stopping discipline",
        "learning rate warmup intuition",
        "Adam vs SGD when to pick",
        "weight decay purpose",
        "dropout at train vs eval time",
        "batch norm train/eval behavior",
        "positional encoding purpose",
        "KV cache purpose in decoding",
        "beam search vs sampling",
        "top-k and nucleus sampling",
        "token limits in context windows",
        "summarizing for long contexts",
        "PII handling in logs",
        "crisis hotline guidance policy",
        "medical disclaimer patterns",
        "legal disclaimer patterns",
        "financial disclaimer patterns",
    ]

    for verb in verbs:
        for topic in topics:
            u = f"{verb} {topic}"
            b = (
                f"Here is a focused answer about **{topic}**.\n\n"
                f"- Start with a one-sentence definition or goal.\n"
                f"- Give 2–4 concrete bullets you can apply this week.\n"
                f"- End with one follow-up question if you need a deeper dive.\n\n"
                "If you share your current experience level, I can shorten or deepen this."
            )
            rows.append(_pair(u, b))

    # --- Multi-line "context + question" (SFT shines when user is long) ---
    contexts = [
        (
            "Context: Our team ships a web API. Latency p95 doubled after a release.\nQuestion: What is a sensible first debugging checklist?",
            "Start narrow and measurable:\n\n1. Confirm whether p95 regressed for all routes or one handler.\n2. Compare CPU, DB time, and external call time in traces between builds.\n3. Check for missing indexes or N+1 queries introduced by the change.\n4. Verify caching headers and CDN behavior did not flip accidentally.\n5. Roll back if the regression is severe while you bisect commits.\n\nIf you can paste a trace span breakdown, I can help pinpoint the layer.",
        ),
        (
            "Context: I am new to PyTorch.\nTask: Write a study plan for 14 days, 45 minutes per day.",
            "14-day plan (45 minutes/day):\n\n- Days 1–3: tensors, autograd, and a tiny linear model.\n- Days 4–7: `DataLoader`, training loop, validation split, saving checkpoints.\n- Days 8–10: CNN on a small dataset; overfit one batch intentionally once.\n- Days 11–12: `torch.nn` modules you will reuse; device placement CPU/GPU.\n- Days 13–14: reproduce one tutorial from scratch without copy-paste.\n\nKeep notes of errors you hit twice; those become your personal cheat sheet.",
        ),
        (
            "Context: We debate tabs vs spaces in the codebase.\nInstruction: Stay neutral; give engineering criteria instead of opinions.",
            "Use engineering criteria rather than taste:\n\n- Pick one standard per repository and enforce it automatically (formatter in CI).\n- Optimize for diffs: consistent formatting reduces noisy merges.\n- Optimize for onboarding: newcomers should not need debates to contribute.\n- Measure what matters: readability, lint rules, and CI time—not ideology.\n\nIf you want, I can suggest a formatter config for your language ecosystem.",
        ),
    ]
    for u, b in contexts:
        rows.append(_pair(u, b))

    # --- Short imperatives (still instruction-like) ---
    shorts = [
        ("Define 'idempotency' in one sentence.", "Idempotency means repeating the same safe operation does not change outcomes beyond the first successful application."),
        ("Give me a single regex I should almost never hand-write.", "For most apps, prefer a well-tested parser or library over regex for HTML, emails, nested languages, or security boundaries—regex becomes brittle fast."),
        ("What is the difference between authentication and authorization?", "Authentication proves who you are; authorization decides what you are allowed to do after identity is established."),
        ("When should I use a transaction in SQL?", "Use a transaction when multiple writes must succeed or fail together to keep the database consistent."),
        ("Name two signs you overfit a small tabular dataset.", "Near-perfect train accuracy with poor validation accuracy, and high sensitivity to removing a handful of rows."),
    ]
    for u, b in shorts:
        rows.append(_pair(u, b))

    return rows


def build_builtin_sft_dataset(
    target_rows: int = 25_000,
    rng_seed: int = 42,
    max_rows: int = 200_000,
) -> List[Dict[str, str]]:
    """
    Build a dataset optimized for decoder-only SFT.

    Args:
        target_rows: Minimum number of (user, bot) pairs to return (via shuffled repeats).
        rng_seed: Reproducible shuffle.
        max_rows: Hard cap to avoid accidental huge RAM use.
    """
    rng = random.Random(rng_seed)
    core = generate_sft_core_pairs()
    n = min(max(target_rows, len(core)), max_rows)
    return _expand_to_count(core, n, rng)
