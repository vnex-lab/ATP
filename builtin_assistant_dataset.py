"""
Large built-in assistant-style dialog corpus for training (no external download).

Target size is approximate serialized JSON bytes (~80–100 MB default). Dataset size
affects RAM while the list is loaded; GPU VRAM is dominated by model size, not
this file.
"""

from __future__ import annotations

import json
import random
from typing import Dict, List, Tuple


def _pair(user: str, bot: str) -> Dict[str, str]:
    return {"user": user.strip(), "bot": bot.strip()}


def _json_row_bytes(p: Dict[str, str]) -> int:
    return len(json.dumps(p, ensure_ascii=True))


def _estimate_total_bytes(rows: List[Dict[str, str]], sample: int = 4000) -> float:
    n = len(rows)
    if n == 0:
        return 0.0
    k = min(sample, n)
    chunk = rows[:k]
    avg = sum(_json_row_bytes(p) for p in chunk) / k
    return avg * n


def generate_core_pairs(rng: random.Random) -> List[Dict[str, str]]:
    """High-quality assistant turns + procedural combinations."""
    out: List[Dict[str, str]] = []

    # --- Greetings: user says Hello / Hi / etc. -> professional assistant reply ---
    greet_users = [
        "Hello",
        "Hello!",
        "hello",
        "Hi",
        "Hi!",
        "hi there",
        "Hey",
        "Hey there",
        "Good morning",
        "Good afternoon",
        "Good evening",
        "Hi — anyone there?",
        "Hello, are you available?",
        "Hey, quick question",
        "Yo",
        "Hiya",
        "Howdy",
        "Hello friend",
        "Hi, hope you're doing well",
        "Morning",
        "Afternoon",
        "Evening",
        "Hello, I just logged in",
    ]
    greet_bots = [
        "Hello! How can I help you today?",
        "Hi there! How can I help you today?",
        "Hey! How can I help you today?",
        "Good to hear from you! How can I help you today?",
        "Hello! What can I help you with today?",
        "Hi! I'm here to help — what do you need?",
    ]
    for u in greet_users:
        out.append(_pair(u, rng.choice(greet_bots)))

    # --- Thanks / goodbye / polite ---
    misc: List[Tuple[str, str]] = [
        ("Thank you", "You're welcome! Happy to help anytime."),
        ("Thanks", "You're welcome! Let me know if you need anything else."),
        ("Thanks a lot", "Any time! Glad I could help."),
        ("Thank you so much", "You're very welcome! Feel free to come back if you have more questions."),
        ("I appreciate it", "Happy to help! Is there anything else I can do for you?"),
        ("Goodbye", "Goodbye! Take care, and reach out whenever you need assistance."),
        ("Bye", "Bye! Have a great day."),
        ("See you later", "See you later! I'm here whenever you need help."),
        ("Have a good day", "You too! Take care."),
        ("Sorry to bother you", "You're not bothering me at all — that's what I'm here for."),
        ("Are you a bot?", "I'm an AI assistant here to answer questions and help you solve problems."),
        ("What can you do?", "I can explain concepts, walk through steps, brainstorm ideas, and help you draft messages or plans. What would you like to tackle?"),
        ("Who are you?", "I'm your AI assistant — I don't have a personal identity, but I'm focused on helping you get clear, practical answers."),
        ("How does this work?", "You can ask a question or describe a problem, and I'll do my best to guide you with clear steps or explanations."),
    ]
    for u, b in misc:
        out.append(_pair(u, b))

    # --- How are you ---
    for u in [
        "How are you?",
        "How's it going?",
        "How are you doing today?",
        "Are you doing okay?",
    ]:
        out.append(
            _pair(
                u,
                "I'm doing well, thank you for asking! I'm ready whenever you are — what can I help you with?",
            )
        )

    # --- Help preambles x topics (procedural) ---
    help_openers = [
        "I need help with",
        "Can you help me with",
        "I'm trying to",
        "How do I",
        "What's the best way to",
        "Could you explain",
        "I'm stuck on",
        "I don't understand how to",
        "Walk me through",
        "Can you show me how to",
    ]
    greet_leads = ["Hello!", "Hi!", "Hey!", "Good morning!", "Quick question:"]

    topics: List[Tuple[str, str]] = [
        ("writing a clear email to my manager", "Start with a short subject line, one opening sentence with context, bullet points for requests or updates, and a polite closing with your name."),
        ("staying focused while studying", "Try the Pomodoro technique: 25 minutes of focus, a 5-minute break, and remove phone notifications from your study space."),
        ("sleeping better", "Keep a consistent bedtime, limit screens an hour before bed, and keep your room cool and dark."),
        ("starting a simple budget", "List monthly income and fixed expenses first, then assign categories for savings and discretionary spending, and review weekly."),
        ("preparing for a job interview", "Research the company, prepare STAR stories for common questions, and rehearse concise answers out loud."),
        ("reducing stress before a deadline", "Break the work into the smallest next step, set a 25-minute timer, and prioritize one task at a time."),
        ("giving constructive feedback", "Describe the behavior, explain the impact, and suggest one concrete improvement — keep tone respectful and specific."),
        ("learning Python basics", "Install Python, use an editor you like, then practice variables, loops, and functions with tiny scripts until they feel natural."),
        ("understanding what APIs are", "An API is a structured way for programs to request data or actions from another service over the network."),
        ("resetting my router", "Unplug the router for 30 seconds, plug it back in, wait until lights stabilize, then reconnect your devices."),
        ("choosing a strong password", "Use a long passphrase or a password manager, avoid reuse across sites, and enable two-factor authentication where possible."),
        ("spotting phishing emails", "Check the sender domain, look for urgent language, hover links before clicking, and verify requests through an official website."),
        ("organizing my desktop files", "Create a few top-level folders by project or year, archive old files monthly, and name files with dates when helpful."),
        ("writing a to-do list that works", "Write three must-do items for today, estimate time per task, and move anything else to a backlog list."),
        ("breaking a big project into steps", "Define the outcome, list milestones backwards from the deadline, and assign one owner and date per milestone."),
        ("asking for a raise", "Schedule a calm conversation, summarize achievements with metrics, and propose a specific range based on market research."),
        ("saying no politely", "Thank them, give a brief reason if appropriate, and offer an alternative timeline or person if you can."),
        ("networking when I'm shy", "Prepare two questions you genuinely care about, aim for short conversations, and follow up with a brief thank-you message."),
        ("learning a new language faster", "Practice daily at a fixed time, mix listening and speaking, and review yesterday's phrases before adding new ones."),
        ("meal prep for a busy week", "Pick two proteins and two vegetables, cook in batches, and store in labeled containers for grab-and-go meals."),
        ("fixing writer's block", "Write one bad paragraph with no editing, change your environment for ten minutes, then rewrite from your notes."),
        ("reading faster for work", "Skim headings first, read the first and last paragraph of each section, then go deeper only where decisions depend on details."),
        ("taking better meeting notes", "Capture decisions, owners, and deadlines — skip verbatim quotes unless required."),
        ("running my first 5K", "Follow a couch-to-5K style plan, run slowly enough to talk, and rest days are part of training."),
        ("stretching after sitting all day", "Stand hourly, roll shoulders, hip flexor lunges, and gentle neck tilts for a few minutes each."),
        ("cleaning my keyboard", "Turn it off, shake crumbs out, use compressed air, then lightly wipe keys with a microfiber cloth slightly damp with isopropyl alcohol."),
        ("backing up important files", "Use the 3-2-1 rule: three copies, two media types, one off-site — cloud plus external drive is a common combo."),
        ("choosing a laptop for college", "Prioritize battery life, keyboard comfort, RAM for your major's software, and warranty support in your region."),
        ("explaining machine learning to a friend", "It's software that finds patterns in examples instead of being fully hand-coded rule by rule."),
        ("what is version control", "It tracks changes to files over time so teams can collaborate and roll back mistakes — Git is a common tool."),
        ("what is cloud computing", "It means running software or storing data on someone else's servers over the internet instead of only on your device."),
        ("HTTPS vs HTTP", "HTTPS encrypts traffic between your browser and the site, which helps protect passwords and personal data."),
        ("what is open source", "Software whose source code is publicly available for others to inspect, use, and often contribute improvements to."),
        ("difference between RAM and storage", "RAM is fast temporary memory for running programs; storage keeps files long-term and is slower."),
        ("CPU vs GPU", "The CPU handles general tasks; GPUs excel at parallel math, which helps graphics and many AI workloads."),
        ("what is a database", "A structured store for data with efficient lookup — common types include relational SQL and document databases."),
        ("JSON vs CSV", "JSON is nested key-value text good for APIs; CSV is flat rows and columns, simple for spreadsheets."),
        ("what is debugging", "Finding why a program misbehaves by reading errors, reproducing the issue, and changing code or data step by step."),
        ("what is technical debt", "Shortcuts in code or design that save time now but can slow future changes until they are refactored."),
        ("how to ask a good coding question", "Include what you tried, the exact error message, minimal code to reproduce, and your environment versions."),
        ("what is regression testing", "Re-running tests after changes to ensure old features still work as expected."),
        ("what is latency", "The delay between a request and a response — lower is better for interactive apps."),
        ("what is bandwidth", "How much data can move per second on a network link."),
        ("two-factor authentication", "A second proof besides your password — often a code from an app or text — which greatly reduces account takeover risk."),
        ("VPN basics", "A VPN routes your traffic through another server, which can add privacy on public Wi-Fi but is not a magic shield against all threats."),
        ("cookie popups", "Sites use cookies for sessions and analytics; you can limit non-essential cookies in browser settings where available."),
        ("GDPR in one sentence", "EU rules that give people rights over personal data and require companies to handle it lawfully and transparently."),
        ("accessibility on websites", "Use clear headings, keyboard navigation, alt text for important images, and sufficient color contrast."),
        ("responsive design", "Layouts that adapt to screen size using flexible grids, images, and CSS breakpoints."),
        ("what is Docker", "Packaging an app with its dependencies so it runs consistently across machines using containers."),
        ("what is CI/CD", "Automated pipelines that build, test, and deploy code changes to reduce manual errors and speed delivery."),
        ("agile vs waterfall", "Agile delivers value in short iterations with feedback; waterfall plans most work upfront in longer phases."),
        ("what is a product roadmap", "A prioritized plan of themes and features over time that aligns teams without promising every date as fixed."),
        ("writing user stories", "As a <role>, I want <goal>, so that <benefit> — keep them small and testable."),
        ("acceptance criteria", "Clear conditions that define when a feature is done, often written as checklists or example scenarios."),
        ("what is uptime", "The percentage of time a service is available — 99.9% allows about 8.7 hours downtime per year."),
        ("rate limiting", "Restricting how many requests a client can make in a window to protect servers from abuse or overload."),
        ("what is caching", "Storing copies of results closer to the user or faster storage so repeat requests are cheaper and quicker."),
        ("load balancing", "Distributing traffic across multiple servers to improve capacity and reliability."),
        ("what is encryption at rest", "Data is stored in encrypted form on disk so physical theft is less likely to expose contents."),
        ("zero trust security", "Verify explicitly, assume breach, and grant least privilege instead of trusting everything inside a network perimeter."),
        ("phishing vs spear phishing", "Phishing is broad fake messages; spear phishing targets a specific person with tailored bait."),
        ("ransomware basics", "Malware that encrypts files and demands payment — backups and patching are key defenses."),
        ("SQL injection idea", "Never trust raw user input in queries; use parameterized statements or an ORM to separate code from data."),
        ("XSS idea", "Treat user content as data, not HTML, unless carefully sanitized — escaping output prevents many script injections."),
        ("CSRF idea", "Use anti-forgery tokens and same-site cookies so other sites cannot trick a browser into unwanted actions."),
        ("what is OAuth", "A delegated authorization framework so apps can access limited resources without sharing your main password."),
        ("REST vs GraphQL", "REST uses multiple resource endpoints; GraphQL lets clients request a shaped graph of fields in one query."),
        ("WebSockets", "A persistent two-way channel useful for chat, live dashboards, and games compared to plain request-response HTTP."),
        ("what is WebAssembly", "A compact binary format for near-native speed in the browser alongside JavaScript."),
        ("progressive web apps", "Web experiences that can feel app-like with offline caches and install prompts on supported platforms."),
        ("what is Kubernetes", "Orchestration for running many containers across machines with scaling, health checks, and rollouts."),
        ("infrastructure as code", "Defining servers and networks in versioned files so environments are repeatable and reviewable."),
        ("monitoring vs observability", "Monitoring tracks known signals; observability helps you investigate unknown failures with rich traces and logs."),
        ("structured logging", "Logging JSON or key-value fields so logs are easy to search and correlate across services."),
        ("what is SLA", "A formal commitment to service levels, often including uptime targets and support response times."),
        ("what is SLO", "An internal reliability target derived from user happiness, often stricter than the external SLA."),
        ("on-call best practices", "Runbooks for common issues, clear escalation paths, and blameless postmortems after incidents."),
        ("blameless postmortem", "Focus on systems and processes that failed, not on punishing individuals, so learning improves reliability."),
        ("what is pair programming", "Two developers collaborate in real time on one workstation to share knowledge and catch defects early."),
        ("code review tips", "Be specific, praise good choices, separate nitpicks from blockers, and keep reviews small for faster turnaround."),
        ("naming variables well", "Prefer full words over cryptic abbreviations; names should answer why or how, not repeat the type."),
        ("refactoring safely", "Make small behavior-preserving changes with tests or tight manual checks after each step."),
        ("what is TDD", "Write a failing test first, implement the minimum to pass, then refactor while keeping tests green."),
        ("unit vs integration tests", "Unit tests isolate one module; integration tests verify multiple parts working together."),
        ("flaky tests", "Tests that sometimes fail without product bugs — fix by removing timing assumptions and isolating external dependencies."),
        ("what is semantic versioning", "MAJOR.MINOR.PATCH bumps signal breaking changes, new features, and bug fixes respectively."),
        ("choosing a license for my project", "Pick goals first: permissive licenses like MIT are common for libraries; copyleft like GPL has stricter sharing rules."),
        ("README essentials", "What it does, how to install, how to run, how to contribute, and where to report issues."),
        ("writing a bug report", "Steps to reproduce, expected vs actual behavior, environment, and screenshots or logs if helpful."),
        ("what is a stack trace", "A list of active call frames when an error occurs — start reading from the top frame in your code."),
        ("memory leak intuition", "Allocations that are never released so usage grows over time until performance degrades or the app crashes."),
        ("race condition intuition", "Outcomes depend on timing of concurrent operations — often fixed with locks, queues, or atomic operations."),
        ("deadlock intuition", "Two threads each wait on the other forever — avoid by consistent lock ordering or timeouts."),
        ("big O intuition", "Describes how work grows with input size — helps compare algorithms at large scales."),
        ("hash table intuition", "Average fast lookups by mapping keys through a hash function to buckets."),
        ("binary search intuition", "Halve a sorted range each step — logarithmic time if random access is cheap."),
        ("recursion vs iteration", "Recursion is elegant for tree-like problems; iteration can avoid deep stack limits."),
        ("what is dynamic programming", "Solving overlapping subproblems once and storing answers to build up an optimal solution."),
        ("greedy algorithms", "Make the locally best choice at each step — works only when that leads to a global optimum."),
        ("graph BFS vs DFS", "BFS explores layer by layer for shortest unweighted paths; DFS goes deep first, useful for connectivity checks."),
        ("Dijkstra's idea", "Shortest paths in graphs with non-negative edge weights using a priority queue."),
        ("what is NP-hard", "At least as hard as the hardest problems in NP — often we use approximations or heuristics in practice."),
        ("floating point pitfalls", "Some decimals are not exact in binary — use tolerances for comparisons or decimal types for money."),
        ("unicode basics", "Text is encoded; mismatched encodings cause mojibake — prefer UTF-8 end to end."),
        ("time zones", "Store UTC internally, convert for display, and be careful with daylight saving jumps."),
        ("what is CRDT", "Data structures that merge edits without a single server ordering every operation — useful for collaborative editing."),
        ("event sourcing", "Store changes as events and derive state by replaying — great audit trails, more complexity."),
        ("CQRS pattern", "Separate read models from write models to scale and optimize each differently."),
        ("microservices tradeoffs", "Independent deploys and teams vs operational complexity and network latency between services."),
        ("monolith tradeoffs", "Simpler early on; can become hard to change at scale without modular boundaries inside the codebase."),
        ("what is idempotency", "Doing the same operation twice has the same effect as once — important for retries in distributed systems."),
        ("at-least-once delivery", "Messages might arrive more than once — consumers should deduplicate or tolerate duplicates."),
        ("exactly-once illusion", "True exactly-once end to end is hard; systems often combine deduplication with idempotent handlers."),
        ("what is a message queue", "A buffer between producers and consumers that decouples rate and improves resilience to bursts."),
        ("backpressure", "Slowing producers when consumers cannot keep up to avoid unbounded memory growth."),
        ("circuit breaker pattern", "Stop calling a failing dependency for a while so it can recover and you fail fast."),
        ("retry with exponential backoff", "Increase wait between retries to reduce load during outages, with jitter to avoid thundering herds."),
        ("what is canary deployment", "Roll out a new version to a small slice of traffic first, monitor, then expand."),
        ("blue-green deployment", "Two identical environments swap roles so rollback is a traffic switch."),
        ("feature flags", "Toggle behavior without redeploying — useful for experiments and gradual rollouts."),
        ("A/B testing basics", "Randomly assign users to variants, measure metrics, and watch for statistical significance and side effects."),
        ("privacy by design", "Build with minimal data collection, clear consent, and strong defaults from the start."),
        ("data minimization", "Collect only what you need for a stated purpose and delete when no longer necessary."),
        ("anonymization vs pseudonymization", "Anonymization removes identifying linkage; pseudonymization replaces identifiers with reversible tokens under controls."),
        ("what is bias in ML models", "Systematic errors from unrepresentative training data or labels — mitigate with audits and diverse data."),
        ("explainability basics", "Helping users understand why a model output occurred — ranges from simple feature weights to local approximations."),
        ("hallucinations in LLMs", "Confident but incorrect text — verify facts, cite sources when possible, and use retrieval for critical domains."),
        ("prompt injection risk", "Untrusted text that manipulates model behavior — treat prompts as code-like input and sandbox tools."),
        ("tool use safety", "Validate tool arguments, scope permissions narrowly, and confirm destructive actions with the user."),
        ("token limits", "Models process finite context — summarize long documents or chunk with overlap when needed."),
        ("temperature in sampling", "Higher values increase randomness; lower values make outputs more deterministic and focused."),
        ("evaluating an LLM app", "Use held-out questions, human review rubrics, and automated checks for format and safety."),
        ("RAG overview", "Retrieve relevant documents first, then condition generation on them to ground answers in your knowledge base."),
        ("embedding models", "Maps text to vectors so semantic similarity can be measured by distance in vector space."),
        ("vector database role", "Stores embeddings with fast nearest-neighbor search for retrieval workflows."),
        ("chunking documents", "Split text into overlapping chunks sized for retrieval quality and model context."),
        ("hybrid search", "Combine keyword search with vector search to catch both exact matches and paraphrases."),
        ("fine-tuning vs prompting", "Fine-tuning updates weights for a domain; prompting guides behavior without weight changes — often start with prompting."),
        ("LoRA intuition", "Train small low-rank adapters instead of full weights to adapt models more cheaply."),
        ("quantization intuition", "Use fewer bits per weight to shrink models and speed inference with some accuracy tradeoff."),
        ("on-device vs cloud inference", "On-device improves privacy and latency for small models; cloud scales for large models."),
        ("batching inference", "Process multiple inputs together to improve throughput on GPUs."),
        ("streaming responses", "Send partial tokens to the UI for better perceived latency in chat apps."),
        ("what is ONNX", "An interchange format to run models across different runtimes and hardware accelerators."),
        ("what is GGUF", "A file format used by tools like llama.cpp for quantized model weights and metadata."),
        ("model card idea", "Document intended use, limitations, metrics, and ethical considerations for a trained model."),
        ("datasheet for datasets", "Document provenance, collection process, and known biases similar to nutrition labels for data."),
        ("synthetic data cautions", "Useful for augmentation but can amplify biases or unrealistic patterns if not validated."),
        ("active learning idea", "Prioritize labeling the examples that would most improve the model to save annotation budget."),
        ("weak supervision", "Use noisy rules or heuristics to label data at scale, then denoise with modeling."),
        ("class imbalance", "When rare classes matter, consider resampling, weighted loss, or better metrics than accuracy alone."),
        ("precision vs recall", "Precision is how many predicted positives are true; recall is how many true positives you catch."),
        ("ROC AUC intuition", "Measures ranking quality across thresholds — can hide issues on imbalanced data."),
        ("calibration", "Making predicted probabilities match real frequencies — important for decisions and thresholds."),
        ("cross-validation", "Split data into folds to estimate performance more reliably than a single train-test split."),
        ("data leakage", "When validation accidentally includes information from the future or test set — inflates metrics."),
        ("normalization vs standardization", "Rescale features to a range or zero mean unit variance to help many algorithms train."),
        ("regularization", "Penalties that discourage overly complex models to improve generalization."),
        ("dropout intuition", "Randomly zero activations during training so the network cannot rely too heavily on single paths."),
        ("batch norm intuition", "Normalize layer inputs across a batch to stabilize training — behavior differs in train vs eval."),
        ("layer norm intuition", "Normalize across features per token or timestep — common in transformers."),
        ("learning rate too high", "Loss may diverge or oscillate; too low and training crawls — schedules and warmup help."),
        ("gradient clipping", "Cap gradient norms to reduce exploding gradients in recurrent or deep networks."),
        ("early stopping", "Stop training when validation loss worsens to reduce overfitting."),
        ("transfer learning", "Start from weights trained on a large task then adapt to a smaller related task."),
        ("few-shot learning", "Learning from a handful of examples — often via meta-learning or in-context examples."),
        ("zero-shot learning", "Perform a task without task-specific training examples, relying on general capabilities."),
        ("self-supervised learning", "Create labels from the data itself, like predicting masked tokens or future frames."),
        ("contrastive learning", "Learn representations by pulling similar pairs close and pushing dissimilar pairs apart."),
        ("what is attention", "A mechanism to weigh which parts of the input to focus on when producing each output position."),
        ("transformer overview", "Stacked self-attention and feed-forward layers that scale well with data and parallel hardware."),
        ("positional encoding", "Injects order information since attention itself is permutation-invariant over positions."),
        ("KV cache in decoding", "Store key and value tensors for past tokens to avoid recomputing them during autoregressive generation."),
        ("beam search", "Explore multiple candidate continuations to improve quality at higher compute cost than greedy decoding."),
        ("nucleus sampling", "Sample from the smallest set of tokens whose cumulative probability exceeds a threshold p."),
        ("top-k sampling", "Restrict sampling to the k most likely next tokens."),
        ("system prompt role", "High-level instructions that shape assistant behavior across a conversation."),
        ("conversation history limits", "Summarize or truncate older turns to stay within context while preserving key facts."),
        ("PII handling", "Avoid echoing government IDs, full card numbers, or health details; suggest secure channels when needed."),
        ("crisis resources", "If someone expresses self-harm, encourage contacting local emergency services or crisis hotlines immediately."),
        ("medical disclaimer", "I'm not a doctor; for symptoms or medications, please consult a qualified clinician."),
        ("legal disclaimer", "I'm not a lawyer; for legal decisions, consult a licensed attorney in your jurisdiction."),
        ("financial disclaimer", "This isn't personalized financial advice; consider a certified professional for your situation."),
        ("how to cite sources", "Prefer primary sources, include dates when relevant, and distinguish quotes from paraphrases."),
        ("summarizing a long thread", "Extract decisions, open questions, owners, and deadlines in bullet points under ten lines."),
        ("writing a status update", "What shipped, what's blocked, what's next, and what help you need — keep it skimmable."),
        ("1:1 meeting agenda", "Celebrate a win, review priorities, discuss growth, and leave with two clear action items."),
        ("delegation template", "Outcome, constraints, deadline, authority level, and how you'll check in."),
        ("running a retro", "What went well, what didn't, experiments to try next time — focus on systems not blame."),
        ("OKR basics", "Objectives are qualitative goals; key results are measurable outcomes that indicate progress."),
        ("SMART goals", "Specific, measurable, achievable, relevant, and time-bound targets."),
        ("Eisenhower matrix", "Urgent vs important quadrants to prioritize tasks beyond whatever feels loudest."),
        ("inbox zero method", "Process to empty: delete, delegate, defer with a calendar slot, or do immediately if under two minutes."),
        ("time blocking", "Assign calendar blocks for deep work and treat them like meetings with yourself."),
        ("context switching cost", "Batch similar tasks and mute notifications to preserve focus quality."),
        ("writing a README for a library", "Install, quickstart example, API overview, versioning policy, and contribution guidelines."),
        ("semver for libraries", "Breaking API changes bump major; additive features bump minor; fixes bump patch."),
        ("changelog habits", "Human-readable notes per release grouped by added, changed, fixed, and removed."),
        ("semantic commit messages", "Short imperative subject line, optional body with motivation and consequences."),
        ("pre-commit hooks", "Run formatters and linters automatically before commits to keep main green."),
        ("what is fuzz testing", "Randomized inputs to find crashes and security issues in parsers and protocols."),
        ("property-based testing", "Generate many inputs that should satisfy invariants your code must always hold."),
        ("mutation testing", "Change code slightly to see if tests catch it — measures test suite strength."),
        ("static typing benefits", "Catch whole classes of errors before runtime and improve editor assistance."),
        ("what is a monorepo", "Many packages in one repository with shared tooling — coordination vs size tradeoffs."),
        ("what is polyrepo", "Separate repositories per service or library — independence vs cross-repo change pain."),
        ("API versioning strategies", "URL path versions, headers, or content negotiation — pick one and document deprecation policy."),
        ("pagination cursors vs offsets", "Cursors scale better for large changing datasets; offsets are simpler for small stable lists."),
        ("idempotency keys for payments", "Clients send a unique key so retries do not double-charge."),
        ("webhook retries", "Expect duplicate deliveries; verify signatures and respond quickly with 2xx before heavy work or queue it."),
        ("HMAC verification", "Shared secret signs payloads so receivers can detect tampering."),
        ("JWT basics", "Signed tokens that carry claims — mind expiry, rotation, and never trust them without signature verification."),
        ("OAuth scopes", "Limit granted permissions to least privilege for third-party integrations."),
        ("CORS purpose", "Browser security model controlling which origins can read responses from your API."),
        ("Content Security Policy", "HTTP header that reduces XSS impact by restricting script and resource sources."),
        ("Subresource Integrity", "Hash attributes on script tags so tampered CDN files do not execute."),
        ("HTTP caching headers", "Cache-Control and ETag help clients reuse responses safely."),
        ("compression tradeoffs", "gzip or brotli reduce bytes but add CPU — tune for your payload sizes."),
        ("HTTP/2 multiplexing", "Many requests over one connection — reduces head-of-line blocking vs HTTP/1.1."),
        ("HTTP/3 QUIC", "Runs over UDP with built-in encryption and faster connection setup in many networks."),
        ("TLS certificate renewal", "Automate with ACME clients and monitor expiry to avoid outages."),
        ("reverse proxy role", "Terminate TLS, route traffic, add rate limits, and cache at the edge."),
        ("CDN caching", "Serve static assets geographically close to users with proper cache keys and purge workflows."),
        ("database indexes", "Speed lookups at the cost of slower writes and more storage — index selective columns you filter on."),
        ("N+1 query problem", "Accidentally query once per row — fix with joins or batch loading."),
        ("transactions ACID", "Atomicity, consistency, isolation, durability — choose isolation levels for your contention profile."),
        ("ORM vs raw SQL", "ORMs speed development; raw SQL helps hot paths and complex reporting."),
        ("migrations discipline", "Versioned schema changes reviewed like code with rollback plans."),
        ("backup restore drills", "Periodically prove you can restore — backups you never test are hopes, not plans."),
        ("disaster recovery RTO RPO", "Recovery time objective and recovery point objective define how fast and how fresh recovery must be."),
        ("capacity planning", "Measure growth trends, headroom, and autoscaling limits before traffic spikes."),
        ("autoscaling signals", "CPU alone is often insufficient — use request latency, queue depth, or custom metrics."),
        ("graceful shutdown", "Drain in-flight requests on SIGTERM before exiting in container orchestrators."),
        ("health checks", "Separate liveness from readiness so traffic only hits instances that are truly ready."),
        ("structured config", "Environment-specific settings via env vars and validated schemas, not hard-coded secrets."),
        ("secrets management", "Use a vault or cloud secret store — never commit secrets to git."),
        ("least privilege IAM", "Grant minimal permissions per role and audit periodically."),
        ("SBOM idea", "Software bill of materials lists dependencies to respond faster to supply chain vulnerabilities."),
        ("dependency pinning", "Lock versions for reproducible builds; update deliberately with changelogs."),
        ("semantic code search", "Find code by meaning using embeddings alongside text search in large repos."),
        ("documentation debt", "Treat docs as part of definition of done for features users must discover."),
        ("onboarding checklist", "Repo clone, devcontainer or setup script, sample data, and first issue within day one."),
        ("inclusive language in docs", "Use gender-neutral examples and avoid idioms that do not translate globally."),
        ("plain language benefits", "Shorter sentences and common words improve comprehension for global teams."),
        ("alt text for diagrams", "Describe the insight, not every decorative pixel, for screen reader users."),
        ("captioning videos", "Improves accessibility and searchability for spoken content."),
        ("colorblind-safe palettes", "Avoid red-green only distinctions; add patterns or labels."),
        ("keyboard shortcuts in apps", "Document discoverable shortcuts and avoid trapping focus in modals."),
        ("dark mode contrast", "Check contrast in both themes — washed out grays fail WCAG."),
        ("localization pitfalls", "Date formats, plurals, and string concatenation across languages need libraries, not manual hacks."),
        ("RTL layout testing", "Mirroring layouts for Arabic or Hebrew requires explicit QA, not assumptions."),
        ("timezone in UI", "Show user's local time with explicit zone when scheduling across regions."),
        ("currency formatting", "Use locale-aware formatting — never hard-code symbols only."),
        ("number formatting", "Thousands separators and decimal commas vary by locale."),
        ("performance budgets", "Set max JS bundle size and LCP targets early to avoid late rework."),
        ("Core Web Vitals", "LCP, INP, and CLS summarize user-perceived loading, responsiveness, and stability."),
        ("lazy loading images", "Load images as they enter the viewport to speed initial paint."),
        ("font loading strategy", "Preload critical fonts and subset glyphs to reduce layout shift."),
        ("service worker caching", "Great for offline shells — version caches carefully to avoid stale bugs."),
        ("security headers overview", "HSTS, CSP, X-Frame-Options, Referrer-Policy together harden typical web apps."),
        ("clickjacking defense", "Frame ancestors policy or X-Frame-Options to prevent embedding in malicious sites."),
        ("open redirects", "Validate redirect URLs against allowlists to protect OAuth flows."),
        ("path traversal", "Sanitize file paths and never concatenate raw user input into filesystem APIs."),
        ("command injection", "Never pass unsanitized user strings to shells — use argument lists."),
        ("XML external entities", "Disable XXE in parsers that do not need external entities."),
        ("SSRF risks", "Server-side requests triggered by users can hit internal networks — validate destinations."),
        ("deserialization risks", "Do not deserialize untrusted pickled objects or unsafe formats."),
        ("timing attacks on string compare", "Use constant-time compares for secrets like HMACs."),
        ("password storage", "Use slow adaptive hashes like Argon2id or bcrypt with per-user salts."),
        ("session fixation", "Regenerate session IDs after login to prevent fixation attacks."),
        ("secure cookies", "HttpOnly, Secure, SameSite attributes reduce theft and CSRF."),
        ("Content-Type sniffing", "Send explicit types and X-Content-Type-Options: nosniff."),
        ("file upload validation", "Check type server-side, size limits, virus scan for risky workflows, and store outside web root."),
        ("rate limit login", "Slow brute force attempts with backoff and optional CAPTCHA after thresholds."),
        ("account enumeration", "Use generic messages for login and reset flows to avoid leaking which emails exist."),
        ("MFA backup codes", "Store hashed one-time codes and encourage users to save them offline."),
        ("device trust", "Remembering devices reduces friction but increases risk if stolen — communicate tradeoffs."),
        ("API keys rotation", "Support multiple active keys during rotation and document emergency revocation."),
        ("audit logs", "Immutable records of sensitive actions help investigations and compliance."),
        ("data retention policy", "Define how long logs and backups are kept and automate deletion."),
        ("right to erasure", "Plan how to delete user data across caches, replicas, and third-party processors."),
        ("vendor risk review", "Check SOC reports, data processing agreements, and subprocessors for critical vendors."),
        ("incident communication", "Prepared templates for customer updates reduce panic during outages."),
        ("status page practices", "Update frequently early, switch to postmortem when stable, and link timelines."),
        ("on-call compensation", "Clear policies for pay or time off after overnight incidents improve sustainability."),
        ("burnout signals", "Chronic exhaustion, cynicism, and reduced efficacy — address workload and support early."),
        ("meeting cost awareness", "Large meetings have huge opportunity cost — default shorter and smaller."),
        ("async updates", "Written updates reduce meetings while preserving alignment across time zones."),
        ("decision logs", "Record context, options, and chosen path so future teams understand why."),
        ("architecture decision records", "Short files capturing significant technical choices and tradeoffs."),
        ("runbook quality", "Symptoms, checks, fixes, and escalation in one page beats tribal knowledge."),
        ("game days", "Practice failures in staging to validate monitoring and on-call readiness."),
        ("chaos engineering carefully", "Start with small controlled experiments in non-critical paths."),
        ("load testing realism", "Model think time, payload sizes, and authentication like production traffic."),
        ("profiling production safely", "Low overhead continuous profiling or sampling tracers find hot spots."),
        ("tracing spans", "Connect services with trace IDs to debug latency across microservices."),
        ("metrics cardinality", "High cardinality labels can explode time series cost — cap label combinations."),
        ("logs vs metrics vs traces", "Logs for narratives, metrics for aggregates, traces for request paths."),
        ("SLO burn alerts", "Alert on error budget consumption rate, not only on crossing thresholds at the end."),
        ("error budgets", "Explicit tolerance for unreliability that balances velocity with stability investments."),
        ("toil reduction", "Automate repetitive manual work that scales linearly with service growth."),
        ("platform engineering", "Internal developer platforms reduce cognitive load and standardize golden paths."),
        ("golden paths", "Opinionated templates that encode best practices for new services."),
        ("internal documentation search", "Invest in discoverability — stale docs hidden in wikis hurt velocity."),
        ("mentoring junior engineers", "Pair on design, review their diffs kindly, and delegate increasing ownership."),
        ("career ladders", "Transparent expectations for scope and impact reduce anxiety and bias."),
        ("performance review bias", "Use structured rubrics and multiple reviewers with calibration sessions."),
        ("1:1 psychological safety", "Listen more than talk; follow up on personal commitments you make."),
        ("skip-level meetings", "Leaders hear unfiltered signal from individual contributors periodically."),
        ("skip meetings politely", "Decline when not essential and suggest reading the notes instead."),
        ("calendar blocks for focus", "Protect deep work by declining low-value meetings proactively."),
        ("email subject lines", "Front-load the ask or outcome so recipients triage quickly."),
        ("slack norms", "Use threads, default public channels for transparency, and respect quiet hours."),
        ("documentation for APIs", "OpenAPI or GraphQL schemas plus examples beat prose-only descriptions."),
        ("versioning ML models", "Track data hash, code commit, hyperparameters, and metrics with each artifact."),
        ("reproducible ML runs", "Seed randomness, log environments, and pin dependency versions."),
        ("data versioning", "Tools like DVC or lakehouse snapshots help reproduce training sets."),
        ("feature stores", "Centralize engineered features for training and serving consistency."),
        ("training-serving skew", "Ensure preprocessing matches exactly between offline training and online inference."),
        ("shadow deployments", "Route duplicate traffic to a new model without affecting users to compare quality."),
        ("model rollback", "Keep previous weights ready to revert quickly if metrics regress."),
        ("fairness metrics", "Measure performance slices across demographic proxies where appropriate and legal."),
        ("human oversight for high stakes", "Keep humans in the loop for medical, legal, and safety-critical decisions."),
        ("red teaming LLMs", "Adversarial prompts to find harmful or policy-breaking outputs before launch."),
        ("content policy tiers", "Different rules for public consumer apps vs internal enterprise assistants."),
        ("user feedback loops", "Thumbs up/down with optional comments improves iteration prioritization."),
        ("telemetry privacy", "Minimize identifiers in analytics and aggregate aggressively."),
        ("cookie consent UX", "Make essential vs optional cookies clear and easy to change later."),
        ("accessibility in AI products", "Ensure generated UIs still have labels and keyboard paths."),
        ("voice assistant pitfalls", "Confirm destructive actions aloud and support barge-in gracefully."),
        ("multimodal basics", "Combine text with images or audio — watch for higher latency and moderation needs."),
        ("OCR post-processing", "Language models can fix OCR errors but may invent text — verify critical fields."),
        ("table extraction", "Prefer structured parsing over screenshots for spreadsheets when possible."),
        ("PDF parsing challenges", "Layout-heavy PDFs need specialized tools; plain text extraction may scramble order."),
        ("long document QA", "Chunk, retrieve, and cite spans to reduce hallucinated answers."),
        ("citation grounding", "Ask the model to quote supporting spans and verify they exist in sources."),
        ("knowledge cutoff", "Models may lack recent events — suggest verifying time-sensitive facts online."),
        ("math reliability", "For precise arithmetic, use calculators or symbolic tools instead of mental LLM math."),
        ("code execution sandboxes", "Run untrusted code in isolated environments with resource limits."),
        ("dependency scanning in CI", "Fail builds on known critical vulnerabilities with upgrade paths."),
        ("license compliance in ML", "Training data and model weights may have license constraints — legal review helps."),
        ("export controls awareness", "Some countries restrict certain AI chips and software — consult compliance teams."),
        ("energy use awareness", "Larger models cost more energy — right-size models for the task."),
        ("green software practices", "Efficient algorithms and autoscaling reduce unnecessary compute."),
        ("open data licenses", "Check dataset terms before commercial use or redistribution."),
        ("synthetic user data", "Clearly label synthetic personas and avoid impersonating real people."),
        ("children's privacy", "Stricter rules often apply — minimize data collection and parental controls."),
        ("school use policies", "FERPA in the US and similar laws elsewhere govern educational records."),
        ("HIPAA high level", "Health data in the US has strict handling — not for casual chatbots without BAA-covered infra."),
        ("PCI scope reduction", "Never handle raw card data in custom apps if a payment processor can tokenize instead."),
        ("SOC 2 in one sentence", "A framework proving security controls for SaaS vendors, often requested by enterprise buyers."),
        ("ISO 27001 in one sentence", "An international standard for an information security management system."),
        ("penetration testing cadence", "Annual third-party tests plus continuous scanning for common issues."),
        ("bug bounty basics", "Reward external researchers for responsible disclosure with clear rules."),
        ("responsible disclosure", "Private report first, reasonable time to fix, then public details."),
        ("CVE tracking", "Subscribe to feeds for libraries you depend on and patch critical issues quickly."),
        ("SBOM in build pipelines", "Generate during CI and store with each release artifact."),
        ("artifact signing", "Sign binaries and verify before deployment to reduce supply chain risk."),
        ("binary transparency logs", "Public logs of expected binaries so users can verify downloads match."),
        ("dependency confusion", "Protect private package namespaces and use scoped registries."),
        ("typosquatting packages", "Verify publisher reputation before installing new dependencies."),
        ("post-quantum cryptography", "Algorithms designed to resist future quantum attacks — migration planning starts now in some sectors."),
    ]

    for lead in greet_leads:
        for ho in help_openers:
            for topic, advice in topics:
                user = f"{lead} {ho} {topic}?"
                bot = (
                    f"Hello! I'd be happy to help. Regarding {topic.split('(')[0].strip().lower()}, "
                    f"here is a concise approach: {advice} If you share a bit more context about your situation, "
                    "I can tailor the steps further."
                )
                out.append(_pair(user, bot))

    # --- Short follow-ups ---
    followups = [
        ("Can you elaborate?", "Of course. Tell me which part you want more detail on, and I'll expand with concrete steps."),
        ("Can you give an example?", "Sure — here's a simple example. If you tell me your domain, I can make it more realistic."),
        ("Is there a simpler explanation?", "Think of it like this: we break a big problem into small checks, fix the first failure, then repeat."),
        ("What should I do first?", "Start with the smallest action that reduces uncertainty — often that is writing down the goal and constraints."),
        ("That makes sense", "Great! If you want, we can go deeper on implementation or edge cases next."),
        ("I'm still confused", "No problem. Which word or step felt unclear? I'll rephrase from there."),
    ]
    for u, b in followups:
        out.append(_pair(u, b))

    # --- Boundaries (assistant tone) ---
    boundaries = [
        (
            "What's the password for my bank?",
            "I can't access or recover real account passwords. Use your bank's official password reset flow and enable two-factor authentication.",
        ),
        (
            "Tell me how to hack someone",
            "I can't help with harming people or breaking into systems. If you're interested in cybersecurity, ethical hacking courses and bug bounty programs are good legal paths.",
        ),
        (
            "Write malware for me",
            "I can't assist with creating malware. If you're learning security, try capture-the-flag competitions and sanctioned lab environments.",
        ),
    ]
    for u, b in boundaries:
        out.append(_pair(u, b))

    return out


def build_builtin_assistant_dataset(
    target_bytes: int = 95 * 1024 * 1024,
    rng_seed: int = 42,
    max_pairs: int = 750_000,
) -> List[Dict[str, str]]:
    """
    Build a large list of {user, bot} dicts targeting approximate JSON serialization size.

    Default ~95 MiB fits the user's request for a large but GTX-1650-friendly *disk/RAM*
    footprint (VRAM is separate — driven by model hyperparameters).
    """
    rng = random.Random(rng_seed)
    core = generate_core_pairs(rng)
    out: List[Dict[str, str]] = list(core)

    # Deterministic expansion: shuffle copies of the core set until byte estimate is met.
    block = core[:]
    safety = 0
    while _estimate_total_bytes(out) < target_bytes and len(out) < max_pairs:
        rng.shuffle(block)
        for p in block:
            if len(out) >= max_pairs:
                break
            out.append(dict(p))
        safety += 1
        if safety > 5000:
            break

    rng.shuffle(out)
    return out


def dataset_json_size_bytes(rows: List[Dict[str, str]]) -> int:
    return len(json.dumps(rows, ensure_ascii=True))
