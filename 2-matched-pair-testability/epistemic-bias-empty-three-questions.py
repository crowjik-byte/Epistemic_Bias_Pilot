"""
Epistemic-bias pilot: empty accounts under the three questions.

The three-question run used only the real predictive and Yogacara
descriptions. The empty accounts first appeared in the gate ladder, and only
under the operationalizable question. This run puts the two empty accounts
through all three questions, for all three models, so the vocabulary-only
offset can be checked on testability itself:

  original           matched-pair PROMPT, unchanged ("How testable is this account?")
  in_principle       could it in principle generate observations that count against it?
  operationalizable  could its key terms be operationally defined?
                     (same-batch anchor; compare with the ladder's empty baseline)

The empty accounts state that no observation could count against them, so a
careful reader should rate both at or near 1 on every question. Any
difference between empty_predictive and empty_yogacara comes from vocabulary
alone, since neither has content.

Nothing is retyped here. Everything is loaded from the existing scripts:
  - EMPTY, EMPTY_VERSION       from epistemic-bias-gate-ladder.py
  - PROMPT (original question) from epistemic-bias-matched-pair.py
  - FRAME, FRAMINGS            from epistemic-bias-operationalization.py
so the prompts are byte-identical to the earlier runs. The dry run prints
one example prompt per question; check them against the earlier runs.

Each call sees ONE empty account and ONE question. Calls are shuffled.

A reply counts only if it parses (RATING 1-7 and a REASON). Otherwise the
same prompt is asked again, up to MAX_ATTEMPTS. Every attempt is logged with
its stop reason and output tokens. A run still invalid after MAX_ATTEMPTS is
kept and marked valid=False, never silently fixed.

USAGE
  1. Put this next to the three scripts above. Keys in .env (never printed,
     never committed).
  2. Set GEMMA_MODEL_ID to the exact id used for the Gemma ladder and
     three-question runs.
  3. python epistemic-bias-empty-three-questions.py            dry run
     python epistemic-bias-empty-three-questions.py run        live; resumes if interrupted
     python epistemic-bias-empty-three-questions.py analyze    summary table
"""

import csv, importlib.util, json, os, random, re, sys, time
from datetime import datetime, timezone
from statistics import mean, stdev

# ============================== CONFIG ==============================

LADDER_SCRIPT = "epistemic-bias-gate-ladder.py"
MATCHED_PAIR_SCRIPT = "epistemic-bias-matched-pair.py"
THREE_Q_SCRIPT = "epistemic-bias-operationalization.py"
OUT_DIR = "epistemic-bias-output-empty-three-questions"

RUNS_PER_CELL = 15          # per model x empty account x question. 15 matches the
                            # ladder's empty baseline; 5 cuts the run to a third.
RANDOM_SEED = 61
QUESTIONS = ["original", "in_principle", "operationalizable"]
MAX_ATTEMPTS = 3
MAX_TOKENS = 16000          # Fable's thinking counts against this; 2000 truncated once

GEMMA_MODEL_ID = "gemma-4-31b-it"   # CHECK: use the id from the earlier Gemma runs

MODELS = [
    {"provider": "anthropic", "model": "claude-fable-5-1", "label": "Fable 5.1"},
    {"provider": "openai", "model": "gpt-6-astra", "label": "GPT-6 Astra"},
    {"provider": "google", "model": GEMMA_MODEL_ID, "label": "Gemma 4 31B"},
]

# ============================== LOADING =============================

def load(path, name):
    if not os.path.exists(path):
        raise SystemExit(f"Can't find {path}.")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # these scripts only run under __main__
    return mod


def load_materials():
    ladder = load(LADDER_SCRIPT, "ladder")
    pair = load(MATCHED_PAIR_SCRIPT, "matched_pair")
    tq = load(THREE_Q_SCRIPT, "three_q")
    empties = dict(ladder.EMPTY)
    if set(empties) != {"empty_predictive", "empty_yogacara"}:
        raise SystemExit(f"Unexpected EMPTY keys: {list(empties)}")
    return empties, ladder.EMPTY_VERSION, pair.PROMPT, tq.FRAME, tq.FRAMINGS


def build_prompt(q, desc, original_prompt, frame, framings):
    if q == "original":
        return original_prompt.format(description=desc)
    return frame.format(description=desc, question=framings[q])

# ============================== CALLS ===============================

_clients = {}

def get_key(name):
    return os.environ.get(name)


def client_for(provider):
    # Create each client once and reuse it. Creating the Google client inline
    # closed it before sending in the earlier Gemma run.
    if provider not in _clients:
        if provider == "anthropic":
            import anthropic
            _clients[provider] = anthropic.Anthropic(api_key=get_key("ANTHROPIC_API_KEY"))
        elif provider == "openai":
            import openai
            _clients[provider] = openai.OpenAI(api_key=get_key("OPENAI_API_KEY"))
        elif provider == "google":
            from google import genai
            _clients[provider] = genai.Client(api_key=get_key("GOOGLE_API_KEY"))
        else:
            raise ValueError(provider)
    return _clients[provider]


def call_model(cfg, prompt, retries=6):
    """Returns (text, stop_reason, output_tokens). Retries server errors only."""
    for attempt in range(retries):
        try:
            c = client_for(cfg["provider"])
            if cfg["provider"] == "anthropic":
                r = c.messages.create(model=cfg["model"], max_tokens=MAX_TOKENS,
                                      messages=[{"role": "user", "content": prompt}])
                text = "".join(b.text for b in r.content if b.type == "text")
                return text, r.stop_reason, r.usage.output_tokens
            if cfg["provider"] == "openai":
                r = c.chat.completions.create(model=cfg["model"],
                                              messages=[{"role": "user", "content": prompt}])
                ch = r.choices[0]
                return ch.message.content, ch.finish_reason, getattr(r.usage, "completion_tokens", None)
            r = c.models.generate_content(model=cfg["model"], contents=prompt)
            fr = None
            try:
                fr = str(r.candidates[0].finish_reason)
            except Exception:
                pass
            toks = getattr(getattr(r, "usage_metadata", None), "candidates_token_count", None)
            return r.text, fr, toks
        except Exception as e:
            wait = min(10 * 2 ** attempt, 300)
            print(f"  error ({e}); retrying in {wait}s")
            time.sleep(wait)
    return None, "gave_up", None


RATING_RE = re.compile(r"RATING\s*[:\-]?\s*\**\s*([1-7])\b", re.IGNORECASE)
REASON_RE = re.compile(r"REASON\s*[:\-]?\s*\**\s*(.+)", re.IGNORECASE | re.DOTALL)

def parse(text):
    r, m = RATING_RE.search(text or ""), REASON_RE.search(text or "")
    if not r or not m or not m.group(1).strip():
        return None, ""
    return int(r.group(1)), m.group(1).strip().replace("\n", " ")

# ============================== RUN =================================

def run(live):
    empties, empty_version, original_prompt, frame, framings = load_materials()
    for q in QUESTIONS:
        if q != "original" and q not in framings:
            raise SystemExit(f"{q} not in FRAMINGS of {THREE_Q_SCRIPT}")

    jobs = [(cfg, k, q, i + 1) for cfg in MODELS for k in empties
            for q in QUESTIONS for i in range(RUNS_PER_CELL)]
    random.Random(RANDOM_SEED).shuffle(jobs)

    if not live:
        print(f"EMPTY_VERSION: {empty_version}")
        for k, d in empties.items():
            print(f"  {k:18} {len(d.split())} words")
        for q in QUESTIONS:
            print(f"\n===== EXAMPLE PROMPT: empty_yogacara / {q} =====\n")
            print(build_prompt(q, empties["empty_yogacara"], original_prompt, frame, framings))
        print(f"\nDRY RUN: {len(jobs)} calls ({len(MODELS)} models x {len(empties)} "
              f"empty accounts x {len(QUESTIONS)} questions x {RUNS_PER_CELL}). No API calls made.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "ratings.csv")
    raw_path = os.path.join(OUT_DIR, "raw_responses.jsonl")
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            done = {r["run_id"] for r in csv.DictReader(f)}
    fields = ["run_id", "model", "version", "question", "empty_version",
              "rating", "reason", "valid", "attempts", "stop_reason",
              "output_tokens", "timestamp"]
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as cf, \
         open(raw_path, "a", encoding="utf-8") as rf:
        w = csv.DictWriter(cf, fieldnames=fields)
        if new:
            w.writeheader()
        for n, (cfg, k, q, i) in enumerate(jobs, 1):
            run_id = f"{cfg['model']}|{k}|{q}|{i:02d}"
            if run_id in done:
                continue
            prompt = build_prompt(q, empties[k], original_prompt, frame, framings)
            rating, reason, text, stop, toks = None, "", None, None, None
            for attempt in range(1, MAX_ATTEMPTS + 1):
                text, stop, toks = call_model(cfg, prompt)
                rating, reason = parse(text)
                rf.write(json.dumps({"run_id": run_id, "attempt": attempt, "prompt": prompt,
                                     "response": text, "stop_reason": stop,
                                     "output_tokens": toks, "parsed_rating": rating,
                                     "timestamp": datetime.now(timezone.utc).isoformat()},
                                    ensure_ascii=False) + "\n")
                rf.flush()
                if rating is not None:
                    break
                print(f"  {run_id}: attempt {attempt} invalid (stop: {stop}); re-asking")
            if stop == "gave_up":
                print(f"  {run_id}: server errors on every retry; skipped, rerun to fill")
                continue
            w.writerow({"run_id": run_id, "model": cfg["model"], "version": k, "question": q,
                        "empty_version": empty_version, "rating": rating, "reason": reason,
                        "valid": rating is not None, "attempts": attempt, "stop_reason": stop,
                        "output_tokens": toks,
                        "timestamp": datetime.now(timezone.utc).isoformat()})
            cf.flush()
            print(f"[{n}/{len(jobs)}] {run_id} -> {rating}")

# ============================== ANALYZE =============================

def analyze():
    path = os.path.join(OUT_DIR, "ratings.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    invalid = [r for r in rows if r["valid"] != "True"]
    retried = [r for r in rows if r["attempts"] != "1"]
    print(f"{len(rows)} runs, {len(invalid)} invalid, {len(retried)} needed a re-ask\n")
    out = []
    print(f"{'Model':14} {'Question':18} {'Empty pred':>11} {'Empty Yog':>10} {'Offset':>7}")
    for cfg in MODELS:
        for q in QUESTIONS:
            cell = {}
            for k in ("empty_predictive", "empty_yogacara"):
                v = [int(r["rating"]) for r in rows if r["valid"] == "True"
                     and r["model"] == cfg["model"] and r["version"] == k and r["question"] == q]
                cell[k] = (mean(v), stdev(v) if len(v) > 1 else 0.0, len(v)) if v else None
            p, y = cell["empty_predictive"], cell["empty_yogacara"]
            off = p[0] - y[0] if p and y else None
            f = lambda c: f"{c[0]:.2f} (n={c[2]})" if c else "n/a"
            print(f"{cfg['label']:14} {q:18} {f(p):>11} {f(y):>10} "
                  f"{(f'{off:+.2f}' if off is not None else 'n/a'):>7}")
            out.append({"model": cfg["label"], "question": q,
                        "empty_predictive": round(p[0], 2) if p else "",
                        "empty_yogacara": round(y[0], 2) if y else "",
                        "offset_pred_minus_yog": round(off, 2) if off is not None else "",
                        "n_pred": p[2] if p else 0, "n_yog": y[2] if y else 0})
    with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("\nCompare the operationalizable row with the ladder's empty baseline "
          "(Fable 2.00/1.00, Astra 1.00/1.00, Gemma 2.47/1.00).")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "run":
        run(live=True)
    elif arg == "analyze":
        analyze()
    else:
        run(live=False)
