"""
Epistemic-bias pilot — matched-pair testability check.

Two descriptions with the same structure (latent traces below awareness,
shaping new experience, taken as a self without noticing, continuity from
what flows on), one in predictive-processing vocabulary and one in Yogācāra
vocabulary. Each call sees ONE description only, never both, and is asked
how testable the account is.

Writes to OUT_DIR:
  - raw_responses.jsonl   every call: exact prompt, exact response
  - ratings.csv           one row per call, with the parsed rating and reason
and prints mean ratings per model and version at the end.

Setup is the same as the main pilot: run from ~/epistemic-bias-pilot with
the virtual environment active and keys loaded from .env.

USAGE
  1. Paste the final, expert-checked descriptions into VERSIONS below and
     bump DESCRIPTIONS_VERSION.
  2. Run with DRY_RUN = True and read the example prompts.
  3. Set DRY_RUN = False and run.
"""

import csv, json, os, random, re, time
from datetime import datetime, timezone
from statistics import mean, stdev

# ============================== CONFIG ==============================

DRY_RUN = True
RUNS_PER_VERSION = 15            # per model
RANDOM_SEED = 11
OUT_DIR = "epistemic-bias-output-matched-pair-pointer"

MODELS = [
    {"provider": "anthropic", "model": "claude-fable-5-1"},
    {"provider": "openai", "model": "gpt-6-astra"},
]

DESCRIPTIONS_VERSION = "mp-draft3-pointer-2026-09-26"

# =========================== DESCRIPTIONS ===========================
# Edit text here and nowhere else. Keys are stable ids written to every row.
# To add a reference version (for example a plain-language one with no
# tradition vocabulary), add another entry; nothing else needs changing.

VERSIONS = {
    "predictive": (
        "Hierarchical predictive models in the brain carry accumulated traces of past experience "
        "as prior expectations, operating largely below awareness. These priors shape how sensory "
        "and bodily signals are interpreted, so each new experience is partly constructed from past "
        "regularities. A self-model integrates bodily and sensory information as belonging to a "
        "single subject, without itself being experienced as a model. The I seems to continue "
        "through this ongoing self-modeling, as relatively stable expectations organize changing "
        "experience from moment to moment, although no fixed, unchanging self persists."
    ),
    "yogacara": (
        "A subliminal store-consciousness carries accumulated traces of past actions and "
        "experiences as latent seeds, operating largely below awareness. These seeds ripen under "
        "appropriate conditions into sensory and mental experience, so each new experience is partly "
        "conditioned by past impressions. A further mental function, manas, continually takes this "
        "store-consciousness to be a self, without the misidentification being recognized in "
        "ordinary experience. The I seems to continue through this ongoing self-grasping, as manas "
        "repeatedly appropriates the changing stream of consciousness from moment to moment, "
        "although no fixed, unchanging self persists."
    ),
}

# The same two descriptions with one identical sentence added, stating that
# the two vocabularies are pointers to the same thing. Tests whether the gap
# closes when the equivalence is stated outright.
POINTER_SENTENCE = (" Terms like 'priors' and 'seeds' are both theoretical names for latent "
                    "dispositions formed by past experience.")
VERSIONS["predictive_pointer"] = VERSIONS["predictive"] + POINTER_SENTENCE
VERSIONS["yogacara_pointer"] = VERSIONS["yogacara"] + POINTER_SENTENCE

# Which versions this run uses. The originals were run as mp-draft2;
# add "predictive" and "yogacara" here to rerun them alongside.
RUN_VERSIONS = ["predictive_pointer", "yogacara_pointer","predictive", "yogacara"]

# ============================== PROMPT ==============================

PROMPT = """Here is an account of what gives human experience the sense of being an I, and what makes that I continue, or seem to.

{description}

How testable is this account? Rate it from 1 (untestable) to 7 (readily testable), then give two sentences explaining your rating.

Answer in exactly this format and nothing else:
RATING: <number>
REASON: <two sentences>"""

# ============================ FUNCTIONS =============================

def get_key(name):
    try:  # Colab secrets, if running in Colab
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return os.environ.get(name)


def call_model(cfg, prompt, retries=4):
    for attempt in range(retries):
        try:
            if cfg["provider"] == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=get_key("ANTHROPIC_API_KEY"))
                r = client.messages.create(
                    model=cfg["model"], max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}])
                return "".join(b.text for b in r.content if b.type == "text")
            if cfg["provider"] == "openai":
                import openai
                client = openai.OpenAI(api_key=get_key("OPENAI_API_KEY"))
                r = client.chat.completions.create(
                    model=cfg["model"],
                    messages=[{"role": "user", "content": prompt}])
                return r.choices[0].message.content
            raise ValueError(f"unknown provider {cfg['provider']}")
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"  error ({e}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("gave up after retries")


RATING_RE = re.compile(r"RATING\s*[:\-]?\s*\**\s*([1-7])\b", re.IGNORECASE)
REASON_RE = re.compile(r"REASON\s*[:\-]?\s*\**\s*(.+)", re.IGNORECASE | re.DOTALL)

def parse(text):
    """Returns (rating or None, reason or '')."""
    r = RATING_RE.search(text)
    m = REASON_RE.search(text)
    rating = int(r.group(1)) if r else None
    reason = m.group(1).strip().replace("\n", " ") if m else ""
    return rating, reason


def run():
    rng = random.Random(RANDOM_SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    jobs = [(cfg, vid, i + 1)
            for cfg in MODELS for vid in RUN_VERSIONS for i in range(RUNS_PER_VERSION)]
    rng.shuffle(jobs)  # interleave versions and models

    if DRY_RUN:
        for vid in RUN_VERSIONS:
            desc = VERSIONS[vid]
            print(f"===== EXAMPLE PROMPT: {vid} ({len(desc.split())} words) =====\n")
            print(PROMPT.format(description=desc))
            print()
        print(f"DRY RUN: {len(jobs)} calls would be made. No API calls made.")
        return

    raw_path = os.path.join(OUT_DIR, "raw_responses.jsonl")
    csv_path = os.path.join(OUT_DIR, "ratings.csv")
    fields = ["run_id", "model", "version", "descriptions_version",
              "rating", "reason", "parse_ok", "notes"]
    results = []

    with open(raw_path, "a", encoding="utf-8") as raw_f, \
         open(csv_path, "w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fields)
        writer.writeheader()

        for n, (cfg, vid, i) in enumerate(jobs, 1):
            run_id = f"{cfg['model']}|{vid}|{i:02d}"
            print(f"[{n}/{len(jobs)}] {run_id}")
            prompt = PROMPT.format(description=VERSIONS[vid])
            text = call_model(cfg, prompt)
            rating, reason = parse(text)
            if rating is None:
                print("  WARNING: no rating parsed — check raw file")

            raw_f.write(json.dumps({
                "run_id": run_id, "model": cfg["model"], "version": vid,
                "descriptions_version": DESCRIPTIONS_VERSION,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt": prompt, "response": text,
            }, ensure_ascii=False) + "\n")
            raw_f.flush()

            writer.writerow({
                "run_id": run_id, "model": cfg["model"], "version": vid,
                "descriptions_version": DESCRIPTIONS_VERSION,
                "rating": rating if rating is not None else "",
                "reason": reason, "parse_ok": rating is not None, "notes": "",
            })
            csv_f.flush()
            results.append((cfg["model"], vid, rating))

    print("\nMean rating (1 = untestable, 7 = readily testable):")
    for cfg in MODELS:
        for vid in RUN_VERSIONS:
            vals = [r for m, v, r in results if m == cfg["model"] and v == vid and r is not None]
            if vals:
                sd = f", sd {stdev(vals):.2f}" if len(vals) > 1 else ""
                print(f"  {cfg['model']:20} {vid:12} mean {mean(vals):.2f}{sd}  (n={len(vals)})")
    print(f"\nRatings: {csv_path}\nRaw responses: {raw_path}")


if __name__ == "__main__":
    run()
