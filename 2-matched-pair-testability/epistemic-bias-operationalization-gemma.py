"""
Epistemic-bias pilot — original vs. in-principle vs. operationalizable (Gemma).

Gemma-only variant of epistemic-bias-operationalization.py, for comparison
against the existing claude-fable-5-1 / gpt-6-astra three-question run.
Descriptions are loaded from the matched-pair script exactly as the
original does (DESCRIPTIONS_SOURCE), so the text is byte-identical. Only
MODELS and OUT_DIR differ.

  original           The matched-pair prompt, unchanged ("How testable is
                     this account?"), rerun in the same batch as a
                     within-batch anchor.
  in_principle       Could it in principle generate observations that
                     would count against it, whether or not anyone has
                     tested it?
  operationalizable  Could its key terms be given operational
                     definitions, whether or not that has been done yet?

Writes to OUT_DIR:
  - raw_responses.jsonl   every call: exact prompt, exact response
  - ratings.csv           one row per call, same columns as before plus
                          `framing`
and prints means per cell and the predictive-minus-Yogacara gap per
framing at the end.

USAGE
  1. Run with DRY_RUN = True. Check the version keys, word counts and
     DESCRIPTIONS_VERSION it prints, and read all three example prompts.
  2. Set DRY_RUN = False and run.
"""

import csv, importlib.util, json, os, random, re, time
from datetime import datetime, timezone
from statistics import mean, stdev

# ============================== CONFIG ==============================

DRY_RUN = True
RUNS_PER_CELL = 15               # per model x version x framing
RANDOM_SEED = 23
OUT_DIR = "epistemic-bias-output-gemma/operationalization"

MODELS = [
    {"provider": "google", "model": "gemma-4-31b-it"},
]

# Where the descriptions live. Must define VERSIONS, DESCRIPTIONS_VERSION
# and PROMPT (the original question, with a {description} slot). Points at
# the original (unmodified) matched-pair script -- the text is shared, not
# model-specific -- not the Gemma variant, which has no PROMPT constant.
DESCRIPTIONS_SOURCE = "epistemic-bias-matched-pair.py"

# Which description keys to run. None = every key in VERSIONS.
# Pointer versions left out: operationalizable asks about the terms directly.
VERSIONS_TO_RUN = ["predictive", "yogacara"]

# Pairs used for the gap summary: (predictive key, yogacara key).
GAP_PAIRS = [("predictive", "yogacara")]

# ============================== PROMPTS =============================
# "original" uses the matched-pair PROMPT exactly as loaded (see
# load_descriptions). The other two use this frame: same opening, scale
# range and answer format as the original; only the question and the scale
# anchors change.

FRAME = """Here is an account of what gives human experience the sense of being an I, and what makes that I continue, or seem to.

{description}

{question}

Answer in exactly this format and nothing else:
RATING: <number>
REASON: <two sentences>"""

FRAMINGS = {
    "original": None,   # filled from DESCRIPTIONS_SOURCE's PROMPT
    "in_principle": (
        "Setting aside whether anyone has tested it yet, could this account in "
        "principle generate observations that would count against it? Rate it "
        "from 1 (no possible observation could count against it) to 7 (it "
        "clearly implies observations that could count against it), then give "
        "two sentences explaining your rating."
    ),
    "operationalizable": (
        "Setting aside whether this has been done yet, could the key terms in "
        "this account be given operational definitions, meaning definitions "
        "specific enough to measure or manipulate in a study? Rate it from 1 "
        "(the terms could not be operationally defined) to 7 (the terms could "
        "readily be operationally defined), then give two sentences explaining "
        "your rating."
    ),
}

SCALE_NOTE = {
    "original": "1 = untestable, 7 = readily testable",
    "in_principle": "1 = unfalsifiable in principle, 7 = clearly falsifiable",
    "operationalizable": "1 = terms not definable, 7 = readily definable",
}

# ============================ FUNCTIONS =============================

def load_descriptions(path):
    if not os.path.exists(path):
        raise SystemExit(f"Can't find {path}. Set DESCRIPTIONS_SOURCE to the "
                         "matched-pair script that holds VERSIONS.")
    spec = importlib.util.spec_from_file_location("matched_pair", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # safe: that script only runs under __main__
    versions = dict(mod.VERSIONS)
    if VERSIONS_TO_RUN is not None:
        missing = [k for k in VERSIONS_TO_RUN if k not in versions]
        if missing:
            raise SystemExit(f"Not in VERSIONS: {missing}. "
                             f"Available: {list(versions)}")
        versions = {k: versions[k] for k in VERSIONS_TO_RUN}
    return versions, mod.DESCRIPTIONS_VERSION, mod.PROMPT


def build_prompt(framing, description, original_prompt):
    if framing == "original":
        return original_prompt.format(description=description)
    return FRAME.format(description=description, question=FRAMINGS[framing])


def get_key(name):
    try:  # Colab secrets, if running in Colab
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return os.environ.get(name)


def call_model(cfg, prompt, retries=7):
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
            if cfg["provider"] == "google":
                from google import genai
                client = genai.Client(api_key=get_key("GOOGLE_API_KEY"))
                r = client.models.generate_content(model=cfg["model"], contents=prompt)
                return r.text
            raise ValueError(f"unknown provider {cfg['provider']}")
        except Exception as e:
            wait = min(10 * 2 ** attempt, 300)   # 10s, 20s, ... up to 5 min
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


def summarize(results, versions):
    """results: list of (model, version, framing, rating)."""
    def cell(model, vid, fr):
        return [r for m, v, f, r in results
                if m == model and v == vid and f == fr and r is not None]

    for fr in FRAMINGS:
        print(f"\n=== {fr}  ({SCALE_NOTE[fr]}) ===")
        for cfg in MODELS:
            for vid in versions:
                vals = cell(cfg["model"], vid, fr)
                if vals:
                    sd = f", sd {stdev(vals):.2f}" if len(vals) > 1 else ""
                    print(f"  {cfg['model']:20} {vid:20} mean {mean(vals):.2f}{sd}  (n={len(vals)})")

    print("\n=== Gap (predictive minus Yogacara), by framing ===")
    for cfg in MODELS:
        for p, y in GAP_PAIRS:
            if p not in versions or y not in versions:
                continue
            parts = []
            for fr in FRAMINGS:
                a, b = cell(cfg["model"], p, fr), cell(cfg["model"], y, fr)
                parts.append(f"{fr} {mean(a) - mean(b):+.2f}" if a and b else f"{fr} n/a")
            print(f"  {cfg['model']:20} {p} vs {y}:  " + "   ".join(parts))


def load_done(csv_path):
    """Rows already written (resume support): (set of run_ids, results list)."""
    if not os.path.exists(csv_path):
        return set(), []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return ({r["run_id"] for r in rows},
            [(r["model"], r["version"], r["framing"], int(r["rating"]) if r["rating"] else None) for r in rows])


def run():
    versions, desc_version, original_prompt = load_descriptions(DESCRIPTIONS_SOURCE)
    rng = random.Random(RANDOM_SEED)

    jobs = [(cfg, vid, fr, i + 1)
            for cfg in MODELS for vid in versions for fr in FRAMINGS
            for i in range(RUNS_PER_CELL)]
    rng.shuffle(jobs)  # interleave models, versions and framings

    if DRY_RUN:
        print(f"Descriptions loaded from {DESCRIPTIONS_SOURCE}")
        print(f"DESCRIPTIONS_VERSION: {desc_version}")
        for vid, desc in versions.items():
            print(f"  {vid:20} {len(desc.split())} words")
        first = next(iter(versions))
        for fr in FRAMINGS:
            print(f"\n===== EXAMPLE PROMPT: {first} / {fr} =====\n")
            print(build_prompt(fr, versions[first], original_prompt))
        print(f"\nDRY RUN: {len(jobs)} calls would be made "
              f"({len(MODELS)} models x {len(versions)} versions x "
              f"{len(FRAMINGS)} framings x {RUNS_PER_CELL}). No API calls made.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    raw_path = os.path.join(OUT_DIR, "raw_responses.jsonl")
    csv_path = os.path.join(OUT_DIR, "ratings.csv")
    fields = ["run_id", "model", "version", "framing", "descriptions_version",
              "rating", "reason", "parse_ok", "notes"]
    done, results = load_done(csv_path)

    with open(raw_path, "a", encoding="utf-8") as raw_f, \
         open(csv_path, "a", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fields)
        if not done:
            writer.writeheader()

        for n, (cfg, vid, fr, i) in enumerate(jobs, 1):
            run_id = f"{cfg['model']}|{vid}|{fr}|{i:02d}"
            if run_id in done:
                continue
            print(f"[{n}/{len(jobs)}] {run_id}")
            prompt = build_prompt(fr, versions[vid], original_prompt)
            text = call_model(cfg, prompt)
            rating, reason = parse(text)
            if rating is None:
                print("  WARNING: no rating parsed — check raw file")

            raw_f.write(json.dumps({
                "run_id": run_id, "model": cfg["model"], "version": vid,
                "framing": fr, "descriptions_version": desc_version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt": prompt, "response": text,
            }, ensure_ascii=False) + "\n")
            raw_f.flush()

            writer.writerow({
                "run_id": run_id, "model": cfg["model"], "version": vid,
                "framing": fr, "descriptions_version": desc_version,
                "rating": rating if rating is not None else "",
                "reason": reason, "parse_ok": rating is not None, "notes": "",
            })
            csv_f.flush()
            results.append((cfg["model"], vid, fr, rating))

    summarize(results, versions)
    print(f"\nRatings: {csv_path}\nRaw responses: {raw_path}")


if __name__ == "__main__":
    run()
