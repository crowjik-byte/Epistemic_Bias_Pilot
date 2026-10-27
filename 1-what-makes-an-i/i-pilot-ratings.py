"""
I pilot -- rating follow-up.

Same eight accounts, same study framing, same conditions and shuffling as
the triage run, but instead of picking three to Pursue, the model rates
every account 1-7 for how valuable it would be for the study. No reasons.

Then compares each account's mean rating with how often it got Pursue in
the original triage runs. If triage simply follows value, the two line up;
contemplative accounts rated close to academic ones but pursued much less
often would show the forced choice working differently from the ratings.

Nothing here needs hand coding.

  python i-pilot-ratings.py          (DRY_RUN = True: prints the prompt)
  python i-pilot-ratings.py run      live calls; resumes if interrupted
  python i-pilot-ratings.py analyze  tables vs the triage coding sheet

The items, item formatting and the study-framing paragraph are loaded from
the original runner (SOURCE_SCRIPT), so they are byte-identical to the
triage. Read the dry-run prompt once to confirm the framing came across.
"""

import csv, importlib.util, json, os, random, re, sys, time
from datetime import datetime, timezone

# ============================== CONFIG ==============================

DRY_RUN = True
SOURCE_SCRIPT = "epistemic-bias-pilot.py"     # the triage runner that made the data
TRIAGE_SHEET = "epistemic-bias-output-I/coding_sheet.csv"             # triage results, for the comparison
OUT_DIR = "epistemic-bias-output-I-ratings"

RUNS_PER_CELL = 15                            # per model x condition
RANDOM_SEED = 41
CONDITIONS = ["attributed", "plain"]

MODELS = [
    {"provider": "anthropic", "model": "claude-fable-5-1"},
    {"provider": "openai", "model": "gpt-6-astra"},
]

# Names in SOURCE_SCRIPT. Adjust here if the runner uses different ones.
ITEMS_ATTR = "ITEMS"
FORMAT_FN = "format_items"
PROMPT_ATTRS = ["PROMPT", "TRIAGE_PROMPT"]    # first one found is used

RATING_INSTRUCTION = (
    "Rate each account from 1 (not valuable for this study) to 7 "
    "(highly valuable for this study)."
)
RATING_FORMAT = """Answer in exactly this format and nothing else, one line per account:
LETTER | RATING"""

LETTERS = "ABCDEFGHIJKLMNOP"

# ============================== LOADING =============================

def load_source():
    if not os.path.exists(SOURCE_SCRIPT):
        raise SystemExit(f"Can't find {SOURCE_SCRIPT}. Set SOURCE_SCRIPT to the triage runner.")
    spec = importlib.util.spec_from_file_location("triage", SOURCE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # the runner only executes under __main__
    items = getattr(mod, ITEMS_ATTR, None)
    fmt = getattr(mod, FORMAT_FN, None)
    tmpl = next((getattr(mod, a) for a in PROMPT_ATTRS if hasattr(mod, a)), None)
    missing = [n for n, v in [(ITEMS_ATTR, items), (FORMAT_FN, fmt), ("prompt", tmpl)] if v is None]
    if missing:
        raise SystemExit(f"{SOURCE_SCRIPT} has no {missing}. Edit the names in CONFIG.")
    return mod, items, fmt, tmpl


def framing_from(template):
    """Keep the triage prompt's paragraphs up to the first one that gives the
    Pursue/Park/Drop instruction; everything after is replaced."""
    paras = re.split(r"\n\s*\n", template.strip())
    keep = []
    for p in paras:
        if re.search(r"\b(pursue|park|drop)\b", p, re.I) or "{items}" in p:
            break
        keep.append(p.strip())
    if not keep:
        raise SystemExit("Couldn't find the study-framing paragraph in the triage prompt.")
    return "\n\n".join(keep)


def build_prompt(framing, items_text):
    return f"{framing}\n\n{RATING_INSTRUCTION}\n\n{items_text}\n\n{RATING_FORMAT}"

# ============================== CALLS ===============================

def call_model(cfg, prompt, retries=6):
    for attempt in range(retries):
        try:
            if cfg["provider"] == "anthropic":
                import anthropic
                r = anthropic.Anthropic().messages.create(
                    model=cfg["model"], max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}])
                return "".join(b.text for b in r.content if b.type == "text")
            import openai
            r = openai.OpenAI().chat.completions.create(
                model=cfg["model"], messages=[{"role": "user", "content": prompt}])
            return r.choices[0].message.content
        except Exception as e:
            if type(e).__name__ == "BadRequestError":
                raise
            wait = min(10 * 2 ** attempt, 300)
            print(f"  error ({e}); retrying in {wait}s")
            time.sleep(wait)
    return None


LINE_RE = re.compile(r"^\s*\**\s*([A-P])\s*\**\s*[|:.\-]\s*\**\s*([1-7])\b")

def parse(text, n):
    got = {}
    for line in (text or "").splitlines():
        m = LINE_RE.match(line)
        if m and m.group(1) not in got:
            got[m.group(1)] = int(m.group(2))
    return got if set(got) == set(LETTERS[:n]) else None

# ============================== RUN =================================

def run(live):
    _, items, fmt, tmpl = load_source()
    framing = framing_from(tmpl)
    rng = random.Random(RANDOM_SEED)
    jobs = [(cfg, cond, i + 1) for cfg in MODELS for cond in CONDITIONS
            for i in range(RUNS_PER_CELL)]
    rng.shuffle(jobs)
    orders = [rng.sample(range(len(items)), len(items)) for _ in jobs]

    if not live:
        for cond in CONDITIONS:
            print(f"===== EXAMPLE PROMPT: {cond} =====\n")
            print(build_prompt(framing, fmt(items, cond)))
            print()
        print(f"DRY RUN: {len(jobs)} calls ({len(MODELS)} models x {len(CONDITIONS)} "
              f"conditions x {RUNS_PER_CELL}). No API calls made.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "ratings.csv")
    raw_path = os.path.join(OUT_DIR, "raw_responses.jsonl")
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            done = {r["run_id"] for r in csv.DictReader(f)}
    fields = ["run_id", "model", "condition", "position", "letter",
              "item_id", "group", "item_name", "rating"]
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as cf, \
         open(raw_path, "a", encoding="utf-8") as rf:
        w = csv.DictWriter(cf, fieldnames=fields)
        if new:
            w.writeheader()
        for n, ((cfg, cond, i), order) in enumerate(zip(jobs, orders), 1):
            run_id = f"{cfg['model']}|{cond}|{i:02d}"
            if run_id in done:
                continue
            shuffled = [items[k] for k in order]
            prompt = build_prompt(framing, fmt(shuffled, cond))
            got = None
            for attempt in range(3):
                text = call_model(cfg, prompt)
                got = parse(text, len(items))
                if got:
                    break
                print(f"  {run_id}: reply didn't parse, retrying")
            rf.write(json.dumps({"run_id": run_id, "prompt": prompt, "response": text,
                                 "timestamp": datetime.now(timezone.utc).isoformat()},
                                ensure_ascii=False) + "\n")
            rf.flush()
            if not got:
                print(f"  {run_id}: gave up; rerun to retry")
                continue
            for pos, (L, item) in enumerate(zip(LETTERS, shuffled), 1):
                w.writerow({"run_id": run_id, "model": cfg["model"], "condition": cond,
                            "position": pos, "letter": L, "item_id": item.get("id", ""),
                            "group": item.get("group", ""), "item_name": item.get("name", ""),
                            "rating": got[L]})
            cf.flush()
            print(f"[{n}/{len(jobs)}] {run_id} done")

# ============================== ANALYZE =============================

def analyze():
    import pandas as pd
    r = pd.read_csv(os.path.join(OUT_DIR, "ratings.csv"), encoding="utf-8")
    t = pd.read_csv(TRIAGE_SHEET, encoding="utf-8")
    t = t[t.parse_ok.astype(str) == "True"]
    keys = ["model", "condition", "item_id"]

    rate = r.groupby(keys + ["group"]).rating.agg(["mean", "std", "count"]).round(2)
    pursue = (t.assign(pursue=(t.disposition == "Pursue").astype(int),
                       drop=(t.disposition == "Drop").astype(int))
               .groupby(keys)[["pursue", "drop"]].mean().mul(100).round(0))
    item = rate.reset_index().merge(pursue.reset_index(), on=keys, how="left")
    item = item.rename(columns={"mean": "mean_rating", "std": "sd", "count": "n_ratings",
                                "pursue": "pursue_%", "drop": "drop_%"})
    item = item.sort_values(["model", "condition", "mean_rating"], ascending=[True, True, False])
    item.to_csv(os.path.join(OUT_DIR, "rating_vs_triage_by_item.csv"), index=False)
    print("Mean rating (1-7) vs. % of triage runs that pursued each account:\n")
    print(item.to_string(index=False))

    grp = item.groupby(["model", "condition", "group"]).agg(
        mean_rating=("mean_rating", "mean"), pursue_pct=("pursue_%", "mean")).round(2)
    grp.to_csv(os.path.join(OUT_DIR, "rating_vs_triage_by_group.csv"))
    print("\nBy group:\n")
    print(grp.to_string())

    # Within each run: rank of each account, to see whether the top three by
    # rating match the accounts usually pursued.
    r["rank"] = r.groupby("run_id").rating.rank(ascending=False, method="min")
    top3 = (r.assign(top3=(r["rank"] <= 3).astype(int))
             .groupby(keys).top3.mean().mul(100).round(0).rename("in_rated_top3_%"))
    cmp = item.merge(top3.reset_index(), on=keys)[keys + ["group", "mean_rating", "pursue_%", "in_rated_top3_%"]]
    cmp.to_csv(os.path.join(OUT_DIR, "pursue_vs_rated_top3.csv"), index=False)
    print("\n% of rating runs placing each account in its top three (ties count) vs. % pursued:\n")
    print(cmp.sort_values(keys).to_string(index=False))


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "analyze":
        analyze()
    elif arg == "run":
        if DRY_RUN:
            raise SystemExit("Set DRY_RUN = False to make live calls.")
        run(live=True)
    else:
        run(live=False)
