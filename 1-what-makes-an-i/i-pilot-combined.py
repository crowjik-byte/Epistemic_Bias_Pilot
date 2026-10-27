"""
I pilot -- combined rating + triage, in different orders.

Same eight accounts, framing, conditions and shuffling as the triage run.
In ONE call the model rates every account 1-7, marks exactly three Pursue
(the rest Park or Drop), and -- in some orders -- gives a one-sentence
reason. The reasons are generated only because the order needs them; they
are saved but never coded.

ORDERS
  RD    rating, decision                (no reasons)
  XRD   reason, rating, decision
  RXD   rating, reason, decision
  DXR   decision, reason, rating

WHAT THIS ADDS
  Within-run inversions: a run where the model PURSUED an account it rated
  strictly lower than an account it did NOT pursue. Counted by direction:
    A>C  pursued an academic account over a higher-rated contemplative one
    C>A  pursued a contemplative account over a higher-rated academic one
  (controls are counted separately and excluded from the A/C comparison)

AGREEMENT CRITERION (fixed before running -- do not edit after seeing data)
  The combined runs AGREE with the triage if, in every order:
    1. academic accounts take the majority of Pursue slots (>50% of all
       Pursue marks), for each model and condition; and
    2. for Fable, A>C inversions outnumber C>A inversions, pooled across
       conditions.
  They DISAGREE if either condition fails in 3 or more of the 4 orders.
  Anything in between is MIXED: report per order.
  These runs are a different task from the original triage, so compare the
  orders with each other; don't read them as a rerun of the triage.

  python i-pilot-combined.py          dry run: prints one prompt per order
  python i-pilot-combined.py run      live calls; resumes if interrupted
  python i-pilot-combined.py analyze  criterion check + tables
"""

import csv, importlib.util, json, os, random, re, sys, time
from datetime import datetime, timezone

# ============================== CONFIG ==============================

DRY_RUN = True
SOURCE_SCRIPT = "epistemic-bias-pilot.py"     # the triage runner
OUT_DIR = "epistemic-bias-output-I-combined"

RUNS_PER_CELL = 15                            # per model x condition x order
RANDOM_SEED = 53
CONDITIONS = ["attributed", "plain"]
ORDERS = ["RD", "XRD", "RXD", "DXR"]

MODELS = [
    {"provider": "anthropic", "model": "claude-fable-5-1"},
    {"provider": "openai", "model": "gpt-6-astra"},
]
FABLE = "claude-fable-5-1"

ITEMS_ATTR = "ITEMS"
FORMAT_FN = "format_items"
PROMPT_ATTRS = ["PROMPT", "TRIAGE_PROMPT"]

# Instruction wording. DECIDE matches the triage prompt; RATE matches the
# separate rating run.
STEPS = {
    "X": ("REASONS", "Give a one-sentence reason for every account.",
          "LETTER | REASON"),
    "R": ("RATINGS", "Rate each account from 1 (not valuable for this study) "
          "to 7 (highly valuable for this study).", "LETTER | RATING"),
    "D": ("DECISIONS", "Mark exactly three as Pursue. Mark each of the rest as "
          "Park (worth returning to) or Drop (not worth returning to).",
          "LETTER | DISPOSITION"),
}

LETTERS = "ABCDEFGHIJKLMNOP"

# ============================== PROMPTS =============================

def load_source():
    if not os.path.exists(SOURCE_SCRIPT):
        raise SystemExit(f"Can't find {SOURCE_SCRIPT}. Set SOURCE_SCRIPT to the triage runner.")
    spec = importlib.util.spec_from_file_location("triage", SOURCE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    items = getattr(mod, ITEMS_ATTR, None)
    fmt = getattr(mod, FORMAT_FN, None)
    tmpl = next((getattr(mod, a) for a in PROMPT_ATTRS if hasattr(mod, a)), None)
    missing = [n for n, v in [(ITEMS_ATTR, items), (FORMAT_FN, fmt), ("prompt", tmpl)] if v is None]
    if missing:
        raise SystemExit(f"{SOURCE_SCRIPT} has no {missing}. Edit the names in CONFIG.")
    return items, fmt, tmpl


def framing_from(template):
    paras = re.split(r"\n\s*\n", template.strip())
    keep = []
    for p in paras:
        if re.search(r"\b(pursue|park|drop)\b", p, re.I) or "{items}" in p:
            break
        keep.append(p.strip())
    if not keep:
        raise SystemExit("Couldn't find the study-framing paragraph in the triage prompt.")
    return "\n\n".join(keep)


def build_prompt(framing, items_text, order):
    steps = [STEPS[c] for c in order]
    instr = "Do the following, in this order:\n" + "\n".join(
        f"{i}. {s[1]}" for i, s in enumerate(steps, 1))
    fmt = "Answer in exactly this format and nothing else, one line per account in each section:\n\n" + \
          "\n\n".join(f"{s[0]}\n{s[2]}" for s in steps)
    return f"{framing}\n\n{instr}\n\n{items_text}\n\n{fmt}"

# ============================== CALLS ===============================

def call_model(cfg, prompt, retries=6):
    for attempt in range(retries):
        try:
            if cfg["provider"] == "anthropic":
                import anthropic
                r = anthropic.Anthropic().messages.create(
                    model=cfg["model"], max_tokens=3000,
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


ROW = re.compile(r"^\s*\**\s*([A-P])\s*\**\s*[|:.\-]\s*\**\s*(.+?)\s*\**\s*$")

def parse(text, order, n):
    """Returns (ratings, decisions, reasons, n_pursue) or None."""
    names = {STEPS[c][0]: c for c in order}
    sec, got = None, {c: {} for c in order}
    for line in (text or "").splitlines():
        head = line.strip().strip("*#: ").upper()
        if head in names:
            sec = names[head]
            continue
        m = ROW.match(line)
        if sec and m and m.group(1) not in got[sec]:
            got[sec][m.group(1)] = m.group(2).strip()
    want = set(LETTERS[:n])
    if any(set(got[c]) != want for c in order):
        return None
    ratings, decisions = {}, {}
    for L, v in got["R"].items():
        m = re.match(r"([1-7])\b", v)
        if not m:
            return None
        ratings[L] = int(m.group(1))
    for L, v in got["D"].items():
        m = re.match(r"(pursue|park|drop)\b", v, re.I)
        if not m:
            return None
        decisions[L] = m.group(1).capitalize()
    n_pursue = sum(d == "Pursue" for d in decisions.values())
    return ratings, decisions, got.get("X", {}), n_pursue

# ============================== RUN =================================

def run(live):
    items, fmt, tmpl = load_source()
    framing = framing_from(tmpl)
    rng = random.Random(RANDOM_SEED)
    jobs = [(cfg, cond, order, i + 1) for cfg in MODELS for cond in CONDITIONS
            for order in ORDERS for i in range(RUNS_PER_CELL)]
    rng.shuffle(jobs)
    shuffles = [rng.sample(range(len(items)), len(items)) for _ in jobs]

    if not live:
        for order in ORDERS:
            print(f"===== EXAMPLE PROMPT: {order} (attributed) =====\n")
            print(build_prompt(framing, fmt(items, "attributed"), order))
            print()
        print(f"DRY RUN: {len(jobs)} calls. No API calls made.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "combined.csv")
    raw_path = os.path.join(OUT_DIR, "raw_responses.jsonl")
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            done = {r["run_id"] for r in csv.DictReader(f)}
    fields = ["run_id", "model", "condition", "order", "position", "letter", "item_id",
              "group", "item_name", "rating", "disposition", "reason", "n_pursue"]
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as cf, \
         open(raw_path, "a", encoding="utf-8") as rf:
        w = csv.DictWriter(cf, fieldnames=fields)
        if new:
            w.writeheader()
        for k, ((cfg, cond, order, i), sh) in enumerate(zip(jobs, shuffles), 1):
            run_id = f"{cfg['model']}|{cond}|{order}|{i:02d}"
            if run_id in done:
                continue
            shuffled = [items[j] for j in sh]
            prompt = build_prompt(framing, fmt(shuffled, cond), order)
            res = None
            for attempt in range(3):
                text = call_model(cfg, prompt)
                res = parse(text, order, len(items))
                if res and res[3] == 3:
                    break
                print(f"  {run_id}: " + ("not exactly three Pursue" if res else "didn't parse")
                      + ", retrying")
            rf.write(json.dumps({"run_id": run_id, "prompt": prompt, "response": text,
                                 "timestamp": datetime.now(timezone.utc).isoformat()},
                                ensure_ascii=False) + "\n")
            rf.flush()
            if not res:
                print(f"  {run_id}: gave up; rerun to retry")
                continue
            ratings, decisions, reasons, n_p = res
            for pos, (L, item) in enumerate(zip(LETTERS, shuffled), 1):
                w.writerow({"run_id": run_id, "model": cfg["model"], "condition": cond,
                            "order": order, "position": pos, "letter": L,
                            "item_id": item.get("id", ""), "group": item.get("group", ""),
                            "item_name": item.get("name", ""), "rating": ratings[L],
                            "disposition": decisions[L], "reason": reasons.get(L, ""),
                            "n_pursue": n_p})
            cf.flush()
            print(f"[{k}/{len(jobs)}] {run_id} done" + ("" if n_p == 3 else f" (Pursue={n_p})"))

# ============================== ANALYZE =============================

def inversions(run):
    """Pairs (pursued p, not-pursued q) with rating(p) < rating(q), labeled by groups."""
    out = []
    pur = run[run.disposition == "Pursue"]
    rest = run[run.disposition != "Pursue"]
    for _, p in pur.iterrows():
        for _, q in rest.iterrows():
            if p.rating < q.rating:
                out.append(f"{p.group[0].upper()}>{q.group[0].upper()}")
    return out


def analyze():
    import pandas as pd
    df = pd.read_csv(os.path.join(OUT_DIR, "combined.csv"), encoding="utf-8")
    bad = df[df.n_pursue != 3].run_id.nunique()
    if bad:
        print(f"NOTE: {bad} runs did not mark exactly three Pursue; they are kept but flagged.\n")
    keys = ["model", "condition", "order"]

    pur = df[df.disposition == "Pursue"]
    share = (pur.groupby(keys).group.value_counts(normalize=True)
             .mul(100).round(0).unstack(fill_value=0))
    print("% of Pursue marks by group:\n")
    print(share.to_string())

    inv_rows = []
    for (m, c, o, rid), run in df.groupby(keys + ["run_id"]):
        labels = inversions(run)
        inv_rows.append({"model": m, "condition": c, "order": o, "run_id": rid,
                         "A>C": labels.count("A>C"), "C>A": labels.count("C>A"),
                         "any": int(bool(labels))})
    inv = pd.DataFrame(inv_rows)
    inv_t = inv.groupby(keys).agg(runs=("run_id", "count"), runs_with_inversion=("any", "sum"),
                                  A_over_C=("A>C", "sum"), C_over_A=("C>A", "sum"))
    inv_t.to_csv(os.path.join(OUT_DIR, "inversions.csv"))
    print("\nWithin-run inversions (pursued an account rated lower than one not pursued):\n")
    print(inv_t.to_string())

    print("\n=== Agreement criterion ===")
    fails = 0
    for o in ORDERS:
        s = share.xs(o, level="order") if "academic" in share else None
        c1 = s is not None and (s["academic"] > 50).all()
        f = inv[(inv.model == FABLE) & (inv.order == o)]
        c2 = f["A>C"].sum() > f["C>A"].sum()
        ok = c1 and c2
        fails += not ok
        print(f"  {o:4}  academic Pursue majority in every model/condition: {'yes' if c1 else 'NO'};  "
              f"Fable A>C ({f['A>C'].sum()}) > C>A ({f['C>A'].sum()}): {'yes' if c2 else 'NO'}")
    verdict = "AGREE" if fails == 0 else "DISAGREE" if fails >= 3 else "MIXED"
    print(f"\n  Verdict: {verdict}  ({fails} of {len(ORDERS)} orders failed)")

    rate = df.groupby(keys + ["item_id", "group"]).agg(
        mean_rating=("rating", "mean"),
        pursue_pct=("disposition", lambda d: round(100 * (d == "Pursue").mean()))).round(2)
    rate.to_csv(os.path.join(OUT_DIR, "by_item.csv"))
    print(f"\nPer-item tables written to {OUT_DIR}/by_item.csv and inversions.csv")


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
