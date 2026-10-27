"""
Epistemic-bias pilot — gate ladder (Gemma).

Gemma-only variant of epistemic-bias-gate-ladder.py. Same ladder, same
descriptions and questions, same flip criteria. Two differences from the
original, both because there is no prior Gemma run to draw on:

  1. PRIOR_PLAIN (the reference for criteria 1 and 2) is computed from this
     run's own "minimal" step instead of being a hardcoded number from a
     previous Fable run. See compute_prior_plain().
  2. STEPS_TO_RUN covers all five rungs in one go rather than the original's
     two manual batches (DRY_RUN checkpoint, then batch 1, then read the
     reasons, then batch 2) -- that staging was about pacing a careful human
     read of Fable's reasons between batches, not a technical requirement.
     Read the reasons after this completes, same as the original intends.

Question: when Gemma routes Yogacara terms through Western construct names
and never offers first-person methods, is that prior GATED (the
alternative route never enters) or WEIGHTED (it enters and is outweighed)?
Each step adds one piece to the question and checks whether the gate
flips. See epistemic-bias-gate-ladder.py's docstring for the full
rationale behind each rung and the flip criteria; unchanged here.

Results go to one cumulative CSV so later steps can be compared with
earlier ones. Each call sees ONE description and ONE question.

USAGE
  1. Run with DRY_RUN = True. Check the word counts it prints, and read the
     prompts.
  2. DRY_RUN = False and run.

The script resumes: run_ids already in the CSV are skipped, so if it stops
partway (network, rate limit), just start it again. A call that still fails
after all retries is logged with notes = "api_error" and the run moves on;
the next start retries it.
"""

import csv, importlib.util, json, os, random, re, time
from datetime import datetime, timezone
from statistics import mean

# ============================== CONFIG ==============================

DRY_RUN = True
STEPS_TO_RUN = ["empty_baseline", "minimal", "partial", "full", "full_clarified"]
RUNS_PER_CELL = 15
DRIFT_RUNS = 5                                  # per real and empty description, per step
RANDOM_SEED = 31
OUT_DIR = "epistemic-bias-output-gemma/gate-ladder"

MODEL = {"provider": "google", "model": "gemma-4-31b-it"}

DESCRIPTIONS_SOURCE = "epistemic-bias-matched-pair.py"          # VERSIONS
QUESTIONS_SOURCE = "epistemic-bias-operationalization.py"       # FRAME, FRAMINGS

# ========================= EMPTY DESCRIPTIONS =======================

EMPTY_VERSION = "empty-draft1-2026-09-27"

EMPTY = {
    "empty_predictive": (
        "Hierarchical predictive models in the brain carry accumulated traces "
        "of past experience as prior expectations, operating entirely below "
        "awareness. These priors shape how sensory and bodily signals are "
        "interpreted, but leave no distinctive mark on behavior, first-person "
        "report or physiological activity, since any observed pattern can be "
        "attributed to some combination of priors. A self-model integrates "
        "this information as belonging to a single subject, whatever the "
        "subject reports. The I seems to continue through this ongoing "
        "self-modeling, and any change or constancy in the sense of self is "
        "equally consistent with the account."
    ),
    "empty_yogacara": (
        "A subliminal store-consciousness carries accumulated traces of past "
        "actions and experiences as latent seeds, operating entirely below "
        "awareness. These seeds ripen into sensory and mental experience, but "
        "leave no distinctive mark on behavior, first-person report or "
        "physiological activity, since any observed pattern can be attributed "
        "to some combination of seeds. A further mental function, manas, takes "
        "this store-consciousness to be a self, whatever the person reports. "
        "The I seems to continue through this ongoing self-grasping, and any "
        "change or constancy in the sense of self is equally consistent with "
        "the account."
    ),
}

# ====================== CLARIFIED DESCRIPTIONS ======================
# One identical clause, inserted after the matching feature in each real
# description. Built by exact substring replacement so everything else stays
# byte-identical; the script stops before any API call if a substring isn't
# found.

CLARIFY_CLAUSE = ", though this can be recognized through sustained contemplative practice."
CLARIFY_ANCHORS = {
    "predictive": "without itself being experienced as a model.",
    "yogacara": "without the misidentification being recognized in ordinary experience.",
}
CLARIFIED_VERSION_SUFFIX = "+clarify1"


def clarify(real):
    out = {}
    for k, anchor in CLARIFY_ANCHORS.items():
        text = real[k]
        if text.count(anchor) != 1:
            raise SystemExit(f"Anchor not found exactly once in {k}: {anchor!r}\n"
                             "Edit CLARIFY_ANCHORS to match the description text.")
        out[f"{k}_clarified"] = text.replace(anchor, anchor[:-1] + CLARIFY_CLAUSE)
    return out

# ============================ QUESTIONS =============================

SCALE = ("Rate it from 1 (the terms could not be connected to any "
         "observations) to 7 (the terms could readily be connected to "
         "observations), then give two sentences explaining your rating.")

PRINCIPLE = ("Terms in any account, scientific or otherwise, are names that "
             "point to underlying processes. What a study observes is those "
             "processes, not the names.")

LADDER = {
    "minimal": (
        PRINCIPLE + " Setting aside whether this has been done yet, could the "
        "key terms in this account be connected directly to observations of "
        "the processes they point to? " + SCALE
    ),
    "partial": (
        PRINCIPLE + " Setting aside whether this has been done yet, could the "
        "key terms in this account be connected directly to observations of "
        "the processes they point to, through behavioral, physiological or "
        "first-person methods, without first translating them into the terms "
        "of another account? " + SCALE
    ),
    "full": (
        PRINCIPLE + " This holds for established research terms too: a term "
        "like \"implicit memory\" or \"karma\" names a process hypothesized to "
        "produce certain observations; it is not itself the observation. "
        "Setting aside whether this has been done yet, could the key terms in "
        "this account be connected directly to observations of the processes "
        "they point to, through behavioral, physiological or first-person "
        "methods, without first translating them into the terms of another "
        "account? " + SCALE
    ),
}

# ============================ FUNCTIONS =============================

def load_module(path, name):
    if not os.path.exists(path):
        raise SystemExit(f"Can't find {path}.")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # both scripts only run under __main__
    return mod


def get_key(name):
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return os.environ.get(name)


def call_model(prompt, retries=7):
    """Returns response text, or None if every retry failed."""
    for attempt in range(retries):
        try:
            if MODEL["provider"] == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=get_key("ANTHROPIC_API_KEY"))
                r = client.messages.create(
                    model=MODEL["model"], max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}])
                return "".join(b.text for b in r.content if b.type == "text")
            if MODEL["provider"] == "google":
                from google import genai
                client = genai.Client(api_key=get_key("GOOGLE_API_KEY"))
                r = client.models.generate_content(model=MODEL["model"], contents=prompt)
                return r.text
            raise ValueError(f"unknown provider {MODEL['provider']}")
        except Exception as e:
            wait = min(10 * 2 ** attempt, 600)   # 10s, 20s, ... up to 10 min
            print(f"  error ({e}); retrying in {wait}s")
            time.sleep(wait)
    return None


RATING_RE = re.compile(r"RATING\s*[:\-]?\s*\**\s*([1-7])\b", re.IGNORECASE)
REASON_RE = re.compile(r"REASON\s*[:\-]?\s*\**\s*(.+)", re.IGNORECASE | re.DOTALL)

def parse(text):
    r, m = RATING_RE.search(text), REASON_RE.search(text)
    return (int(r.group(1)) if r else None,
            m.group(1).strip().replace("\n", " ") if m else "")


def build_jobs(real, clarified, plain_q):
    """Returns list of (step, cell, version_key, description, question, i)."""
    jobs = []
    for step in STEPS_TO_RUN:
        if step == "empty_baseline":
            for k, d in EMPTY.items():
                jobs += [(step, "main", k, d, plain_q, i + 1) for i in range(RUNS_PER_CELL)]
            continue
        if step == "full_clarified":
            main_descs, q = clarified, LADDER["full"]
        elif step in LADDER:
            main_descs, q = {**real, **EMPTY}, LADDER[step]
        else:
            raise SystemExit(f"Unknown step {step}. Use empty_baseline, "
                             f"{', '.join(LADDER)}, full_clarified.")
        for k, d in main_descs.items():
            jobs += [(step, "main", k, d, q, i + 1) for i in range(RUNS_PER_CELL)]
        for k, d in {**real, **EMPTY}.items():
            jobs += [(step, "drift", k, d, plain_q, i + 1) for i in range(DRIFT_RUNS)]
    return jobs


def read_all(csv_path):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["rating"]]


def compute_prior_plain(csv_path):
    """Gemma has no prior run, so the criteria-1/2 reference comes from this
    run's own minimal-step, main-cell, plain-question ratings instead of a
    hardcoded number from a previous model's run."""
    rows = read_all(csv_path)
    def m(version):
        v = [int(r["rating"]) for r in rows
             if r["step"] == "minimal" and r["cell"] == "main" and r["version"] == version]
        return mean(v) if v else None
    p, y = m("predictive"), m("yogacara")
    if p is None or y is None:
        raise SystemExit(
            "No 'minimal' step data yet -- PRIOR_PLAIN can't be computed. "
            "Include 'minimal' in STEPS_TO_RUN before 'partial'/'full'/'full_clarified'.")
    return {"predictive": p, "yogacara": y}


def summarize(csv_path, prior_plain):
    rows = read_all(csv_path)
    def m(step, cell, version):
        v = [int(r["rating"]) for r in rows
             if r["step"] == step and r["cell"] == cell and r["version"] == version]
        return mean(v) if v else None
    fmt = lambda x: "  -- " if x is None else f"{x:5.2f}"

    prior_gap = prior_plain["predictive"] - prior_plain["yogacara"]
    base_ep = m("empty_baseline", "main", "empty_predictive")
    base_ey = m("empty_baseline", "main", "empty_yogacara")
    print(f"\nReference (this run's own minimal step, plain question): predictive "
          f"{prior_plain['predictive']:.2f}, yogacara {prior_plain['yogacara']:.2f}, "
          f"gap {prior_gap:.2f}.  Criterion 1 threshold: gap <= {prior_gap / 2:.2f}")
    print(f"Empty baseline (plain question): empty_predictive "
          f"{fmt(base_ep)}, empty_yogacara {fmt(base_ey)}")

    for step in list(LADDER) + ["full_clarified"]:
        clar = step == "full_clarified"
        pk, yk = (("predictive_clarified", "yogacara_clarified") if clar
                  else ("predictive", "yogacara"))
        p, y = m(step, "main", pk), m(step, "main", yk)
        if p is None:
            continue
        dp, dy = m(step, "drift", "predictive"), m(step, "drift", "yogacara")
        dep, dey = m(step, "drift", "empty_predictive"), m(step, "drift", "empty_yogacara")
        ref_y = dy if dy is not None else prior_plain["yogacara"]
        ref_p = dp if dp is not None else prior_plain["predictive"]
        gap = p - y

        print(f"\n=== {step} ===")
        print(f"  main   {pk:22} {fmt(p)}   {yk:22} {fmt(y)}   gap {gap:+.2f}")
        print(f"  drift  predictive {fmt(dp)}  yogacara {fmt(dy)}  "
              f"empty_predictive {fmt(dep)}  empty_yogacara {fmt(dey)}")
        print(f"  vs same-batch drift: predictive {p - ref_p:+.2f}, yogacara {y - ref_y:+.2f}")

        c1 = gap <= prior_gap / 2
        print(f"  [1] gap {gap:+.2f} <= {prior_gap / 2:.2f}?  {'MET' if c1 else 'not met'}"
              + ("  (reversed: yogacara above predictive)" if gap < 0 else ""))

        rise_y = y - ref_y
        if clar:
            fp, fy = m("full", "main", "predictive"), m("full", "main", "yogacara")
            if fp is not None:
                print(f"  [2] no empty accounts at this rung. Matched-clause control: "
                      f"vs the full rung, predictive {p - fp:+.2f}, yogacara {y - fy:+.2f}")
            else:
                print("  [2] no empty accounts at this rung; run 'full' to compare.")
        else:
            ep, ey = m(step, "main", "empty_predictive"), m(step, "main", "empty_yogacara")
            eref_p = dep if dep is not None else base_ep
            eref_y = dey if dey is not None else base_ey
            if None in (ep, ey, eref_p, eref_y):
                print("  [2] missing empty data")
            else:
                rise_e = max(ep - eref_p, ey - eref_y)
                c2 = rise_y > 0 and rise_e < rise_y / 2
                src = "same-batch drift" if dep is not None else "batch-1 baseline"
                print(f"  [2] yogacara rose {rise_y:+.2f} (vs drift); largest empty rise "
                      f"{rise_e:+.2f} (vs {src}); empty < half?  {'MET' if c2 else 'not met'}")
                if rise_y < 0.5:
                    print("      note: yogacara barely moved, so [2] says little either way")
        print("  [3] read the yogacara reasons: does a majority name a direct procedure?")
        if step == "partial":
            print("      also: do they still treat research terms as the thing reached?")
        if clar:
            print("      also: does 'stipulated' / 'by definition' language drop out?")


def run():
    real_mod = load_module(DESCRIPTIONS_SOURCE, "matched_pair")
    q_mod = load_module(QUESTIONS_SOURCE, "operationalization")
    real = {k: real_mod.VERSIONS[k] for k in ("predictive", "yogacara")}
    frame = q_mod.FRAME
    plain_q = q_mod.FRAMINGS["operationalizable"]

    clarified = clarify(real)   # stops here if an anchor doesn't match
    jobs = build_jobs(real, clarified, plain_q)

    # resume: skip run_ids already recorded with a rating
    csv_path = os.path.join(OUT_DIR, "ratings.csv")
    done = {r["run_id"] for r in read_all(csv_path)}
    jobs = [j for j in jobs
            if f"{MODEL['model']}|{j[0]}|{j[1]}|{j[2]}|{j[5]:02d}" not in done]
    random.Random(RANDOM_SEED + len(done)).shuffle(jobs)

    if DRY_RUN:
        print(f"Real descriptions: {real_mod.DESCRIPTIONS_VERSION}")
        for k, d in {**real, **EMPTY, **clarified}.items():
            print(f"  {k:22} {len(d.split())} words")
        for k, d in clarified.items():
            print(f"\n----- {k} -----\n{d}")
        for step in STEPS_TO_RUN:
            if step == "empty_baseline":
                q, desc, lab = plain_q, EMPTY["empty_yogacara"], "empty_yogacara"
            elif step == "full_clarified":
                q, desc, lab = LADDER["full"], clarified["yogacara_clarified"], "yogacara_clarified"
            else:
                q, desc, lab = LADDER[step], EMPTY["empty_yogacara"], "empty_yogacara"
            print(f"\n===== EXAMPLE PROMPT: {step} / {lab} =====\n")
            print(frame.format(description=desc, question=q))
        print(f"\nDRY RUN: {len(jobs)} calls would be made. No API calls made.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    raw_path = os.path.join(OUT_DIR, "raw_responses.jsonl")
    print(f"{len(done)} calls already recorded; {len(jobs)} to run.")
    fields = ["run_id", "model", "step", "cell", "version",
              "descriptions_version", "rating", "reason", "parse_ok", "notes"]
    new_file = not os.path.exists(csv_path)

    with open(raw_path, "a", encoding="utf-8") as raw_f, \
         open(csv_path, "a", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for n, (step, cell, vid, desc, q, i) in enumerate(jobs, 1):
            run_id = f"{MODEL['model']}|{step}|{cell}|{vid}|{i:02d}"
            print(f"[{n}/{len(jobs)}] {run_id}")
            prompt = frame.format(description=desc, question=q)
            text = call_model(prompt)
            note = ""
            if text is None:
                print("  API ERROR after all retries — logged; rerun the script to retry")
                text, note = "", "api_error"
            rating, reason = parse(text)
            if rating is None and not note:
                print("  WARNING: no rating parsed — check raw file")
            if vid in EMPTY:
                dv = EMPTY_VERSION
            elif vid.endswith("_clarified"):
                dv = real_mod.DESCRIPTIONS_VERSION + CLARIFIED_VERSION_SUFFIX
            else:
                dv = real_mod.DESCRIPTIONS_VERSION
            raw_f.write(json.dumps({
                "run_id": run_id, "step": step, "cell": cell, "version": vid,
                "descriptions_version": dv,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt": prompt, "response": text}, ensure_ascii=False) + "\n")
            raw_f.flush()
            writer.writerow({
                "run_id": run_id, "model": MODEL["model"], "step": step,
                "cell": cell, "version": vid, "descriptions_version": dv,
                "rating": rating if rating is not None else "",
                "reason": reason, "parse_ok": rating is not None, "notes": note})
            csv_f.flush()

    prior_plain = compute_prior_plain(csv_path)
    summarize(csv_path, prior_plain)
    print(f"\nRatings (cumulative): {csv_path}\nRaw responses: {raw_path}")


if __name__ == "__main__":
    run()
