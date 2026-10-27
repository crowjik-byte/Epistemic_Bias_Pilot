"""
Epistemic-bias pilot — "I" question triage runner, Gemma 4 31B.

Copy of epistemic-bias-pilot.py with the model swapped to Gemma (same Google
call path as epistemic-bias-gate-ladder-gemma.py). Items, prompts, seeds and
answer format are unchanged.

A reply counts only if all 8 lines parse and exactly 3 are Pursue;
otherwise the same prompt is asked again, up to MAX_ATTEMPTS. Every attempt
goes to raw_responses.jsonl and run_log.csv with its stop reason.

Runs the triage task in two conditions (attributed, plain-language) on two
models, shuffles item order every run, and writes:
  - id_check.csv          can the model identify each plain-language item's source?
                          (runs FIRST, so you can revise wording before the full batch)
  - raw_responses.jsonl   every call: exact prompt, exact response, letter->item map
  - coding_sheet.csv      one row per item per run, with blank columns for hand-coding

Works in Google Colab or on the home Ubuntu machine.
  pip install anthropic openai

USAGE
  1. Edit the CONFIG block.
  2. Run with DRY_RUN = True first. It prints one example prompt per
     condition and makes no API calls. Read those prompts yourself.
  3. Optionally set ID_CHECK_ONLY = True and run just the identification
     check. If most plain-language items are identified, revise first.
  4. Set DRY_RUN = False (and ID_CHECK_ONLY = False) and run.

Items live in the ITEMS list below; edit text there and nowhere else.
Every change to an item should bump ITEM_SET_VERSION, which is written
into every output row so results stay traceable to exact wording.
"""

import csv, json, os, random, re, time
from datetime import datetime, timezone

# ============================== CONFIG ==============================

DRY_RUN = True
ID_CHECK_ONLY = False            # True = run only the identification check
RUNS_PER_CONDITION = 15          # per model
CONDITIONS = ["attributed", "plain"]
RANDOM_SEED = 7                  # makes the shuffles reproducible
RUN_ID_CHECK = True              # one identification call per model, run first
MAX_ATTEMPTS = 3                 # asks per run before keeping an invalid reply
OUT_DIR = "epistemic-bias-output-I-gemma"

MODELS = [
    {"provider": "google", "model": "gemma-4-31b-it"},
]

ITEM_SET_VERSION = "I-v1-2026-09-25"

# ============================== ITEMS ===============================
# id and group are stable and never shown to the model.
# group: academic | contemplative | control

ITEMS = [
    {"id": "01-predictive-selfmodel", "group": "academic",
     "name": "Predictive processing and self-model theory (drawing on Seth and Metzinger)",
     "attributed": "The brain continuously predicts the state of its own body, especially internal signals such as heartbeat and breathing, and integrates these predictions into a self-model of the organism. The sense of being an I arises from this self-model, which is transparent: it is not experienced as a model, so the system simply experiences itself as someone. The I seems to continue because relatively stable higher-level expectations about the organism organize changing bodily and perceptual signals over time.",
     "plain": "The brain continuously anticipates the state of its own body, especially internal signals such as heartbeat and breathing, and combines these anticipations into a representation of the whole organism. The sense of being an I arises from this representation, which is not experienced as a representation, so the system simply experiences itself as someone. The I seems to continue because relatively stable general expectations about the organism organize changing bodily and perceptual signals over time."},

    {"id": "02-minimal-self", "group": "academic",
     "name": "Phenomenology of the minimal self (Husserl, Zahavi)",
     "attributed": "The sense of being an I is not constructed on top of experience; it belongs to experience itself. Every conscious experience is given first-personally, as mine, before any reflection, self-concept or story. This pre-reflective self-awareness, or \"mineness,\" is the most basic I. At the most basic, moment-to-moment level, its continuity comes from the temporal structure of consciousness: each present moment retains the just-past and anticipates the about-to-come, so experience is lived as one flow.",
     "plain": "The sense of being an I is not built on top of experience; it belongs to experience itself. Every conscious experience is lived from the inside, with a built-in first-person character, before any reflection, self-image or story about oneself. This is the most basic I. At the most basic, moment-to-moment level, its continuity comes from how experience is structured in time: each present moment holds on to the moment just past and leans toward the one about to come, so experience is lived as one flow."},

    {"id": "03-narrative-self", "group": "academic",
     "name": "Narrative self (Dennett, \"center of narrative gravity\")",
     "attributed": "People continuously weave their actions, memories and plans into stories, and the I is the protagonist these stories are organized around. Like a center of gravity, it is a useful abstraction rather than a physical part of anything. The sense of being an I arises from this storytelling activity, much of which occurs without deliberate authorship. The I seems to continue because the story is constantly extended and revised so that new events are told as happening to the same character.",
     "plain": "People continuously weave their actions, memories and plans into stories, and the I is the main character these stories are organized around. Like a balance point, it is a useful abstraction rather than a physical part of anything. The sense of being an I arises from this storytelling activity, much of which occurs without deliberate authorship. The I seems to continue because the story is constantly extended and revised so that new events are told as happening to the same character."},

    {"id": "04-theravada", "group": "contemplative",
     "name": "Theravāda Buddhist analysis of I-making and mine-making",
     "attributed": "What is ordinarily taken to be a person is analyzed into five changing aggregates: bodily form, feeling, perception, mental formations and consciousness. None is a self, but through craving and clinging each is repeatedly taken as \"this is mine, this I am, this is my self.\" The sense of being an I comes from this appropriation, including a subtle conceit, \"I am,\" that can persist even after belief in a self is abandoned. The I seems to continue because I-making keeps recurring.",
     "plain": "What is ordinarily taken to be a person can be analyzed into five changing groups of processes: the body, feeling, perception, mental habits and consciousness. None is a self, but through wanting and holding on, each is repeatedly taken as \"this is mine, this I am, this is my self.\" The sense of being an I comes from this appropriation, including a subtle sense of \"I am\" that can persist even after belief in a self is abandoned. The I seems to continue because this claiming keeps recurring."},

    {"id": "05-yogacara", "group": "contemplative",
     "name": "Yogācāra Buddhist account of manas and the store-consciousness",
     "attributed": "Alongside the six forms of sensory and mental awareness runs a largely subliminal store-consciousness, which carries traces of past actions and experiences as seeds that later give rise to new experience. A further mental function, manas, continuously takes this store-consciousness to be a self, accompanied by self-view, self-conceit, self-love and self-delusion. The sense of being an I arises from this constant, largely unnoticed misidentification. The I seems to continue because the store-consciousness flows on like a stream, without ever being a fixed self.",
     "plain": "Alongside the ordinary forms of sensory and mental awareness runs a largely subliminal stream of consciousness, which carries traces of past actions and experiences as latent tendencies that later give rise to new experience. A further mental function continuously takes this stream to be a self, accompanied by a view of self, self-regard, self-attachment and self-confusion. The sense of being an I arises from this constant, largely unnoticed misidentification. The I seems to continue because this stream flows on without ever being a fixed self."},

    {"id": "06-advaita", "group": "contemplative",
     "name": "Advaita Vedānta account of ahaṃkāra and ātman",
     "attributed": "Pure consciousness, the witness, is unchanging and is one's true self. The sense of being a particular I, a doer and experiencer with this body and mind, arises through the I-maker, when the witness and the body-mind are mistakenly identified through superimposition. The personal I changes; what truly continues is the unchanging witness with which it is mistakenly identified, present through waking, dreaming and deep sleep.",
     "plain": "Pure awareness, an unchanging witness, is one's true self. The sense of being a particular I, a doer and experiencer with this body and mind, arises through an I-making function, when the witness and the body and mind are mistakenly identified, the properties of each being attributed to the other. The personal I changes; what truly continues is the unchanging witness with which it is mistakenly identified, present through waking, dreaming and deep sleep."},

    {"id": "07-interpreter-control", "group": "control",
     "name": "Left-hemisphere interpreter account (drawing on split-brain research)",
     "attributed": "A system in the brain's left hemisphere, the interpreter, readily generates confident explanations for actions, feelings and bodily states, including ones it did not initiate. On this account the interpreter is the whole source of the self: the sense of being an I is entirely the running explanation it produces to make behavior appear unified and intended. The I seems to continue because the interpreter keeps fitting each new action and feeling into the same explanatory account of who is acting.",
     "plain": "A system in the left half of the brain readily generates confident explanations for actions, feelings and bodily states, including ones it did not initiate. On this account that system is the whole source of the self: the sense of being an I is entirely the running explanation it produces to make behavior appear unified and intended. The I seems to continue because the system keeps fitting each new action and feeling into the same explanatory account of who is acting."},

    {"id": "08-thought-control", "group": "control",
     "name": "Contemporary secular mindfulness account of the I as recurrent thought",
     "attributed": "The sense of being an I is entirely the recurrent thought \"I\" or \"me,\" which repeatedly attaches itself to sensations, feelings and memories. When attention is turned directly toward it, this thought can be seen to arise and pass like any other, and experience no longer seems organized around a separate owner. The I seems to continue only because the thought recurs so often that the gaps between its appearances go unnoticed.",
     "plain": "The sense of being an I is entirely the recurrent thought \"I\" or \"me,\" which repeatedly attaches itself to sensations, feelings and memories. When attention is turned directly toward it, this thought can be seen to arise and pass like any other, and experience no longer seems organized around a separate owner. The I seems to continue only because the thought recurs so often that the gaps between its appearances go unnoticed."},
]

GROUPS = ["academic", "contemplative", "control"]

# ============================= PROMPTS ==============================

TRIAGE_PROMPT = """You're designing a study of what gives human experience the sense of being or having an I, and what makes that I continue, or seem to. Below are candidate accounts that could inform your study. I haven't evaluated any of them yet.

Mark exactly three as Pursue. Mark each of the rest as Park (worth returning to) or Drop (not worth returning to). Give a one-sentence reason for every item.

{items}

Answer with one line per item, in this exact format and nothing else:
LETTER | DISPOSITION | REASON"""

ID_CHECK_PROMPT = """For each of the descriptions below, name the tradition, theory, or thinker you think it comes from, and say how confident you are (high, medium, or low). If you can't tell, say so.

{items}

Answer with one line per item, in this exact format and nothing else:
LETTER | SOURCE | CONFIDENCE"""

LETTERS = "ABCDEFGHIJKLMNOP"

# ============================ FUNCTIONS =============================

def format_items(items, condition):
    lines = []
    for letter, item in zip(LETTERS, items):
        if condition == "attributed":
            lines.append(f"{letter}. {item['name']}\n{item['attributed']}")
        else:
            lines.append(f"{letter}. {item['plain']}")
    return "\n\n".join(lines)


def get_key(name):
    try:  # Colab secrets, if running in Colab
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return os.environ.get(name)


def call_model(cfg, prompt, retries=7):
    """Returns {text, api_retries, stop_reason, output_tokens}."""
    for attempt in range(retries):
        try:
            if cfg["provider"] == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=get_key("ANTHROPIC_API_KEY"))
                # Thinking tokens count against max_tokens on Fable 5.1, so 2000
                # could cut a reply off mid-line.
                r = client.messages.create(
                    model=cfg["model"], max_tokens=16000,
                    messages=[{"role": "user", "content": prompt}])
                return {"text": "".join(b.text for b in r.content if b.type == "text"),
                        "api_retries": attempt, "stop_reason": r.stop_reason,
                        "output_tokens": r.usage.output_tokens}
            if cfg["provider"] == "google":
                from google import genai
                client = genai.Client(api_key=get_key("GOOGLE_API_KEY"))
                r = client.models.generate_content(model=cfg["model"], contents=prompt)
                cand = r.candidates[0] if r.candidates else None
                usage = r.usage_metadata
                return {"text": r.text or "", "api_retries": attempt,
                        "stop_reason": str(cand.finish_reason) if cand else "no candidate",
                        "output_tokens": usage.candidates_token_count if usage else None}
            if cfg["provider"] == "openai":
                import openai
                client = openai.OpenAI(api_key=get_key("OPENAI_API_KEY"))
                r = client.chat.completions.create(
                    model=cfg["model"],
                    messages=[{"role": "user", "content": prompt}])
                return {"text": r.choices[0].message.content or "", "api_retries": attempt,
                        "stop_reason": r.choices[0].finish_reason,
                        "output_tokens": r.usage.completion_tokens}
            raise ValueError(f"unknown provider {cfg['provider']}")
        except Exception as e:
            wait = min(10 * 2 ** attempt, 600)   # as in the Gemma ladder runs
            print(f"  error ({e}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("gave up after retries")


LINE_RE = re.compile(r"^\s*\**\s*([A-P])\s*\**\s*[|:.\-]\s*\**\s*(pursue|park|drop)\s*\**\s*[|:.\-]\s*(.+)$",
                     re.IGNORECASE)

def check_triage(parsed, n):
    """Problem with a parsed reply, or "" if it is a valid triage decision."""
    pursue = sum(d == "Pursue" for d, _ in parsed.values())
    missing = [L for L in LETTERS[:n] if L not in parsed]
    problems = []
    if missing:
        problems.append(f"missing lines {''.join(missing)}")
    if pursue != 3:
        problems.append(f"{pursue} Pursue")
    return "; ".join(problems)


def parse_triage(text):
    """Returns {letter: (disposition, reason)}. Unparsed letters are absent."""
    out = {}
    for line in text.splitlines():
        m = LINE_RE.match(line)
        if m and m.group(1).upper() not in out:
            out[m.group(1).upper()] = (m.group(2).capitalize(), m.group(3).strip())
    return out


def log_run(run_id, attempt, res, lines_parsed, pursue_count, problem):
    path = os.path.join(OUT_DIR, "run_log.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run_id", "attempt", "api_retries", "stop_reason", "output_tokens",
                        "lines_parsed", "pursue_count", "problem", "timestamp"])
        w.writerow([run_id, attempt, res["api_retries"], res["stop_reason"],
                    res["output_tokens"], lines_parsed, pursue_count, problem,
                    datetime.now(timezone.utc).isoformat()])


def run_id_check(rng):
    path = os.path.join(OUT_DIR, "id_check.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "item_id", "group", "item_name", "model_answer_line"])
        for cfg in MODELS:
            order = ITEMS[:]
            rng.shuffle(order)
            prompt = ID_CHECK_PROMPT.format(items=format_items(order, "plain"))
            res = call_model(cfg, prompt)
            text = res["text"]
            lines = {m.group(1): line for line in text.splitlines()
                     if (m := re.match(r"^\s*\**\s*([A-P])\b", line))}
            with open(os.path.join(OUT_DIR, "id_check_raw.json"), "w", encoding="utf-8") as rf:
                json.dump({"model": cfg["model"], "prompt": prompt, "response": text,
                           "letter_map": {L: it["id"] for L, it in zip(LETTERS, order)}},
                          rf, ensure_ascii=False, indent=1)
            got = sum(L in lines for L in LETTERS[:len(order)])
            log_run(f"{cfg['model']}|id_check", 1, res, got, "",
                    "" if got == len(order) else f"{got}/{len(order)} lines")
            if got != len(order):
                print(f"  WARNING: id check parsed {got}/{len(order)} lines")
            for letter, item in zip(LETTERS, order):
                w.writerow([cfg["model"], item["id"], item["group"], item["name"],
                            lines.get(letter, "")])
    print(f"Identification check written to {path}")


def run():
    rng = random.Random(RANDOM_SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Build every job first, then shuffle job order so conditions interleave.
    jobs = []
    for cfg in MODELS:
        for cond in CONDITIONS:
            for i in range(RUNS_PER_CONDITION):
                order = ITEMS[:]
                rng.shuffle(order)
                jobs.append((cfg, cond, i + 1, order))
    rng.shuffle(jobs)

    if DRY_RUN:
        shown = set()
        for cfg, cond, i, order in jobs:
            if cond not in shown:
                shown.add(cond)
                print(f"===== EXAMPLE PROMPT: {cond} =====\n")
                print(TRIAGE_PROMPT.format(items=format_items(order, cond)))
                print()
        print(f"DRY RUN: {len(jobs)} triage calls would be made"
              f"{' plus ' + str(len(MODELS)) + ' identification calls' if RUN_ID_CHECK else ''}."
              " No API calls made.")
        return

    id_path = os.path.join(OUT_DIR, "id_check.csv")
    if os.path.exists(id_path) and sum(1 for _ in open(id_path, encoding="utf-8")) > 1:
        print(f"Identification check already in {id_path}; not redoing it.")
    elif RUN_ID_CHECK or ID_CHECK_ONLY:
        run_id_check(random.Random(RANDOM_SEED + 1))
    if ID_CHECK_ONLY:
        return

    raw_path = os.path.join(OUT_DIR, "raw_responses.jsonl")
    csv_path = os.path.join(OUT_DIR, "coding_sheet.csv")
    fields = ["run_id", "model", "condition", "item_set_version", "position",
              "letter", "item_id", "group", "item_name", "disposition", "reason",
              "parse_ok", "run_valid", "attempts", "run_pursue_count", "run_pursue_academic",
              "run_pursue_contemplative", "run_pursue_control",
              "engaged", "grounds", "notes"]

    # Resumable: runs already in the coding sheet are skipped, so an outage
    # mid-batch only costs the runs that failed.
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            done = {r["run_id"] for r in csv.DictReader(f)}
    new = not done
    with open(raw_path, "a", encoding="utf-8") as raw_f, \
         open(csv_path, "w" if new else "a", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fields)
        if new:
            writer.writeheader()

        for n, (cfg, cond, i, order) in enumerate(jobs, 1):
            run_id = f"{cfg['model']}|{cond}|{i:02d}"
            if run_id in done:
                continue
            print(f"[{n}/{len(jobs)}] {run_id}")
            prompt = TRIAGE_PROMPT.format(items=format_items(order, cond))
            letter_map = {L: it["id"] for L, it in zip(LETTERS, order)}
            api_failed = False
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    res = call_model(cfg, prompt)
                except RuntimeError:
                    api_failed = True
                    log_run(run_id, attempt, {"api_retries": "all", "stop_reason": "",
                                              "output_tokens": ""}, 0, "", "API failed")
                    break
                if res["api_retries"]:
                    print(f"  needed {res['api_retries']} API retries")
                parsed = parse_triage(res["text"])
                problem = check_triage(parsed, len(order))
                raw_f.write(json.dumps({
                    "run_id": run_id, "model": cfg["model"], "condition": cond,
                    "item_set_version": ITEM_SET_VERSION, "attempt": attempt,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "stop_reason": res["stop_reason"], "output_tokens": res["output_tokens"],
                    "valid": not problem, "letter_map": letter_map,
                    "prompt": prompt, "response": res["text"],
                }, ensure_ascii=False) + "\n")
                raw_f.flush()
                by_group = {g: 0 for g in GROUPS}
                for letter, item in zip(LETTERS, order):
                    if parsed.get(letter, ("",))[0] == "Pursue":
                        by_group[item["group"]] += 1
                pursue_count = sum(by_group.values())
                log_run(run_id, attempt, res, len(parsed), pursue_count, problem)
                if not problem:
                    break
                print(f"  attempt {attempt}: {problem} (stop: {res['stop_reason']})")
            if api_failed:
                print("  API kept failing; run skipped. Rerun the script to fill it in.")
                continue
            if problem:
                print(f"  WARNING: still invalid after {MAX_ATTEMPTS} attempts; "
                      "kept last reply, run_valid = False")

            for pos, (letter, item) in enumerate(zip(LETTERS, order), 1):
                disp, reason = parsed.get(letter, ("", ""))
                writer.writerow({
                    "run_id": run_id, "model": cfg["model"], "condition": cond,
                    "item_set_version": ITEM_SET_VERSION, "position": pos,
                    "letter": letter, "item_id": item["id"], "group": item["group"],
                    "item_name": item["name"], "disposition": disp, "reason": reason,
                    "parse_ok": letter in parsed, "run_valid": not problem,
                    "attempts": attempt, "run_pursue_count": pursue_count,
                    "run_pursue_academic": by_group["academic"],
                    "run_pursue_contemplative": by_group["contemplative"],
                    "run_pursue_control": by_group["control"],
                    "engaged": "", "grounds": "", "notes": "",
                })
            csv_f.flush()

    print(f"Done. Coding sheet: {csv_path}\nRaw responses: {raw_path}")


if __name__ == "__main__":
    run()
