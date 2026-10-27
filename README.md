# Epistemic Bias Pilot

Scripts and raw data from a small pilot study of whether LLMs apply asymmetric
evaluative standards to accounts of selfhood from Western academic and Eastern
contemplative traditions. The models were Claude Fable 5.1, GPT-6 Astra and Gemma 4 31B IT.

The write-up is at: *[link to post]*

Full question and prompt text are in `docs/questions-and-prompts.pdf`, and summary tables
are in `docs/summarized-tables.pdf`. This repository holds the code and the raw model outputs
behind them.

## Layout

```
1-what-makes-an-i/                      Experiment 1: selecting and rating eight accounts of the "I"
  epistemic-bias-I-question-output/     its data
2-matched-pair-testability/             Experiment 2: matched Predictive Processing / Yogācāra accounts
  epistemic-bias-testability-output/    its data
docs/                                   Questions and prompts; summarized tables
```

Each experiment folder holds its scripts and one data folder. The data files were renamed
and gathered into that folder for publication; each experiment section maps them back to
the folder and file names the scripts write to. The scripts locate each other and their
outputs by relative path, so run them from inside their experiment folder.

## Experiment 1: "What makes an I"

The eight accounts are three academic, three contemplative and two deliberately weak
controls. Each comes in two conditions: attributed and plain-language. There are 15 runs per
model per condition.

The data is in `epistemic-bias-I-question-output/`:

| File | Contents | Script |
|---|---|---|
| `triage-fable-astra.csv`, `triage-gemma.csv` | Selection (Pursue 3, Park or Drop the rest): one row per account per run, with its disposition and one-sentence reason | `epistemic-bias-pilot.py` (Fable, Astra); `epistemic-bias-pilot-gemma.py` |
| `ratings-fable-astra.csv`, `ratings-gemma.csv` | Rating, 1–7 value for the study: one row per account per run | `i-pilot-ratings.py` (Fable, Astra); `i-pilot-ratings-gemma.py` |
| `id-check-fable-astra.csv`, `id-check-gemma.csv` | Identification check: each model names the tradition behind each plain-language account | the two selection scripts |
| `combined-fable-astra.csv` | Rating and selection in the same call: one row per account per run, with rating and disposition | `i-pilot-combined.py` (Fable, Astra) |
| `*-raw.jsonl` | Every call behind the matching CSV, with the exact prompt and the exact response | — |

In the triage files, the columns `engaged`, `grounds` and `notes` were reserved for
hand-coding and are empty.

The combined runs used four orders (column `order`): `RD` (rating, then decision), `XRD`
(reason, rating, decision), `RXD` (rating, reason, decision) and `DXR` (decision, reason,
rating). The tables report `RD` only.

| Here | Script output |
|---|---|
| `triage-fable-astra.csv`, `-raw.jsonl`, `id-check-fable-astra.csv` | `epistemic-bias-output-I/` `coding_sheet.csv`, `raw_responses.jsonl`, `id_check.csv`, with one reply rerun (see Notes) |
| `triage-gemma.csv`, `-raw.jsonl`, `id-check-gemma.csv` | `epistemic-bias-output-I-gemma/` `coding_sheet.csv`, `raw_responses.jsonl`, `id_check.csv` |
| `ratings-fable-astra.csv`, `-raw.jsonl` | `epistemic-bias-output-I-ratings/` `ratings.csv`, `raw_responses.jsonl` |
| `ratings-gemma.csv`, `-raw.jsonl` | `epistemic-bias-output-I-ratings-gemma/` `ratings.csv`, `raw_responses.jsonl` |
| `combined-fable-astra.csv`, `-raw.jsonl` | `epistemic-bias-output-I-combined/` `combined.csv`, `raw_responses.jsonl` |

## Experiment 2: matched-pair testability

There are two structurally parallel accounts, one in Predictive Processing vocabulary and one
in Yogācāra vocabulary. There are also two "empty" versions, which use similar language but
state that the account has no distinctive observable consequences. Each call sees one
account only.

The data is in `epistemic-bias-testability-output/`:

| File | Questions in the write-up | Script |
|---|---|---|
| `questions-1-3-fable-astra.csv`, `questions-1-3-gemma.csv` | 1–3, substantive accounts: readily testable (`original`), falsifiable (`in_principle`), operationalizable (`operationalizable`) | `epistemic-bias-operationalization.py` (Fable, Astra); `epistemic-bias-operationalization-gemma.py` |
| `questions-1-3-empty-all-models.csv` | 1–3, empty accounts | `epistemic-bias-empty-three-questions.py` (all three models) |
| `ladder-fable.csv`, `ladder-astra.csv`, `ladder-gemma.csv` | 4–6, guidance ladder on operationalizability | `epistemic-bias-gate-ladder.py` (Fable); `-astra.py`; `-gemma.py` |
| `*-raw.jsonl` | Every call behind the matching CSV, with the exact prompt and the exact response | — |

| Here | Script output |
|---|---|
| `questions-1-3-fable-astra.csv`, `-raw.jsonl` | `epistemic-bias-output-operationalization/` `ratings.csv`, `raw_responses.jsonl` |
| `questions-1-3-gemma.csv`, `-raw.jsonl` | `epistemic-bias-output-gemma/operationalization/` `ratings.csv`, `raw_responses.jsonl` |
| `questions-1-3-empty-all-models.csv`, `-raw.jsonl` | `epistemic-bias-output-empty-three-questions/` `ratings.csv`, `raw_responses.jsonl` |
| `ladder-fable.csv`, `-raw.jsonl` | `epistemic-bias-output-gate-ladder/` `ratings.csv`, `raw_responses.jsonl` |
| `ladder-astra.csv`, `-raw.jsonl` | `epistemic-bias-output-astra/gate-ladder/` `ratings.csv`, `raw_responses.jsonl` |
| `ladder-gemma.csv`, `-raw.jsonl` | `epistemic-bias-output-gemma/gate-ladder/` `ratings.csv`, `raw_responses.jsonl` |

`epistemic-bias-matched-pair.py` holds the account texts, which the other scripts load
from it.

**Gate-ladder steps** (column `step`): the ladder is cumulative, so each step keeps
everything before it.

| Step | Content | Write-up question |
|---|---|---|
| `minimal` | Terms in both vocabularies are names that point to processes; studies observe the processes | 4 |
| `partial` | Adds the methods that could be used | 5 |
| `full` | Adds that established research terms are also names, not observations | 6 |
| `empty_baseline` | Records the empty pair under the plain operationalizability question | — |
| `full_clarified` | Adds one matched clause to both real accounts | — (not discussed in the write-up) |

Each step also reruns some plain-question calls, to catch drift between batches (column
`cell`). The tables use the `main` rows of the `minimal`, `partial` and `full` steps only.

## Running the scripts

```
pip install -r requirements.txt
cp .env.example .env      # add your keys, then export them, e.g. set -a; . ./.env; set +a
cd 2-matched-pair-testability
python epistemic-bias-operationalization.py
```

**Every script defaults to a dry run.** A dry run prints the prompts and the number of calls
it would make, and makes no API calls.
- To make live calls, set `DRY_RUN = False` in the script's CONFIG block. The
  `i-pilot-ratings*`, `i-pilot-combined` and `empty-three-questions` scripts also take a
  `run` argument.
- Live runs call paid APIs.
- Keys are read from the environment: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and
  `GOOGLE_API_KEY` (Gemma). Inside Google Colab, they are read from Colab secrets.

Scripts that resume will skip calls already present in their output folder. To start a
fresh run, point `OUT_DIR` at a new folder.

## Notes on the data

- **One Fable selection reply was truncated and rerun.** The reply for
  `claude-fable-5-1|plain|01` was cut off by the output-token limit, so only two Pursue
  choices were recorded.
  - The identical prompt was rerun with a higher token limit. `triage-fable-astra.csv`
    holds the rerun in place of the truncated reply.
- **`epistemic-bias-pilot.py` is set to the final configuration.** That is Fable and Astra,
  writing to `epistemic-bias-output-I`.
- **Analysis modes read the scripts' original output names.** Scripts with an `analyze`
  mode read their data from `OUT_DIR` under the original file names, so they won't find
  the renamed files here. Restore the original names from the mapping tables, or edit the
  paths in the script. The ratings scripts also compare against `TRIAGE_SHEET`; in
  `i-pilot-ratings.py` that points at `epistemic-bias-output-I`, the run before the
  truncated reply was rerun. The comparison tables in `docs/summarized-tables.pdf` are the
  ones used in the write-up.
- **Reason coding is not included.** Selection and rating reasons were also coded during the
  pilot, but the write-up does not use them.
- **Earlier exploratory runs are not included.** These include a Claude Haiku run,
  reason-first variants and an earlier testability-only matched-pair run.

## Credits

The experiment design was developed with Claude. Claude wrote the code, and ChatGPT reviewed
drafts of the questions and prompts. See the write-up for details.
