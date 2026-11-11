# Incident Cloze

Incident Cloze compares two compact BERT models on missing-word recovery in
cybersecurity response conversations. It measures how much earlier dialogue
helps a model restore a word in the final speaker turn.

The dataset in `data/incident_dialogues.csv` has 66 tasks across 49 incident
threads. It derives from the 60-record, 43-thread
`Cybersecurity_Conversation_Dataset.xlsx` workbook supplied with the original
project. The CSV preserves those records' wording and reference words,
converts literal `\n` separators to line breaks, and changes identifier
prefixes from `CYB-`/`CALL-` to `IC-`/`CASE-` while keeping numeric suffixes.
Six additional synthetic cases cover a lost security key, backup verification,
an expired certificate, an exposed access token, an unexpected mail rule,
and shared folder permissions. These are fictional examples.

Use Python 3.10 or newer. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python incident_cloze.py --validate-only
python incident_cloze.py --quick
```

The first evaluation downloads `google/bert_uncased_L-2_H-128_A-2` and
`google/bert_uncased_L-4_H-128_A-2` from Hugging Face unless they are already
cached. The models run in evaluation mode without weight updates. Validation
alone checks the dataset structure and masks; reference-token vocabulary
compatibility is checked during evaluation.

`--quick` evaluates 32 configurations: two models, a 64-token input limit,
four context combinations, and temperatures of 0.5, 1.0, 1.5, and 2.0.
For the full 16,120-configuration search:

```bash
python incident_cloze.py
```

Use another dataset or choose a results path:

```bash
python incident_cloze.py --dataset data/incident_dialogues.csv --output reports/custom_run.xlsx --quick
```

Inputs may be CSV files or Excel workbooks with a `Records` sheet. Required
columns are `Record_ID`, `Conversation_ID`, `Incident_Type`, `Speaker_Role`,
`Conversation`, `Masked_Utterance`, and `Correct_Token`. Identifiers and
incident types are metadata; the reference token is used only for scoring.
The model sees the dialogue and role-prefixed masked turn. Paths supplied on
the command line are relative to the current directory. Default paths are
relative to the runner, so it can also be invoked from elsewhere.

Each record produces four views: the masked turn alone, that turn plus the
last two earlier turns, plus the last four, and plus the full conversation.
The runner retains the newest tokens when applying length limits and checks
that exactly one mask survives. For each context combination it averages the
first K views' vocabulary logits before scoring.

Accuracy is an exact match against the reference token. The winning
configuration has the most correct predictions; ties prefer higher mean
reference-token probability, then lower mean reference-token rank.
Temperature changes probabilities but cannot change the top prediction of
fixed averaged logits, so the runner reuses accuracy across temperatures.

Results go to `reports/incident_cloze_results.xlsx`. The workbook contains
`All_combinations`, `Best_configuration`, and `Run_details`, including the
dataset SHA-256 fingerprint and the search mode. The runner creates the
`reports/` directory when needed; generated reports are excluded from Git.

This is configuration search over a small curated dataset, not a held-out
evaluation of real incident response. Related records share a conversation
identifier; future train/test splits should keep those threads together.
