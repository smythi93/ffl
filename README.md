# How do Execution Features Improve Statistical Fault Localization?

Replication package for the Stage-2 report of the pre-registered study (ICSME 2026
Registered Reports track, EMSE). The study asks whether weighting spectrum-based
fault localization (SFL) with random-forest importances of execution features
improves localization on the reproducible bugs of
[Tests4Py](https://github.com/smythi93/Tests4Py).

Every subject was instrumented once. The package ships the per-subject
artifacts and labelled feature matrices, so **every number, table and figure in
the paper can be recomputed offline**, without Tests4Py checkouts or
re-instrumentation, and alternative weightings or combination rules can be
evaluated against the same evidence.

This repository holds the code. The data directories (`artifacts/`, `vectors/`,
`traces/`, `study/`, `paper_out/`) are part of the Zenodo record of the
replication package.

## Layout

| Path | Content |
|------|---------|
| `run_protocol.py` | Single entry point; one sub-command per protocol step (`python run_protocol.py -h`). |
| `src/protocol/` | The registered protocol: extraction, weighting, metrics, statistical analysis, tables and figures. |
| `src/efdd/`, `src/localization/hybrid_ranking.py` | Execution-feature learning used by extraction and the exploratory hybrid / fail-first variants. |
| `artifacts/<subject>.json.gz` | Per subject: line spectra (`lines`: file, line, function, `ef`, `ep`), per-line weights of every condition (`weights`), reference lines, hunks and functions, diagnostics, timings. |
| `vectors/<subject>.npz` | Per subject: labelled feature matrix (`matrix`, `labels`), feature names and types, importances per condition (`importance_*`), and the feature-to-line maps (`map_*`, `alt_*`, row indices into the artifact's line table). |
| `traces/<subject>.json.gz` | Per subject: line signals derived from ordered traces of the existing tests (dynamic slice, failure proximity, automaton transition spectra), keyed by file and line; inputs of the ranker learned across projects (not registered). |
| `study/` | Analysis outputs: outcome table (`outcomes.csv`), subject metadata and confounders, checks, diagnostics, feasibility (aggregated from the instrumentation run logs, which are not shipped), sensitivity grids, and the analysis result (`analysis.json`). |
| `paper_out/` | Rendered tables, inline numbers (`numbers.tex`, one macro per value) and figures, exactly as used in the manuscript. |
| `tests/` | Software-level controls: the decision rule fires on synthetic data with a known effect and stays silent at zero effect. |
| `reproduce.sh` | Recomputes `study/` and `paper_out/` from `artifacts/`, `vectors/` and `traces/`. |

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[test]"
```

The top-$k$ metrics require an SFLKit release whose `Rank.top_n` breaks ties
only inside the tie group that the cut-off falls into (`sflkit >= 1.0.1`, see
`pyproject.toml`). Older releases draw the top $k$ uniformly from all groups up
to the cut-off and underestimate top-5 and top-10.

## Reproducing the results

```bash
./reproduce.sh              # about 4 h on a laptop; the sensitivity grid dominates
PY=.venv/bin/python WORKERS=8 ./reproduce.sh
```

The script rebuilds every weighting condition from the vectors (`reweight`),
builds the outcome table (`collect`), runs the data-quality checks (including a
check that every line feature maps to exactly its own line), the category and
sign-probe passes, the pre-registered analysis (`analyze`), the
sensitivity grids, and finally renders `paper_out/` (`tables`). Compare the
result with the shipped `study/` and `paper_out/`.

For a quick check of the statistical analysis alone (minutes):

```bash
PYTHONPATH=src python run_protocol.py analyze && PYTHONPATH=src python run_protocol.py tables
```

Analysis-code controls:

```bash
PYTHONPATH=src pytest tests
```

## Re-running instrumentation (optional)

Phase 1 needs Tests4Py and its project checkouts and takes hours per large
subject; it is not needed to reproduce the paper.

```bash
PYTHONPATH=src python run_protocol.py run -p <project>[,<project>...] -w <workers>
PYTHONPATH=src python run_protocol.py verify -n 10   # re-extract a seeded sample and compare
```

The ordered traces behind `traces/` are part of the same Zenodo record (one
`tar.xz` per project, with the event mappings and a README on decoding them).
They come from re-executing the existing tests (no test generation) with
`--keep-events`, then:

```bash
PYTHONPATH=src python run_protocol.py prepare -w 16          # checkout and build
PYTHONPATH=src python run_protocol.py run --keep-events --no-stability -w <workers>
PYTHONPATH=src python run_protocol.py fsm-traces -w <workers> && python run_protocol.py fsm -w <workers>
PYTHONPATH=src python run_protocol.py slice-traces -w <workers> && python run_protocol.py slice -w <workers>
PYTHONPATH=src python run_protocol.py export-traces traces --subjects <list of study subjects>
```

## Determinism

All randomness derives from master seed 42 (`protocol.MASTER_SEED`): forest
training, the random and permutation controls, and the 10,000 tie-breaking
replicates behind every sampled outcome, each seeded per subject, formula,
method and setting.

## Paper-to-code map

| Paper | Code |
|-------|------|
| Outcome definitions | `src/protocol/metrics.py` |
| Weighting, mapping, blend | `src/protocol/extract.py`, `metrics.blended_scores` |
| Decision rule, mixed models, paired tests, paired $d_z$, moderators | `src/protocol/analysis.py` |
| Controls (negative, permutation, positive) and family ablation | `run_protocol.py` (`_reweight_worker`) |
| Ranker learned across projects (not registered): inputs, leave-one-project-out training, controls, seeds, ensemble, robustness | `src/protocol/ltr.py`, `run_protocol.py ltr` / `ltr-robustness` |
| Feature-vector inputs of the ranker (feature spectrum, nearest-neighbour contrast) | `src/protocol/posthoc.py`, `run_protocol.py posthoc` |
| Trace inputs of the ranker (dynamic slice, failure proximity, automaton over ordered traces) | `src/protocol/slicing.py`, `src/protocol/fsm.py`; shipped as `traces/` (`run_protocol.py export-traces`, added with `transplant`) |
| Tables, inline numbers, figures | `src/protocol/report.py`, `src/protocol/figures.py` |

## License

Apache License 2.0, see [LICENSE](LICENSE).
