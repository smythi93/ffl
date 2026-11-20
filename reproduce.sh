#!/bin/bash
# Recompute every number, table and figure of the manuscript from the shipped
# artifacts/ and vectors/, without re-running instrumentation.
#
#   ./reproduce.sh            # about 4 h on one machine; sensitivity is the long step
#   WORKERS=8 ./reproduce.sh
#
# Order matters: reweight rebuilds every weighting condition in artifacts/ from
# the importances and feature-to-line maps in vectors/, and everything after it
# reads those weights.
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
W=${WORKERS:-4}
export PYTHONPATH=src
METHODS=baseline,rf,alt,neg,perm,pos,hybrid,failfirst,adaptive,fam_coverage,fam_data_flow,fam_value,fam_condition,fam_exception

$PY run_protocol.py reweight -w "$W"                       # weights from vectors/
$PY run_protocol.py collect --methods "$METHODS" -w "$W"   # tidy outcome table
$PY run_protocol.py collect --methods baseline,rf --product prenorm --tag prenorm -w "$W"  # combination-rule probe
$PY run_protocol.py checks                                 # data-quality checks, incl. line-map alignment
$PY run_protocol.py feasibility                            # runtime, timeouts, exclusions; keeps the shipped
                                                           # study/feasibility.csv when the run logs are absent
$PY run_protocol.py categories                             # family importance shares, density
$PY run_protocol.py signprobe                              # Naish2 sign behaviour of the blend
$PY run_protocol.py analyze                                # the pre-registered analysis plan
$PY run_protocol.py analyze --tag prenorm
# Not registered: feature-vector inputs of the cross-project ranker (feature
# spectrum, nearest-neighbour contrast). posthoc adds their weights to the
# artifacts and must follow reweight.
$PY run_protocol.py posthoc -w "$W"
# Not registered: cross-project learning to rank, leave one project out.
# traces/ holds the weightings derived from the ordered traces (export-traces).
$PY run_protocol.py transplant traces -w "$W"
$PY run_protocol.py ltr -w "$W"                            # pre-fixed seed, all variants
for s in 1 2 3 4 5; do                                     # one after another: each rewrites artifacts/
  $PY run_protocol.py ltr -w "$W" --variants vec,trace,full --seed $s
done
$PY run_protocol.py ltr -w "$W" --variants sfl,neg,vec,trace,full --ensemble
LTR=baseline,lex_ltr_sfl,lex_ltr_neg,lex_ltr_perm,lex_ltr_vec,lex_ltr_trace,lex_ltr_full
for v in sfl neg vec trace full; do LTR=$LTR,lex_ltr_${v}_ens; done
for s in 1 2 3 4 5; do LTR=$LTR,lex_ltr_vec_s$s,lex_ltr_trace_s$s,lex_ltr_full_s$s; done
$PY run_protocol.py collect --methods "$LTR" --tag ltr -w "$W"
FFL_ANALYSIS_WORKERS="$W" $PY run_protocol.py analyze --tag ltr
$PY run_protocol.py ltr-robustness -w "$W"
$PY run_protocol.py sensitivity -w "$W"                    # lambda / epsilon / aggregation / threshold
$PY run_protocol.py tables                                 # paper_out/: tables, macros, figures
