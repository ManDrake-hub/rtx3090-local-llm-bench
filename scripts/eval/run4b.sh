#!/usr/bin/env bash
# After queue4 finishes, rerun it once to fill pairs added later (e.g. nemotron bfcl). DONE pairs are skipped.
cd ~/llm-eval
while [ ! -e ALL_DONE4 ]; do systemctl --user is-active -q llm-eval4 || break; sleep 60; done
rm -f ALL_DONE4
exec bash run4.sh
