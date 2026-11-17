#!/bin/bash
# download latest results bundle from blob, extract, sync wandb offline runs
S=/private/tmp/claude-501/-Users-abitpalgyawali-Documents-projects-pitch/91ec032f-0e55-41fb-82bd-64250d8782dc/scratchpad
D=$S/results; mkdir -p $D
az storage blob download --account-name kalshiobfdfaed53 --auth-mode login -c tape -n research/mstudy_20261117/latest.tgz -f $D/latest.tgz -o none 2>/dev/null
cd $D && tar xzf latest.tgz && echo "extracted: $(ls $D)"
if [ "$1" = "sync" ]; then
  for r in $D/runs/wandb/offline-run-*; do
    [ -f "$r/.synced_local" ] && continue
    $S/wbenv/bin/wandb sync -p kalshi-microstructure "$r" > /dev/null 2>&1 && touch "$r/.synced_local" && echo "synced $(basename $r)"
  done
fi
