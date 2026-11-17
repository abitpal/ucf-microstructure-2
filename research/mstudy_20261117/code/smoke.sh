cd /mnt/scratch/mstudy/code
export PYTHONPATH=/mnt/scratch/mstudy/pylib:/mnt/scratch/mstudy/code WANDB_MODE=disabled SMOKE=1 NTHREADS=4
PY=/opt/kalshi/venv/bin/python
L=/mnt/scratch/mstudy/logs/smoke.out
echo "=== start" > $L
echo "=== classify" >> $L; $PY classify.py >> $L 2>&1; echo "rc=$?" >> $L
echo "=== ae" >> $L; AE_BUDGET=20 CLF_BUDGET=20 $PY ae.py >> $L 2>&1; echo "rc=$?" >> $L
echo "=== trade" >> $L; MT_BUDGET=15 $PY trade_model.py >> $L 2>&1; echo "rc=$?" >> $L
echo "=== fair" >> $L; TEST_DAY=2026-09-20 VAL_DAY=2026-09-19 BUDGET=40 $PY fair_models.py BTCPERP transformer >> $L 2>&1; echo "rc=$?" >> $L
TEST_DAY=2026-09-20 VAL_DAY=2026-09-19 BUDGET=40 $PY fair_models.py BTCPERP gru >> $L 2>&1; echo "rc=$?" >> $L
echo SMOKEDONE >> $L
