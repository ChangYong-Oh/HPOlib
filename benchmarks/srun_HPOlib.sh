#!/bin/bash

SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/spearmint/spearmint_april2013 -s $SEED --cwd branin_20"
SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/spearmint/spearmint_april2013 -s $SEED --cwd hartmann6_20"
SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/spearmint/spearmint_april2013 -s $SEED --cwd rosenbrock_20"
SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/spearmint/spearmint_april2013 -s $SEED --cwd levy_20"


SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/spearmint/spearmint_april2013 -s $SEED --cwd branin_50"
SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/spearmint/spearmint_april2013 -s $SEED --cwd hartmann6_50"
SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/spearmint/spearmint_april2013 -s $SEED --cwd rosenbrock_50"
SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/spearmint/spearmint_april2013 -s $SEED --cwd levy_50"


SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/spearmint/spearmint_april2013 -s $SEED --cwd branin_100"
SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/spearmint/spearmint_april2013 -s $SEED --cwd hartmann6_100"
SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/spearmint/spearmint_april2013 -s $SEED --cwd rosenbrock_100"
SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/spearmint/spearmint_april2013 -s $SEED --cwd levy_100"
