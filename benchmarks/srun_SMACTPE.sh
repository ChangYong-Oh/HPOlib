#!/bin/bash

SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/tpe/hyperopt_august2013_mod -s $SEED --cwd branin_20"
SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/tpe/hyperopt_august2013_mod -s $SEED --cwd hartmann6_20"
SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/tpe/hyperopt_august2013_mod -s $SEED --cwd rosenbrock_20"
SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/tpe/hyperopt_august2013_mod -s $SEED --cwd levy_20"


SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/smac/smac_2_10_00-dev -s $SEED --cwd branin_20"
SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/smac/smac_2_10_00-dev -s $SEED --cwd hartmann6_20"
SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/smac/smac_2_10_00-dev -s $SEED --cwd rosenbrock_20"
SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/smac/smac_2_10_00-dev -s $SEED --cwd levy_20"


SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/tpe/hyperopt_august2013_mod -s $SEED --cwd branin_50"
SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/tpe/hyperopt_august2013_mod -s $SEED --cwd hartmann6_50"
SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/tpe/hyperopt_august2013_mod -s $SEED --cwd rosenbrock_50"
SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/tpe/hyperopt_august2013_mod -s $SEED --cwd levy_50"


SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/smac/smac_2_10_00-dev -s $SEED --cwd branin_50"
SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/smac/smac_2_10_00-dev -s $SEED --cwd hartmann6_50"
SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/smac/smac_2_10_00-dev -s $SEED --cwd rosenbrock_50"
SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/smac/smac_2_10_00-dev -s $SEED --cwd levy_50"


SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/tpe/hyperopt_august2013_mod -s $SEED --cwd branin_100"
SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/tpe/hyperopt_august2013_mod -s $SEED --cwd hartmann6_100"
SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/tpe/hyperopt_august2013_mod -s $SEED --cwd rosenbrock_100"
SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/tpe/hyperopt_august2013_mod -s $SEED --cwd levy_100"


SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/smac/smac_2_10_00-dev -s $SEED --cwd branin_100"
SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/smac/smac_2_10_00-dev -s $SEED --cwd hartmann6_100"
SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/smac/smac_2_10_00-dev -s $SEED --cwd rosenbrock_100"
SEED=$(( ( RANDOM % 20000 )  + 1 ))
eval "srun -C cpunode  HPOlib-run -o ../optimizers/smac/smac_2_10_00-dev -s $SEED --cwd levy_100"