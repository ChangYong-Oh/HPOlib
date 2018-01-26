#!/bin/bash

SEED=$(( ( RANDOM % 20000 )  + 1 ))
COMMAND="srun HPOlib-run -o ../optimizers/tpe/hyperopt_august2013_mod -s $SEED --cwd branin_20"
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
eval "$COMMAND"

SEED=$(( ( RANDOM % 20000 )  + 1 ))
COMMAND="srun HPOlib-run -o ../optimizers/tpe/hyperopt_august2013_mod -s $SEED --cwd hartmann6_20"
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
eval "$COMMAND"

SEED=$(( ( RANDOM % 20000 )  + 1 ))
COMMAND="srun HPOlib-run -o ../optimizers/tpe/hyperopt_august2013_mod -s $SEED --cwd rosenbrock_20"
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
eval "$COMMAND"

SEED=$(( ( RANDOM % 20000 )  + 1 ))
COMMAND="srun HPOlib-run -o ../optimizers/tpe/hyperopt_august2013_mod -s $SEED --cwd levy_20"
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
eval "$COMMAND"




SEED=$(( ( RANDOM % 20000 )  + 1 ))
COMMAND="srun HPOlib-run -o ../optimizers/smac/smac_2_10_00-dev -s $SEED --cwd branin_20"
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
eval "$COMMAND"

SEED=$(( ( RANDOM % 20000 )  + 1 ))
COMMAND="srun HPOlib-run -o ../optimizers/smac/smac_2_10_00-dev -s $SEED --cwd hartmann6_20"
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
eval "$COMMAND"

SEED=$(( ( RANDOM % 20000 )  + 1 ))
COMMAND="srun HPOlib-run -o ../optimizers/smac/smac_2_10_00-dev -s $SEED --cwd rosenbrock_20"
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
eval "$COMMAND"

SEED=$(( ( RANDOM % 20000 )  + 1 ))
COMMAND="srun HPOlib-run -o ../optimizers/smac/smac_2_10_00-dev -s $SEED --cwd levy_20"
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
eval "$COMMAND"




SEED=$(( ( RANDOM % 20000 )  + 1 ))
COMMAND="srun HPOlib-run -o ../optimizers/tpe/hyperopt_august2013_mod -s $SEED --cwd branin_50"
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
eval "$COMMAND"

SEED=$(( ( RANDOM % 20000 )  + 1 ))
COMMAND="srun HPOlib-run -o ../optimizers/tpe/hyperopt_august2013_mod -s $SEED --cwd hartmann6_50"
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
eval "$COMMAND"

SEED=$(( ( RANDOM % 20000 )  + 1 ))
COMMAND="srun HPOlib-run -o ../optimizers/tpe/hyperopt_august2013_mod -s $SEED --cwd rosenbrock_50"
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
eval "$COMMAND"

SEED=$(( ( RANDOM % 20000 )  + 1 ))
COMMAND="srun HPOlib-run -o ../optimizers/tpe/hyperopt_august2013_mod -s $SEED --cwd levy_50"
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
eval "$COMMAND"




SEED=$(( ( RANDOM % 20000 )  + 1 ))
COMMAND="srun HPOlib-run -o ../optimizers/smac/smac_2_10_00-dev -s $SEED --cwd branin_50"
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
eval "$COMMAND"

SEED=$(( ( RANDOM % 20000 )  + 1 ))
COMMAND="srun HPOlib-run -o ../optimizers/smac/smac_2_10_00-dev -s $SEED --cwd hartmann6_50"
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
eval "$COMMAND"

SEED=$(( ( RANDOM % 20000 )  + 1 ))
COMMAND="srun HPOlib-run -o ../optimizers/smac/smac_2_10_00-dev -s $SEED --cwd rosenbrock_50"
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
eval "$COMMAND"

SEED=$(( ( RANDOM % 20000 )  + 1 ))
COMMAND="srun HPOlib-run -o ../optimizers/smac/smac_2_10_00-dev -s $SEED --cwd levy_50"
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
eval "$COMMAND"




SEED=$(( ( RANDOM % 20000 )  + 1 ))
COMMAND="srun HPOlib-run -o ../optimizers/tpe/hyperopt_august2013_mod -s $SEED --cwd branin_100"
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
eval "$COMMAND"

SEED=$(( ( RANDOM % 20000 )  + 1 ))
COMMAND="srun HPOlib-run -o ../optimizers/tpe/hyperopt_august2013_mod -s $SEED --cwd hartmann6_100"
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
eval "$COMMAND"

SEED=$(( ( RANDOM % 20000 )  + 1 ))
COMMAND="srun HPOlib-run -o ../optimizers/tpe/hyperopt_august2013_mod -s $SEED --cwd rosenbrock_100"
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
eval "$COMMAND"

SEED=$(( ( RANDOM % 20000 )  + 1 ))
COMMAND="srun HPOlib-run -o ../optimizers/tpe/hyperopt_august2013_mod -s $SEED --cwd levy_100"
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
eval "$COMMAND"




SEED=$(( ( RANDOM % 20000 )  + 1 ))
COMMAND="srun HPOlib-run -o ../optimizers/smac/smac_2_10_00-dev -s $SEED --cwd branin_100"
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
eval "$COMMAND"

SEED=$(( ( RANDOM % 20000 )  + 1 ))
COMMAND="srun HPOlib-run -o ../optimizers/smac/smac_2_10_00-dev -s $SEED --cwd hartmann6_100"
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
eval "$COMMAND"

SEED=$(( ( RANDOM % 20000 )  + 1 ))
COMMAND="srun HPOlib-run -o ../optimizers/smac/smac_2_10_00-dev -s $SEED --cwd rosenbrock_100"
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
eval "$COMMAND"

SEED=$(( ( RANDOM % 20000 )  + 1 ))
COMMAND="srun HPOlib-run -o ../optimizers/smac/smac_2_10_00-dev -s $SEED --cwd levy_100"
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
echo "==============================$srun COMMAND=============================="
eval "$COMMAND"
