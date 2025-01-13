#!/bin/bash
# The script to regenerate the results for the active learning figure in
# the user-preference paper.


killgroup(){
    echo killing...
    kill 0
}

trap killgroup INT TERM



for i in {1..500}
do
    echo "Starting main"
    bash al_psynth_tune14_lin_08.sh
    echo "Sleeping"
    sleep 1m
    echo "ending"
done

