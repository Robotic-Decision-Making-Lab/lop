#!/bin/bash
# The script to regenerate the results for the active learning figure in
# the user-preference paper.


killgroup(){
    echo killing...
    kill 0
}

trap killgroup INT TERM



for i in {1..8}
do
    echo "Starting main"
    bash al_psynth_heuristic_revisions_1.sh
    echo "Sleeping"
    sleep 5m
    echo "ending"
done

