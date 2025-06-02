#!/bin/bash
# The script to regenerate the results for the active learning figure in
# the user-preference paper.


killgroup(){
    echo killing...
    kill 0
}

trap killgroup INT TERM


echo "Starting rev_2"

for i in {1..4}
do
    echo "Starting main"
    bash al_psynth_heuristic_revisions_2.sh
    echo "Sleeping"
    sleep 5m
    echo "ending"
done

echo "Starting rev_3 runs"

for i in {1..3}
do
    echo "Starting main"
    bash al_psynth_heuristic_revisions_3.sh
    echo "Sleeping"
    sleep 5m
    echo "ending"
done
