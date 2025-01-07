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
    bash al_psynth_9_way_known_tune2.sh
    sleep 5m
    bash al_psynth_pair_sigpair_oops.sh
    sleep 5m
done

