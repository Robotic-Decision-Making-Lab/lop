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
    bash al_psynth_long1.sh
    sleep 1m
done

