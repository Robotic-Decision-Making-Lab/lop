#!/bin/bash


killgroup(){
    echo killing...
    kill 0
}

trap killgroup INT TERM

# perfect, human_choice
#user_type=human_choice
number_runs=50
#hyper_sel=no
fake_func=linear

for selc in SGV_UCB MUTUAL_INFO UCB RANDOM
#for selc in UCB SGV_UCB MUTUAL_INFO MUTUAL_UCB
do
    for fake_func in linear logistic squared sin_exp 
    do
        for hyper_sel in no
        do
            for i_env in 0 1 2 3 4 5 6 7 8 9
            do
                python3 single_experiment.py --env $i_env --model linear --selector $selc --sel_type choose1 --num_runs $number_runs --num_alts 4 --user human_choice --hyper $hyper_sel --fake_func $fake_func &
            done
            wait
        done
    done
done


wait