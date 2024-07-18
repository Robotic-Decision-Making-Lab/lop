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
#model=gp

for selc in ACQ_EPIC ACQ_SPEAR ACQ_RHO #BAYES_INFO_GAIN SGV_UCB MUTUAL_INFO UCB
#for selc in UCB SGV_UCB MUTUAL_INFO MUTUAL_UCB
do
    for fake_func in min #linear #squared_min_max max min logistic squared sin_exp 
    do
        for hyper_sel in no
        do
            for model in gp
            do
                for user in perfect
                do
                    for i_env in 0 1 2 3 4 5 6 7 8 9
                    do
                        stdbuf -oL python3 single_experiment.py --env $i_env --model $model --selector $selc --sel_type choose1 --num_runs $number_runs --num_alts 4 --user $user --hyper $hyper_sel --fake_func $fake_func > results/console_output_${selc}_${model}_${i_env}_${hyper_sel}_${fake_func}.txt 2>&1 & 
                    done
                    wait
                done
            done
        done
    done
done


wait