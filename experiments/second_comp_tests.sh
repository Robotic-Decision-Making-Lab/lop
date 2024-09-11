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
#model=gp

def_pareto=false
# choose1, rating, switch
sel_type=switch
kmedoid=medrand
num_alts=2

# SW_BAYES_PROBIT
for model in linear
do
    for rbf_sigma in 1.0
    do
        for fake_func in min #linear #squared_min_max max min logistic squared sin_exp 
        do
            for hyper_sel in no
            do
                for user in human_choice
                do
                    for v in 60.0 #10.0 160.0
                    do
                        for sigma_abs in 0.1 0.5 1.0 1.5 2.0 10.0
                        do
                            for sigma_pair in 2.0 1.0 0.5 0.1
                            do
                                for rbf_l in 1.0
                                do
                                    for selc in SW_ACQ_RHO ACQ_RHO ABS_ACQ_RHO SW_ACQ_LL ACQ_LL #ABS_ACQ_RHO MUTUAL_INFO SW_ACQ_EPIC SW_ACQ_SPEAR ACQ_SPEAR ABS_ACQ_SPEAR #UCB RANDOM ACQ_SPEAR ACQ_RHO ACQ_EPIC ACQ_LL MUTUAL_INFO SGV_UCB #BAYES_INFO_GAIN_PROBIT #BAYES_INFO_GAIN_999 #ACQ_LL ACQ_SPEAR MUTUAL_INFO SGV_UCB UCB 
                                    do
                                        for i_env in 0 1 2 3 4 5 6 7 8 9
                                        do
                                            stdbuf -oL python3 single_experiment.py --env $i_env --model $model --selector $selc --sel_type $sel_type --num_runs $number_runs --num_alts $num_alts --user $user --hyper $hyper_sel --def_pareto $def_pareto --fake_func $fake_func --kmedoid $kmedoid --sigma_pair $sigma_pair --sigma_abs $sigma_abs --v_abs $v --rbf_sigma $rbf_sigma --rbf_l $rbf_l  > results/console_output_${selc}_${model}_${i_env}_${hyper_sel}_${fake_func}.txt 2>&1 & 
                                        done
                                        wait
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done


wait