#!/bin/bash
# The script to regenerate the results for the active learning figure in
# the user-preference paper.


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
kmedoid=medrand
num_alts=2

p_synth_pair=0.95
p_synth_abs=0.95


model=gp
rbf_sigma=1.0
#fake_func=logistic
hyper_sel=no
user=human_choice
v=60.0
sigma_abs=1.0
sigma_pair=0.1
rbf_l=1.2


for fake_func in min logistic linear
do
    for p_synth_abs in 0.9 0.8 0.85 0.7
    do
        # sel_type=rating
        alpha=0.5

        # for selc in UCB
        # do
        #     for i_env in 0 1 2 3 4 5 6 7 8 9
        #     do
        #         stdbuf -oL python3 single_experiment.py --env $i_env --model $model --selector $selc --sel_type $sel_type --num_runs $number_runs --num_alts $num_alts --user $user --hyper $hyper_sel --def_pareto $def_pareto --fake_func $fake_func --kmedoid $kmedoid --p_synth_pair $p_synth_pair --p_synth_abs $p_synth_abs --sigma_pair $sigma_pair --sigma_abs $sigma_abs --v_abs $v --alpha $alpha --rbf_sigma $rbf_sigma --rbf_l $rbf_l  > results/console_output_${selc}_${model}_${i_env}_${hyper_sel}_${fake_func}.txt 2>&1 & 
        #     done
        #     wait
        # done

        sel_type=switch
        for selc in ACQ_SPEAR
        do
            for i_env in 0 1 2 3 4 5 6 7 8 9
            do
                stdbuf -oL python3 single_experiment.py --env $i_env --model $model --selector $selc --sel_type $sel_type --num_runs $number_runs --num_alts $num_alts --user $user --hyper $hyper_sel --def_pareto $def_pareto --fake_func $fake_func --kmedoid $kmedoid --p_synth_pair $p_synth_pair --p_synth_abs $p_synth_abs --sigma_pair $sigma_pair --sigma_abs $sigma_abs --v_abs $v --alpha $alpha --rbf_sigma $rbf_sigma --rbf_l $rbf_l  > results/console_output_${selc}_${model}_${i_env}_${hyper_sel}_${fake_func}.txt 2>&1 & 
            done
            wait
        done



        # for alpha in 0.1 0.9
        # do
        #     sel_type=switch

        #     for selc in SW_UCB_SPEAR
        #     do
        #         for i_env in 0 1 2 3 4 5 6 7 8 9
        #         do
        #             stdbuf -oL python3 single_experiment.py --env $i_env --model $model --selector $selc --sel_type $sel_type --num_runs $number_runs --num_alts $num_alts --user $user --hyper $hyper_sel --def_pareto $def_pareto --fake_func $fake_func --kmedoid $kmedoid --p_synth_pair $p_synth_pair --p_synth_abs $p_synth_abs --sigma_pair $sigma_pair --sigma_abs $sigma_abs --v_abs $v --alpha $alpha --rbf_sigma $rbf_sigma --rbf_l $rbf_l  > results/console_output_${selc}_${model}_${i_env}_${hyper_sel}_${fake_func}.txt 2>&1 & 
        #         done
        #         wait
        #     done
        # done
    done
done