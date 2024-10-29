# test_synthetic_user.py
# Written Ian Rankin - August 2024
#
# A test of the synthetic user class

import pytest
import numpy as np

import lop



def get_samples(env_mu, env_sig, min_plans=50, max_plans=100, num_prob=10, dim_rewards=5):
    rew = []

    for i in range(len(env_mu)):
        
        env_rew = []
        for j in range(num_prob):
            num_plans = np.random.randint(min_plans, max_plans)

            s = np.random.normal(loc=env_mu[i], scale=env_sig, size=(num_plans, dim_rewards))
            env_rew.append(s)
        
        rew.append(env_rew)
    return rew




def test_synthetic2_choose_min():
    dim_rewards=5
    f = lop.FakeWeightedMin(dim_rewards)
    usr = lop.HumanChoiceUser2(f)

    env_mu = [0.0, 0.4, 0.5, 0.6, 0.0]
    env_sig = [1.0, 1.2, 0.9, 0.7, 0.95]
    eval_rew = get_samples(env_mu, env_sig, dim_rewards=dim_rewards)

    p_pair = 0.95
    p_abss = [0.85, 0.8, 0.7]

    for p_abs in p_abss:
        usr.learn_beta(eval_rew, p_pair, Q_size=2, p_sigma=p_abs)

        assert usr.beta > 0 and usr.beta < 100.0
        assert usr.sigma > 0 and usr.sigma < 10.0

        ### Test user synth

        N = 10000

        train_rew = get_samples(env_mu, env_sig)

        count_choose = 0
        count_rate = 0
        total_count = 0

        for i in range(N):
            env_num = int(i * (len(train_rew) / N))
            pro_num = np.random.randint(0, len(train_rew[env_num]))
            pair = np.random.choice(train_rew[env_num][pro_num].shape[0], 2, replace=False)

            rew_pair = train_rew[env_num][pro_num][pair]
            idx = usr.choose(rew_pair)
            idx_rate = np.argmax(usr.rate(rew_pair))
            score_pair = usr.fake_f(rew_pair)
            corr_idx = np.argmax(score_pair)

            if score_pair[0] != score_pair[1]:
                if corr_idx == idx:
                    count_choose += 1

                if idx_rate == idx:
                    count_rate += 1 
                total_count += 1

        acc_rate = count_rate / total_count
        acc = count_choose / total_count

        assert np.abs(acc - p_pair) < 0.02
        assert np.abs(acc_rate - p_abs) < 0.03


