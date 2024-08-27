# read_study.py
# Written Ian Rankin - August 2024
#
# A set of code to read a study yaml and output its results.

import numpy as np
import oyaml as yaml
import argparse
import lop

import matplotlib.pyplot as plt

from glob import glob

import pdb

## train_eval
# Trains and evaluates the given model from the provided yaml file of data
# @param model - the input model data
# @param d - the yaml data of the current set of paths training and eval
# @param train_sort - [opt default=None] the order to train the model
#
# @return numpy array of accuracies (indicies correlate to number of training iteration)
#           i.e., acc[0] = accuracy without any training, acc[1] iteration of training
def train_eval(model, d, train_sort=None):
    if train_sort is None:
        train_sort = np.arange(0, len(d['train']), 1, dtype=int)

    
    train_d = d['train']
    accs = np.empty(len(train_d)+1)

    accs[0] = eval_model(model, d['eval'])
    
    for i, idx in enumerate(train_sort):
        t_d = train_d[idx]

        x_train = np.array(t_d['rewards'])
        pairs = lop.gen_pairs_from_idx(t_d['index_prefered'], np.arange(x_train.shape[0], dtype=int))
        
        model.add(x_train, pairs)
        
        accs[i+1] = eval_model(model, d['eval'])

    return accs


## eval_model
# Given the data provided and the received model, predict the relative accuracy
# @param model - the provided model
# @param eval_d - the evaluation data given as a yaml formal
#
# @return accuracy of the model on the evaluation data
def eval_model(model, eval_d):
    count = 0

    for i, sample in enumerate(eval_d):
        #print('index_prefered: ' + str(sample['index_prefered']))

        mu, sig = model.predict(np.array(sample['rewards']))
        predicted_idx = np.argmax(mu)
        #print(predicted_idx)
        #print(sample['index_prefered'])

        if predicted_idx == sample['index_prefered']:
            count += 1

    acc = count / len(eval_d)
    return acc


def process_all_data(model, directory):
    files = glob(directory+'*.yaml')

    data_runs = []

    for file in files:
        with open(file, 'rb') as f:
            data = yaml.load(f.read(), Loader=yaml.Loader)

            data_runs.append(data['learn'])
            data_runs.append(data['neither-2'])

    acc = np.empty((len(data_runs), len(data_runs[0]['train'])+1))
    
    for i, data in enumerate(data_runs):
        acc_local = train_eval(model, data)
        acc[i] = np.array(acc_local)
        model.reset()

    return acc




def main():
    parser = argparse.ArgumentParser(description='Study data reader')
    parser.add_argument('--folder', type=str, default='../../../user-planner/analysis/results/full_study_active_learning/', help='Filepath to folder for study data')
    parser.add_argument('-f', type=str, default='', help='filepath to yaml file to read')
    args = parser.parse_args()


    model_l = lop.PreferenceLinear(pareto_pairs=True)
    #model_l.add_prior(bounds=np.array([[0,1.5],[0,1.5]]),num_pts=25)

    model_gp = lop.PreferenceGP(lop.RBF_kern(1.0, 0.5), pareto_pairs=True)
    #model_gp.add_prior(bounds=np.array([[0,1.5],[0,1.5]]),num_pts=25)
    model_s = lop.SimplelestModel()

    if args.f == '':
        acc_l = process_all_data(model_l, args.folder)
        std_mean_l = np.std(acc_l, axis=0) / np.sqrt(acc_l.shape[0])
        mean_l = np.mean(acc_l, axis=0)

        acc_gp = process_all_data(model_gp, args.folder)
        std_mean_gp = np.std(acc_l, axis=0) / np.sqrt(acc_gp.shape[0])
        mean_gp = np.mean(acc_gp, axis=0)

        acc_s = process_all_data(model_s, args.folder)
        std_s = np.std(acc_s, axis=0) / np.sqrt(acc_s.shape[0])
        mean_s = np.mean(acc_s, axis=0)


        ## plotting code
        ax = plt.gca()
        colors = {'lin': '#000000', 'gp': '#E69F00', 'random': '#56B4E9', 'sum': '#009E73'}

        x = np.arange(acc_l.shape[1])
        
        
        ax.plot(x, mean_l, color=colors['lin'])
        ax.fill_between(x, mean_l-std_mean_l, mean_l+std_mean_l, 
                        alpha=0.1, color=colors['lin'], label='_nolegend_')
        
        ax.plot(x, mean_gp, color=colors['gp'])
        ax.fill_between(x, mean_gp-std_mean_gp, mean_gp+std_mean_gp, 
                        alpha=0.1, color=colors['gp'], label='_nolegend_')
        
        ax.plot(x, mean_s, color=colors['sum'])
        ax.fill_between(x, mean_s-std_s, mean_s+std_s, 
                        alpha=0.1, color=colors['sum'], label='_nolegend_')



        ax.plot(x, [1/4]*6, color=colors['random'])
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.legend(['Linear', 'GP', 'Static Sum','Random'])

        plt.figure()
        ax = plt.gca()

        ax.boxplot([acc_l[:,-1], acc_gp[:,-1], acc_s[:,-1]], 
                   showmeans=True)
        plt.xticks([1,2,3], ['Linear', 'GP', 'Static Sum'])
        plt.title('Results after 5 iterations')

        plt.show()

    else:
        
        with open(args.folder+args.f, 'rb') as f:
            data = yaml.load(f.read(), Loader=yaml.Loader)

        print(train_eval(model_l, data['learn']))
        model_l.reset()
        print(train_eval(model_l, data['neither-2']))



if __name__ == '__main__':
    main()
