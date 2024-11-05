# Copyright 2024 Ian Rankin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
# to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# MixedTypeDecision.py
# Written Ian Rankin - August 2024
#
# Active learning selection algorithms that select between rating or choosing


import numpy as np
from lop.active_learning import RateChooseLearner

import pdb


class AlignmentDecision(RateChooseLearner):

    ## Constructor
    # @param pairwise_l - the pairwise learning algorithm to use
    # @param abs_l - the absloute learning algorithm to sue
    # @param default_to_pareto - [opt default=False] sets whether to always assume
    #               prefering pareto optimal choices when selecting points, if not particulary told not to
    # @param alaways_select_best - [opt default=False] sets whether the select function should append the
    #               the top solution to the front of the solution set every time.
    def __init__(self, pairwise_l, abs_l, num_calls_decision, default_to_pareto=False, always_select_best=False):
        super(AlignmentDecision, self).__init__(pairwise_l, abs_l, default_to_pareto, always_select_best)

        self.num_calls = 0
        self.calls_to_decision = num_calls_decision

    
    #overide
    ## select
    # Selects the given points (overriden from ActiveLearning)
    # @param candidate_pts - a numpy array of points (nxk), n = number points, k = number of dimmensions
    # @param num_alts - the number of alterantives to selec (including the highest mean)
    # @param prev_selection - [opt, default = []]a list of indicies that 
    # @param prefer_num - [default = None] the points at the start of the candidates
    #                   to prefer selecting from. Returned as:
    #                   a. A number of points at the start of canididate_pts to prefer
    #                   b. A set of points to prefer to select.
    #                   c. 'pareto' to indicate 
    #                   d. Enter 0 explicitly ignore selections
    #                   e. None (default) assumes 0 unless default to pareto is true.
    # @param return_not_selected - [opt default-false] returns the not selected points when there
    #                   a preference to selecting to certian points. [] if not but set to true.
    #                   
    #
    # @return [highest_mean, highest_selection, next highest selection, ...],
    #          selection values for candidate_pts,
    #          only returns highest mean if "always select best is set"
    def select(self, candidate_pts, num_alts, prev_selection=[], prefer_pts=None, return_not_selected=False, select_pair_first=True):
        #pair_idxs = self.pairwise_l.select(candidate_pts, num_alts, prev_selection, prefer_pts, return_not_selected, select_pair_first)
        #abs_idxs = self.abs_l.select(candidate_pts, 1, prev_selection, prefer_pts, return_not_selected, select_pair_first)

        select_pair = True

        if self.num_calls < self.calls_to_decision:
            if self.num_calls % 2 == 0:
                select_pair = True
            else:
                select_pair = False
        else:
            # select using decision metric.
            select_pair = self.determine_query_type(candidate_pts)



        if select_pair:
            print('Selecting a PREFERENCE query')
            sel_idxs = self.pairwise_l.select(candidate_pts, num_alts, prev_selection, prefer_pts, return_not_selected, select_pair_first)
        else:
            print('Selecting a RATING query')
            sel_idxs = self.abs_l.select(candidate_pts, 1, prev_selection, prefer_pts, return_not_selected, select_pair_first)


        self.num_calls += 1
        # return the selected indicies from pairwise select
        return sel_idxs


    def determine_query_type(self, candidate_pts):
        self.pairwise_l.unset_samples()
        #mu, data = self.pairwise_l.model.predict(candidate_pts)
        M_prev = self.pairwise_l.M
        self.pairwise_l.M = 8000

        
        x_rep, Q_rep = self.pairwise_l.get_representative_Q(candidate_pts)
        mu, sig = self.model.predict(x_rep)
        all_cov = self.model.cov
        all_rep, all_Q = self.pairwise_l.get_samples_from_model(candidate_pts, x_rep)



        pref_data = self.model.y_train[self.model.probit_idxs['relative_discrete']]
        rating_data = self.model.y_train[self.model.probit_idxs['abs']]

        print('pref_data')
        print(pref_data)
        print('rating_data')
        print(rating_data)

        ## Generate samples using only preference data
        self.model.y_train[2] = None
        self.model.optimize()
        mu_pref, sig_pref = self.model.predict(x_rep)
        pref_cov = self.model.cov

        pref_rep, pref_Q = self.pairwise_l.get_samples_from_model(candidate_pts, x_rep)


        ## Generate samples using only rating data
        self.model.y_train[0] = None
        self.model.y_train[2] = rating_data
        

        self.model.optimize()
        mu_rating, sig_rating = self.model.predict(x_rep)
        rating_cov = self.model.cov

        rating_rep, rating_Q = self.pairwise_l.get_samples_from_model(candidate_pts, x_rep)
        

        ## Reset model and start evaluating samples
        self.model.y_train[0] = pref_data
        self.model.y_train[2] = rating_data

        ## Evaluate samples
        align_pref_f = self.pairwise_l.alignment_between(pref_rep, all_rep)
        align_rate_f = self.pairwise_l.alignment_between(rating_rep, all_rep)
        #align_pref_f = self.pairwise_l.alignment(pref_rep, Q_rep)
        #align_rate_f = self.pairwise_l.alignment(rating_rep, Q_rep)

        align_pref_mu = np.mean(align_pref_f)
        align_rate_mu = np.mean(align_rate_f)

        print('Align Preference = ' +str(align_pref_mu) + ' align rate = ' + str(align_rate_mu))


        ranked_full = np.argsort(mu)
        ranked_pref = np.argsort(pref_rep, axis=1)
        ranked_rating = np.argsort(rating_rep, axis=1)
        # ranked_full = mu
        # ranked_pref = pref_rep
        # ranked_rating = rating_rep


        corr_pref = np.corrcoef(ranked_pref, ranked_full[np.newaxis,:])
        corr_rating = np.corrcoef(ranked_rating, ranked_full[np.newaxis,:])

        print('corr pref = ' + str(np.mean(corr_pref[:-1, -1])) + 
              ' corr rating = ' + str(np.mean(corr_rating[:-1, -1])))
        #pdb.set_trace()

        self.pairwise_l.M = M_prev
        return align_pref_mu > align_rate_mu


    # @overide
    ## set_model
    # sets the model being used by the active learning framework.
    # should only be called inside a model class,
    def set_model(self, model):
        self.model = model
        self.pairwise_l.set_model(model)
        self.abs_l.set_model(model)
        self.num_calls = 0