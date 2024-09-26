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

# RateChooseLearner.py
# Written Ian Rankin - August 2024
#
# Active learning selection algorithms that select between rating or choosing

import numpy as np

from lop.active_learning import ActiveLearner

## RateChooseLearner
#
# This class works as an Active learning algorithm that selects between using a rating or choose
# learning
class RateChooseLearner(ActiveLearner):

    ## Constructor
    # @param pairwise_l - the pairwise learning algorithm to use
    # @param abs_l - the absloute learning algorithm to sue
    # @param default_to_pareto - [opt default=False] sets whether to always assume
    #               prefering pareto optimal choices when selecting points, if not particulary told not to
    # @param alaways_select_best - [opt default=False] sets whether the select function should append the
    #               the top solution to the front of the solution set every time.
    def __init__(self, pairwise_l, abs_l, default_to_pareto=False, always_select_best=False):
        super(RateChooseLearner, self).__init__(default_to_pareto,always_select_best)
        self.pairwise_l = pairwise_l
        self.abs_l = abs_l

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
        pair_idxs = self.pairwise_l.select(candidate_pts, num_alts, prev_selection, prefer_pts, return_not_selected, select_pair_first)
        abs_idxs = self.abs_l.select(candidate_pts, 1, prev_selection, prefer_pts, return_not_selected, select_pair_first)

        print('Pairwise selected metric: ' + str(self.pairwise_l.sel_metric))
        print('Absloute selected metric: ' + str(self.abs_l.sel_metric))
        if self.pairwise_l.sel_metric > self.abs_l.sel_metric:
            sel_idxs = pair_idxs
        elif self.pairwise_l.sel_metric == self.abs_l.sel_metric:
            # both the same, just randomly pick one of the options.
            if np.random.random() > 0.5:
                sel_idxs = pair_idxs
            else:
                sel_idxs = abs_idxs
        else:
            sel_idxs = abs_idxs

        # return the selected indicies from pairwise select
        return sel_idxs
    

    # @overide
    ## set_model
    # sets the model being used by the active learning framework.
    # should only be called inside a model class,
    def set_model(self, model):
        self.pairwise_l.set_model(model)
        self.abs_l.set_model(model)



class MixedComparision(RateChooseLearner):

    ## Constructor
    # @param pairwise_l - the pairwise learning algorithm to use
    # @param abs_l - the absloute learning algorithm to sue
    # @param default_to_pareto - [opt default=False] sets whether to always assume
    #               prefering pareto optimal choices when selecting points, if not particulary told not to
    # @param alaways_select_best - [opt default=False] sets whether the select function should append the
    #               the top solution to the front of the solution set every time.
    def __init__(self, pairwise_l, abs_l, abs_comp, default_to_pareto=False, always_select_best=False):
        super(MixedComparision, self).__init__(pairwise_l, abs_l, default_to_pareto, always_select_best)

        self.abs_comp = abs_comp

    

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
        pair_idxs = self.pairwise_l.select(candidate_pts, num_alts, prev_selection, prefer_pts, return_not_selected, select_pair_first)
        abs_idxs = self.abs_l.select(candidate_pts, 1, prev_selection, prefer_pts, return_not_selected, select_pair_first)

        pair_metric = self.pairwise_l.sel_metric

        # calculate absloute selected metric
        _ = self.abs_comp.select(candidate_pts, 1, prev_selection, prefer_pts, return_not_selected, select_pair_first)
        abs_metric = self.abs_comp.sel_metric
        
        print('Pairwise selected metric: ' + str(pair_metric))
        print('Absloute selected metric: ' + str(abs_metric))
        if pair_metric > abs_metric:
            sel_idxs = pair_idxs
        elif self.pairwise_l.sel_metric == abs_metric:
            # both the same, just randomly pick one of the options.
            if np.random.random() > 0.5:
                sel_idxs = pair_idxs
            else:
                sel_idxs = abs_idxs
        else:
            sel_idxs = abs_idxs

        # return the selected indicies from pairwise select
        return sel_idxs



    # @overide
    ## set_model
    # sets the model being used by the active learning framework.
    # should only be called inside a model class,
    def set_model(self, model):
        self.pairwise_l.set_model(model)
        self.abs_l.set_model(model)
        self.abs_comp.set_model(model)





class MixedComparisionSetFixed(RateChooseLearner):

    ## Constructor
    # @param pairwise_l - the pairwise learning algorithm to use
    # @param abs_l - the absloute learning algorithm to sue
    # @param default_to_pareto - [opt default=False] sets whether to always assume
    #               prefering pareto optimal choices when selecting points, if not particulary told not to
    # @param alaways_select_best - [opt default=False] sets whether the select function should append the
    #               the top solution to the front of the solution set every time.
    def __init__(self, pairwise_l, abs_l, comp_func='set_time', params={}, default_to_pareto=False, always_select_best=False):
        super(MixedComparision, self).__init__(pairwise_l, abs_l, default_to_pareto, always_select_best)

        self.comp_func = comp_func
        self.num_calls = 0
        self.params = params

    

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
        pair_idxs = self.pairwise_l.select(candidate_pts, num_alts, prev_selection, prefer_pts, return_not_selected, select_pair_first)
        abs_idxs = self.abs_l.select(candidate_pts, 1, prev_selection, prefer_pts, return_not_selected, select_pair_first)

        pair_metric = self.pairwise_l.sel_metric

        

        if self.num_calls > 7:
            sel_idxs = pair_idxs
        else:
            sel_idxs = abs_idxs

        self.num_calls += 1
        # return the selected indicies from pairwise select
        return sel_idxs



    # @overide
    ## set_model
    # sets the model being used by the active learning framework.
    # should only be called inside a model class,
    def set_model(self, model):
        self.pairwise_l.set_model(model)
        self.abs_l.set_model(model)
        self.abs_comp.set_model(model)



