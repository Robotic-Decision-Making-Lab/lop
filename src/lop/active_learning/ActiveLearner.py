# Copyright 2023 Ian Rankin
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

# ActiveLearner.py
# Written Ian Rankin - February 2022
#
# Active learning selection algorithms objective functions

import numpy as np

from lop.utilities import get_pareto

import pdb

## Base Active Learning class.
#
# This class has the needed function to perform active learning from a
# gaussian proccess.
class ActiveLearner:

    ## Constructor
    # @param default_to_pareto - [opt default=False] sets whether to always assume
    #               prefering pareto optimal choices when selecting points, if not particulary told not to
    # @param alaways_select_best - [opt default=False] sets whether the select function should append the
    #               the top solution to the front of the solution set every time.
    def __init__(self, default_to_pareto=False, always_select_best=False):
        self.model = None
        self.default_to_pareto=default_to_pareto
        self.always_select_best = always_select_best
        self.first_call_greedy = True

    ## set_model
    # sets the model being used by the active learning framework.
    # should only be called inside a model class,
    def set_model(self, model):
        self.model = model

    ## select
    # Selects the given points
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
        self.first_call_greedy = True
        prefer_pts = self.get_prefered_set_of_pts(candidate_pts, prefer_pts)
        prev_selection = set(prev_selection)

        # each model will predict slightly different values for data.
        # select_greedy will need to implement this
        mu, data = self.model.predict(candidate_pts)

        sel_pts = []

        # check if always_select the best value is given
        if self.always_select_best and len(prev_selection) == 0:
            best_idx = self.select_best(mu, prefer_pts, prev_selection)
            sel_pts = [best_idx]
            prev_selection.add(best_idx)

        pref_not_sel = prefer_pts - prev_selection
        
        if return_not_selected:
            all_not_selected = (set(range(len(candidate_pts))) - prev_selection) - set(sel_pts)
            not_selected = []

            while len(sel_pts) < num_alts and len(all_not_selected) > 0:
                # weird case where points have been selected already but out of 'prefered points'
                # take the first index of the list then.
                if len(pref_not_sel) == 0 and len(not_selected) > 0:
                    selected_idx = not_selected[0]
                    not_selected.pop(0) 
                else:
                    if len(prev_selection) == 0 and select_pair_first:
                        raise Exception("Oops, I didn't implement pairs for return not selected. TODO")
                    selected_idx = self.select_greedy(candidate_pts, mu, data, all_not_selected, prev_selection | set(sel_pts))


                if selected_idx in pref_not_sel or len(pref_not_sel) == 0:
                    sel_pts.append(selected_idx)
                    pref_not_sel.discard(selected_idx)
                else:
                    not_selected.append(selected_idx)
                all_not_selected.discard(selected_idx)

            if len(sel_pts) != num_alts:
                raise Exception("Something happened and there was not enough points to select")

            return sel_pts, not_selected
        else:
            all_not_selected = None            

            while len(sel_pts) < num_alts:
                # only select from the prefered points if they still exist
                if len(pref_not_sel) > 0:
                    if select_pair_first and len(sel_pts) == 0 and hasattr(self, "select_pair"):
                        selected_idx = self.select_pair(candidate_pts, mu, data, pref_not_sel, prev_selection | set(sel_pts))
                        
                        if selected_idx[0] in pref_not_sel:
                            pref_not_sel.remove(selected_idx[0])
                        if selected_idx[1] in pref_not_sel:
                            pref_not_sel.remove(selected_idx[1])
                    else:
                        selected_idx = self.select_greedy(candidate_pts, mu, data, pref_not_sel, prev_selection | set(sel_pts))
                        pref_not_sel.remove(selected_idx)
                else:
                    # get if the set of points not selected and not prefered if not already defined
                    if all_not_selected is None:
                        all_not_selected = (set(range(len(candidate_pts))) - prev_selection) - set(sel_pts)
                    # ensure that there is at least some pts left to select from
                    if len(all_not_selected) == 0:
                        raise Exception("Not enough points for select to create a full set")
                    if select_pair_first and len(sel_pts) == 0 and hasattr(self, "select_pair"):
                        selected_idx = self.select_pair(candidate_pts, mu, data, all_not_selected, prev_selection | set(sel_pts))
                        all_not_selected.remove(selected_idx[0])
                        all_not_selected.remove(selected_idx[1])
                    else:
                        selected_idx = self.select_greedy(candidate_pts, mu, data, all_not_selected, prev_selection | set(sel_pts))
                        all_not_selected.remove(selected_idx)
                
                if len(selected_idx) > 1:
                    sel_pts += list(selected_idx)
                else:
                    # add the selected index
                    sel_pts.append(selected_idx)
            # end while loop
            return sel_pts   


    ## get_prefered_set_of_pts
    # @param candidate_pts - a numpy array of points (nxk), n = number points, k = number of dimmensions
    # @parm prefer_num - [default = None] the points at the start of the candidates
    #                   to prefer selecting from. Returned as:
    #                   a. A number of points at the start of canididate_pts to prefer
    #                   b. A set of points to prefer to select.
    #                   c. 'pareto' to indicate 
    #                   d. Enter 0 explicitly ignore selections
    #                   e. None (default) assumes 0 unless default to pareto is true.
    #
    # @return indicies of prefered points
    def get_prefered_set_of_pts(self, candidate_pts, prefer_pts=None):
        if isinstance(prefer_pts, set):
            return prefer_pts
        elif isinstance(prefer_pts, list):
            return set(prefer_pts)
        elif prefer_pts is None and self.default_to_pareto == False:
            return set(range(candidate_pts.shape[0]))
        elif (prefer_pts is None and self.default_to_pareto) or (prefer_pts == 'pareto'):
            return set(get_pareto(candidate_pts))
        elif isinstance(prefer_pts, int):
            if prefer_pts == 0:
                return set(range(len(candidate_pts)))
            elif prefer_pts < 0:
                raise ValueError("Can't pass negative number of points to prefer")
            else:
                return set(range(prefer_pts)) # first set of indicies of prefered points
        else:
            raise ValueError("get_prefered_set_of_pts/select passed unknown prefer_pts number")

    ## select_best
    # Selects the best path given the data outputed by predict
    # @param mu - the mean of the canidate paths
    # @param prefer_pts - the set of prefered points
    # @param prev_selected - [opt] a set of indicies of previously selected points
    def select_best(self, mu, prefer_pts, prev_selection=set()):
        if isinstance(prev_selection, list):
            prev_selection = set(prev_selection)

        pref_not_sel = prefer_pts - prev_selection
        if len(pref_not_sel) <= 0:
            # if all points have been selected already from prefer_pts, then include all not selected
            pref_not_sel = set(range(len(mu))) - prev_selection

        if len(pref_not_sel) <= 0:
            raise Exception("Select best not given enough not selected points to select a best point")

        pref_not_sel = list(pref_not_sel)
        mu_prefered = pref_not_sel[np.argmax(mu[pref_not_sel])]

        return mu_prefered

        


    ## select_greedy
    # This function greedily selects the best single data point
    # Depending on the selection method, you are not forced to implement this function
    # @param candidate_pts - a numpy array of points (nxk), n = number points, k = number of dimmensions
    # @param mu - a numpy array of mu values outputed from predict. numpy (n)
    # @param data - a user defined tuple of data (determined by the predict function of the model)
    # @param indicies - a list or set of indicies in candidate points to consider.
    # @param prev_selection - a set ofindicies of previously selected points
    #
    # @return the index of the greedy selection.
    def select_greedy(self, candidate_pts, mu, data, indicies, prev_selection):
        raise NotImplementedError("ActiveLearner select_greedy is not implemented and has been called")

    # select_greedy_k
    # This function selects the top k canidates given the data and the select greedy
    # function. This allows a function to select multiple choices in a greedy manner.
    def select_greedy_k(self, cur_selection, num_alts, data):
        num_itr = num_alts - len(cur_selection)


        sel_values = [-float('inf')] * len(cur_selection)

        for i in range(num_itr):
            selection, sel_value = self.select_greedy(cur_selection, data)
            cur_selection.append(selection)
            sel_values.append(sel_value)

        return cur_selection, sel_values



