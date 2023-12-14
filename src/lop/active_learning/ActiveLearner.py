# Copyright 202 Ian Rankin
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
from copy import copy

from lop.utilities import get_pareto

## Base Active Learning class.
#
# This class has the needed function to perform active learning from a
# gaussian proccess.
class ActiveLearner:

    def __init__(self, default_to_pareto=False, always_select_best=True):
        self.model = None
        self.default_to_pareto=default_to_pareto
        self.always_select_best = always_select_best

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
    # @param return_not - [opt default-false] returns the not selected points when there
    #                   a preference to selecting to certian points. [] if not but set to true.
    #                   
    #
    # @return [highest_mean, highest_selection, next highest selection, ...],
    #          selection values for candidate_pts,
    #          only returns highest mean if "always select best is set"
    def select(self, candidate_pts, num_alts, prev_selection=[], prefer_pts=None, not_selected=False):
        prefer_pts = self.get_prefered_set_of_pts(candidate_pts, prefer_pts)
        
        # if self.always_select_best and len(prev_selection) == 0:
        #     sel_paths

        # # previous points not already selected
        # prev_not_sel = prefer_pts - 

        # prev_selection = set(prev_selection)
        # sel_paths = set()

        # while len(sel_paths) < num_alts:


        raise NotImplementedError('ActiveLearner select is not implemented')

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
        if prefer_pts is None and self.default_to_pareto == False:
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



    ## select_greedy
    # This function greedily selects the best single data point
    # Depending on the selection method, you are not forced to implement this function
    # @param prev_selection - a list of previously selected points
    # @param data - a user defined tuple of data (determined by the select function)
    #
    # @return the index of the greedy selection.
    def select_greedy(self, cur_selection, data):
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
