# MIT License
#
# Copyright (c) 2025 Liu Yang <sdnjly@126.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import numpy as np
from scipy.stats import norm
import time

class QRSDetector(object):

    def __init__(self, init_gmm_components,
                 positive_sensor_densities, negative_sensor_densities, sensor_density_unit=0.01,
                 epsilon=1e-10, largest_rr=5, verbose=0):
        super(QRSDetector, self).__init__()

        self.verbose = verbose
        self.epsilon = epsilon
        self.largest_rr = largest_rr
        self.time_unit = 0.1
        self.max_innerbeat_candidates = int(self.largest_rr / self.time_unit)

        self.signal_length = None
        self.candidates_t = None
        self.candidates_e = None
        self.candidate_number = None

        self.gmm_components = np.copy(init_gmm_components)  # list of (weight, mu, sigma)
        self.gmm_components_number = len(self.gmm_components)

        self.positive_sensor_densities = positive_sensor_densities  # (weight, mu, sigma)
        self.negative_sensor_densities = negative_sensor_densities  # (weight, mu, sigma)
        self.sensor_density_unit = sensor_density_unit

        self.transition_mat = None
        self.margin_probabilities_forward = None
        self.margin_probabilities_backward = None

        self.candidate_positive_sensor_density = None
        self.candidate_negative_sensor_density = None
        self.successive_negative_sensor_density = None

    def fit(self, signal_length, candidates_t, candidates_e):
        self.signal_length = signal_length
        self.candidates_t = candidates_t
        self.candidates_e = candidates_e / candidates_e.max()
        self.candidate_number = len(self.candidates_t)

        t1 = time.time()
        self.compute_sensor_densities()
        t2 = time.time()
        if self.verbose:
            print('compute_sensor_densities cost: ', t2 - t1)
        self.compute_transition_probabilities()
        t3 = time.time()
        if self.verbose:
            print('compute_transition_probabilities cost: ', t3 - t2)

    def compute_sensor_densities(self):

        self.margin_probabilities_forward = np.zeros(self.candidate_number + 2)
        self.margin_probabilities_backward = np.zeros(self.candidate_number + 2)

        self.candidate_positive_sensor_density = np.zeros(self.candidate_number)
        self.candidate_negative_sensor_density = np.zeros(self.candidate_number)
        self.successive_negative_sensor_density = np.zeros((self.candidate_number, self.max_innerbeat_candidates))

        t1 = time.time()
        self.candidate_positive_sensor_density = self.positive_sensor_densities[
            (self.candidates_e/self.sensor_density_unit).astype(np.int64)]
        self.candidate_negative_sensor_density = self.negative_sensor_densities[
            (self.candidates_e / self.sensor_density_unit).astype(np.int64)]
        t2 = time.time()
        if self.verbose:
            print('compute single sensor density cost: ', t2 - t1)

        for i in range(0, self.candidate_number):
            self.successive_negative_sensor_density[i][0] = self.candidate_negative_sensor_density[i]
            for j in range(i + 1, self.candidate_number):
                if self.candidates_t[j] - self.candidates_t[i] > self.largest_rr:
                    break
                self.successive_negative_sensor_density[i][j-i] = self.successive_negative_sensor_density[i][j-i-1] * \
                                                                self.candidate_negative_sensor_density[j]
        t3 = time.time()
        if self.verbose:
            print('compute successive sensor density cost: ', t3 - t2)

    def query_transition_density(self, x):
        pd = 0
        for i in range(self.gmm_components_number):
            pd += self.gmm_components[i][0] * norm.pdf(x, self.gmm_components[i][1], self.gmm_components[i][2])

        return pd

    def query_transition_cumulative(self, x):
        p = 0
        for i in range(self.gmm_components_number):
            p += self.gmm_components[i][0] * norm.cdf(x, self.gmm_components[i][1], self.gmm_components[i][2])

        p = np.clip(p, 0, 1)

        return p

    # compute the transition probability: p(x_{j} = 1, x_{i+1:j-1}=0 | x_{i} = 1)
    # x_{0} is defined as the integrated candidate before the monitoring (pre_monitoring)
    # x_{-1} is defined as the integrated candidate after the monitoring (post_monitoring)
    # P(x_{0} = 1) = 1 and P(x_{-1}) = 1
    def compute_transition_probabilities(self):

        self.transition_mat = np.zeros((self.candidate_number + 1, self.max_innerbeat_candidates))

        # sampling transition probabilities
        transition_densities = self.query_transition_density(np.arange(0, self.largest_rr, self.time_unit))
        transition_densities[0] = 0
        transition_densities[-1] = 0
        transition_cumulative = 1 - self.query_transition_cumulative(np.arange(0, self.largest_rr, self.time_unit))

        # compute the transition probability from pre_monitoring to each candidate
        t1 = time.time()
        if self.candidate_number > self.max_innerbeat_candidates:
            index = np.clip(self.candidates_t[:self.max_innerbeat_candidates] / self.time_unit, 0, len(transition_cumulative) - 1).astype(np.int64)
            self.transition_mat[0, :] = transition_cumulative[index]
        else:
            index = np.clip(self.candidates_t / self.time_unit, 0,
                            len(transition_cumulative) - 1).astype(np.int64)
            self.transition_mat[0, :self.candidate_number] = transition_cumulative[index]
        # normalization
        self.transition_mat[0, :] = self.transition_mat[0, :] / (self.transition_mat[0, :].sum() + 1e-100)

        t2 = time.time()
        if self.verbose:
            print('pre transition cost: ', t2 - t1)

        # compute the transition probability from candidate to candidate
        for i in range(1, self.candidate_number + 1):
            end_i = i + self.max_innerbeat_candidates
            if end_i > self.candidate_number:
                end_i = self.candidate_number
            index_i = np.clip((self.candidates_t[i:end_i] - self.candidates_t[i-1]) / self.time_unit,
                              0, len(transition_densities) - 1).astype(np.int64)
            self.transition_mat[i, 0:end_i-i] = transition_densities[index_i]
            if end_i - i < self.max_innerbeat_candidates:
                index = np.clip((self.signal_length - self.candidates_t[i-1]) / self.time_unit,
                                0, len(transition_cumulative) - 1).astype(np.int64)
                self.transition_mat[i, end_i - i] = transition_cumulative[index]
                # normalization
                self.transition_mat[i, 0:end_i-i] = self.transition_mat[i, 0:end_i-i] / \
                                                    (self.transition_mat[i, 0:end_i-i].sum() + 1e-100) * \
                                                    (1 - self.transition_mat[i, end_i - i])
            else:
                # normalization
                self.transition_mat[i, :] = self.transition_mat[i, :] / (self.transition_mat[i, :].sum() + 1e-100)

        t3 = time.time()
        if self.verbose:
            print('candidates transition cost: ', t3 - t2)


    def compute_margin_probability_forward(self):
        self.margin_probabilities_forward[0] = 1

        for i in range(1, self.candidate_number + 1):
            prob = 0
            for j in range(i - 1, -1, -1):
                if (j > 0 and self.candidates_t[i - 1] - self.candidates_t[j - 1] > self.largest_rr) or \
                        (j == 0 and self.candidates_t[i - 1] - 0 > self.largest_rr):
                    break

                current_prob = self.transition_mat[j, i-j-1] * self.margin_probabilities_forward[j]
                if j > 0:
                    current_prob *= self.candidate_positive_sensor_density[j-1]
                if j < i - 1:
                    current_prob *= self.successive_negative_sensor_density[j, i-j-2]

                prob += current_prob

            self.margin_probabilities_forward[i] = prob

        return

    def compute_margin_probability_backward(self):
        self.margin_probabilities_backward[-1] = 1

        for i in range(self.candidate_number, 0, -1):
            prob = 0
            for j in range(i + 1, self.candidate_number + 2):
                if (j < self.candidate_number + 1 and self.candidates_t[j - 1] - self.candidates_t[
                    i - 1] > self.largest_rr) \
                        or (j == self.candidate_number + 1 and self.signal_length - self.candidates_t[
                    i - 1] > self.largest_rr):
                    break

                current_prob = self.transition_mat[i, j-i-1] * self.margin_probabilities_backward[j]
                if j < self.candidate_number + 1:
                    current_prob *= self.candidate_positive_sensor_density[j-1]
                if i < j - 1:
                    current_prob *= self.successive_negative_sensor_density[i, j - i - 2]

                prob += current_prob

            self.margin_probabilities_backward[i] = prob

    def infer_most_probable_sequence(self):

        max_probabilities = [1]
        most_probable_sequence = [[]]

        for i in range(1, self.candidate_number + 2):
            max_prob = -math.inf
            most_probable_pre_seq = []
            for j in range(i - 1, -1, -1):
                if (j == 0 and i == self.candidate_number + 1)\
                    or (j == 0 and self.candidates_t[i - 1] - 0 > self.largest_rr) \
                    or (i == self.candidate_number + 1 and self.signal_length - self.candidates_t[
                j - 1] > self.largest_rr) \
                    or (j > 0 and i < self.candidate_number + 1 and self.candidates_t[i - 1] - self.candidates_t[j - 1]
                        > self.largest_rr):
                    break

                if self.transition_mat[j, i-j-1] > 0:
                    current_prob = math.log(self.transition_mat[j, i-j-1])
                else:
                    current_prob = -math.inf

                current_prob += max_probabilities[j]
                if j > 0:
                    current_prob += math.log(self.candidate_positive_sensor_density[j - 1] + 1e-200)
                if j < i - 1:
                    current_prob += math.log(self.successive_negative_sensor_density[j, i-j-2] + 1e-200)

                if current_prob > max_prob:
                    max_prob = current_prob
                    if j > 0:
                        most_probable_pre_seq = most_probable_sequence[j] + [j - 1]
                    else:
                        most_probable_pre_seq = []

            max_probabilities.append(max_prob)
            most_probable_sequence.append(most_probable_pre_seq)

        return most_probable_sequence[-1]

    def infer_probable_rr_intervals(self):

        self.compute_margin_probability_forward()
        self.compute_margin_probability_backward()

        rrs = []
        rr_probabilities = []

        for i in range(self.candidate_number - 1):
            for j in range(i + 1, self.candidate_number):
                if self.candidates_t[j] - self.candidates_t[i] > self.largest_rr:
                    break
                prob = self.transition_mat[i + 1, j-i-1] * self.margin_probabilities_forward[i + 1] * \
                       self.margin_probabilities_backward[j + 1]
                prob *= self.candidate_positive_sensor_density[i]
                prob *= self.candidate_positive_sensor_density[j]
                if i < j - 1:
                    prob *= self.successive_negative_sensor_density[i+1, j-i-2]

                rrs.append(self.candidates_t[j] - self.candidates_t[i])
                rr_probabilities.append(prob)

        rrs = np.array(rrs)
        rr_probabilities = np.array(rr_probabilities)

        return rrs, rr_probabilities

    def infer_probable_rr_intervals_simple(self):

        rrs = []
        rr_probabilities = []

        for i in range(self.candidate_number - 1):
            for j in range(i + 1, self.candidate_number):
                if self.candidates_t[j] - self.candidates_t[i] > self.largest_rr:
                    break
                prob = self.transition_mat[i + 1, j-i-1]
                prob *= self.candidate_positive_sensor_density[i]
                prob *= self.candidate_positive_sensor_density[j]
                if i < j - 1:
                    prob *= self.successive_negative_sensor_density[i+1, j-i-2]

                rrs.append(self.candidates_t[j] - self.candidates_t[i])
                rr_probabilities.append(prob)

        rrs = np.array(rrs)
        rr_probabilities = np.array(rr_probabilities)

        return rrs, rr_probabilities

    def infer_probable_rr_intervals_simple2(self):

        rrs = []
        rr_probabilities = []

        for i in range(self.candidate_number - 1):
            for j in range(i + 1, self.candidate_number):
                if self.candidates_t[j] - self.candidates_t[i] > self.largest_rr:
                    break
                prob = self.candidate_positive_sensor_density[i]
                prob *= self.candidate_positive_sensor_density[j]
                if i < j - 1:
                    prob *= self.successive_negative_sensor_density[i+1, j-i-2]

                rrs.append(self.candidates_t[j] - self.candidates_t[i])
                rr_probabilities.append(prob)

        rrs = np.array(rrs)
        rr_probabilities = np.array(rr_probabilities)

        return rrs, rr_probabilities

    def expand_gmm_components(self):
        # sort components according to mu
        self.gmm_components = self.gmm_components[np.argsort(self.gmm_components[:, 1])]
        if self.gmm_components[0][1] > 0.4:
            self.gmm_components = np.concatenate([self.gmm_components, np.array([[0.1, 0.2, 0.1]])])
        for j in range(1, len(self.gmm_components)):
            if self.gmm_components[j][1] < 1 and self.gmm_components[j][1] - self.gmm_components[j - 1][1] > 0.4:
                self.gmm_components = np.concatenate([self.gmm_components, np.array([[0.1,
                                                                                      (self.gmm_components[j][1] +
                                                                                       self.gmm_components[j - 1][
                                                                                           1]) / 2, 0.1]])])
        self.gmm_components = self.gmm_components[np.argsort(self.gmm_components[:, 1])]
        if self.gmm_components[-1, 1] < 1.5:
            self.gmm_components = np.concatenate([self.gmm_components, np.array([[0.2, 2, 0.5]])])
        # normalization
        self.gmm_components[:, 0] /= self.gmm_components[:, 0].sum()
        self.gmm_components_number = len(self.gmm_components)
        # update transition probabilities
        if self.candidate_number is not None:
            self.compute_transition_probabilities()

    def merge_gmm_components(self, gmm_merge_thres):
        end_index = self.gmm_components_number - 1
        for j in range(self.gmm_components_number - 1):
            if self.gmm_components[j][0] == 0:
                break
            for k in range(j + 1, self.gmm_components_number):
                if self.gmm_components[k][0] == 0:
                    break
                if abs(self.gmm_components[j][1] - self.gmm_components[k][1]) < gmm_merge_thres:
                    self.gmm_components[j][1] = (self.gmm_components[j][1] * self.gmm_components[j][0] +
                                                 self.gmm_components[k][1] * self.gmm_components[k][0]) / \
                                                (self.gmm_components[j][0] + self.gmm_components[k][0])
                    self.gmm_components[j][2] = max(self.gmm_components[j][2], self.gmm_components[k][2])
                    self.gmm_components[j][0] += self.gmm_components[k][0]

                    self.gmm_components[k][0] = 0
                    while self.gmm_components[end_index][0] == 0:
                        end_index -= 1
                    if end_index > k:
                        self.gmm_components[[k, end_index]] = self.gmm_components[[end_index, k]]

        while self.gmm_components[end_index][0] == 0:
            end_index -= 1

        self.gmm_components = self.gmm_components[:end_index + 1]
        # normalization
        self.gmm_components[:, 0] /= self.gmm_components[:, 0].sum()
        self.gmm_components_number = len(self.gmm_components)
        return

    def adjust_parameters(self, iterations=10, rr_prob_type=None, gmm_merge_thres=0.05):

        # sort components according to mu
        if iterations > 0:
            self.expand_gmm_components()

        for i in range(iterations):

            # E step
            t1 = time.time()
            if rr_prob_type == 'full':
                rrs, rr_probabilities = self.infer_probable_rr_intervals()
            elif rr_prob_type == 'simple':
                rrs, rr_probabilities = self.infer_probable_rr_intervals_simple()
            elif rr_prob_type == 'simple2':
                rrs, rr_probabilities = self.infer_probable_rr_intervals_simple2()
            elif rr_prob_type == 'simple2-full':
                if i == 0:
                    rrs, rr_probabilities = self.infer_probable_rr_intervals_simple2()
                else:
                    rrs, rr_probabilities = self.infer_probable_rr_intervals()
            else:
                rrs, rr_probabilities = self.infer_probable_rr_intervals()

            p = np.zeros((self.gmm_components_number, len(rrs)))
            for j in range(self.gmm_components_number):
                p[j, :] = rr_probabilities * self.gmm_components[j][0] \
                          * norm.pdf(rrs, self.gmm_components[j][1], self.gmm_components[j][2])

            p_component = p.sum(axis=-1)

            if p_component.sum() == 0:
                print('p_component.sum() is zero')
            weights = p_component / p_component.sum()

            weights[weights < 0.1] = 0.1

            t2 = time.time()
            if self.verbose:
                print('E step cost: ', t2 - t1)
            # M step
            for j in range(self.gmm_components_number):
                self.gmm_components[j][0] = weights[j]  # weight
                self.gmm_components[j][1] = (rrs * p[j, :]).sum() / p_component[j]
                self.gmm_components[j][2] = math.sqrt(
                    ((rrs - self.gmm_components[j][1]) ** 2 * p[j, :] / p_component[j]).sum())
                # set sigma to be no less than 0.1
                self.gmm_components[j][2] = 0.1 if self.gmm_components[j][2] < 0.1 else self.gmm_components[j][2]

            # merge similar component
            self.merge_gmm_components(gmm_merge_thres)

            t3 = time.time()
            if self.verbose:
                print('M step cost: ', t3 - t2)
            self.compute_transition_probabilities()
            t4 = time.time()
            if self.verbose:
                print('Post EM compute_transition_probabilities cost: ', t4 - t3)
        pass

    def adjust_parameters_single(self, rrs, rr_probabilities, gmm_merge_thres=0.05):

        # E step
        p = np.zeros((self.gmm_components_number, len(rrs)))
        for j in range(self.gmm_components_number):
            p[j, :] = rr_probabilities * self.gmm_components[j][0] \
                      * norm.pdf(rrs, self.gmm_components[j][1], self.gmm_components[j][2])

        p_component = p.sum(axis=-1)

        if p_component.sum() == 0:
            print('p_component.sum() is zero')
        weights = p_component / p_component.sum()

        weights[weights < 0.1] = 0.1

        # M step
        for j in range(self.gmm_components_number):
            self.gmm_components[j][0] = weights[j]  # weight
            self.gmm_components[j][1] = (rrs * p[j, :]).sum() / p_component[j]
            self.gmm_components[j][2] = math.sqrt(
                ((rrs - self.gmm_components[j][1]) ** 2 * p[j, :] / p_component[j]).sum())
            # set sigma to be no less than 0.1
            self.gmm_components[j][2] = 0.1 if self.gmm_components[j][2] < 0.1 else self.gmm_components[j][2]

        # merge similar component
        # self.merge_gmm_components(gmm_merge_thres)

        self.compute_transition_probabilities()

        pass

    def print_gmm_params(self):
        print(self.gmm_components)
