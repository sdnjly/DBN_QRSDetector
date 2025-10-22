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

import os
import numpy as np
import wfdb
from scipy.stats import norm

from mitdb_score import load_ans, score
from PT_QRS_detector import QRSDetectorOffline
from DBN_QRS_detector import QRSDetector
import joblib
from sklearn.neighbors import KernelDensity
import scipy.io as sio
import time
from KDModel import KDModel


def main(ecg_path, fs, rr_prob_type, em_iterations=1, positive_sensor_factor=1, k_factor=1, largest_rr=10,
         segment_length=None, context_length=None, global_infer=False, independent_prob_estimate=True,
         seg_update_interval=1):

    beat_labels_all = ['N', 'L', 'R', 'A', 'a', 'J', 'V', 'F', 'e', 'j', 'E', 'f', 'Q', '!', 'x', '/']

    # load data
    ecg_data, fields = wfdb.rdsamp(ecg_path, channels=[0])
    ecg_data = ecg_data.squeeze()
    ann = wfdb.rdann(ecg_path, 'atr')

    r_ref = [round(ann.sample[i]) for i in range(len(ann.sample)) if ann.symbol[i] in beat_labels_all]
    reference = np.array(r_ref)

    # print('reference:', reference)
    signal_length = ecg_data.shape[0] / fs
    # print('ecg_data shape: ', ecg_data.shape)

    # detect candidates
    qrs_detector = QRSDetectorOffline(ecg_data=ecg_data, verbose=False, frequency=fs,
                                      log_data=False, plot_data=False, show_plot=False,
                                      show_reference=True, reference=reference, score_exponents=1,
                                      k_factor=k_factor)
    candidates_t = qrs_detector.detected_peaks_locs / fs
    candidates_e = qrs_detector.detected_peaks_values / qrs_detector.detected_peaks_values.max()

    # finely detecting with DBN
    # init_gmm_components = [[0.2, 0.3, 0.1], [0.5, 1, 0.3], [0.3, 3, 0.6]]
    # init_gmm_components = [[0.2, 0.2, 0.2], [0.5, 1.2, 0.2], [0.3, 3, 0.6]]
    # init_gmm_components = [[0.61, 0.73, 0.05], [0.37, 0.75, 0.11], [0.02, 0.31, 0.1]]
    init_gmm_components = [[0.2, 0.3, 0.1], [0.2, 0.5, 0.1], [0.2, 0.7, 0.1], [0.2, 0.9, 0.1], [0.2, 2, 0.5]]
    # init_gmm_components = [[0.38, 0.97, 0.03], [0.18, 0.29, 0.04], [0.26, 0.80, 0.05], [0.07, 1.23, 0.27], [0.09, 0.53, 0.09]]
    # sensor_models = joblib.load('sensor_kde.sav')
    # positive_sensor_model = sensor_models['positive_kde']
    # negative_sensor_model = sensor_models['negative_kde']

    positive_sensor_model = norm(1, 0.25)
    negative_sensor_model = norm(0, 0.25)
    positive_sensor_densities = positive_sensor_model.pdf(np.arange(0, 1.01, 0.01)) * positive_sensor_factor
    negative_sensor_densities = negative_sensor_model.pdf(np.arange(0, 1.01, 0.01))
    qrs_detector_dbn = QRSDetector(init_gmm_components, positive_sensor_densities, negative_sensor_densities,
                                   largest_rr=10, epsilon=1e-10)
    result = []
    if segment_length is not None:
        if global_infer:
            qrs_detector_dbn = QRSDetector(init_gmm_components, positive_sensor_densities, negative_sensor_densities,
                                           largest_rr=10, epsilon=1e-10)
            for i in range(em_iterations):
                # expand gmm components
                qrs_detector_dbn.expand_gmm_components()

                # extract rr samples in segments
                rrs_all = []
                rr_probabilities_all = []
                signal_length = int(signal_length)
                segment_length = int(segment_length)
                for seg_begin in range(0, signal_length - segment_length, segment_length):
                    if seg_begin + segment_length >= signal_length - segment_length:
                        seg_end = signal_length
                    else:
                        seg_end = seg_begin + segment_length
                    seg_len = seg_end - seg_begin
                    index = np.where(np.logical_and(seg_begin <= candidates_t, candidates_t < seg_end))[0]
                    seg_candi_t = candidates_t[index] - seg_begin
                    seg_candi_e = candidates_e[index]
                    qrs_detector_dbn.fit(seg_len, seg_candi_t, seg_candi_e)
                    rrs, rr_probabilities = qrs_detector_dbn.infer_probable_rr_intervals()
                    rrs_all.extend(rrs)
                    rr_probabilities_all.extend(rr_probabilities)

                # adjust parameters
                qrs_detector_dbn.adjust_parameters_single(rrs_all, rr_probabilities_all)

            # infer most probable states
            qrs_detector_dbn.fit(signal_length, candidates_t, candidates_e)
            result = qrs_detector_dbn.infer_most_probable_sequence()

        else:
            signal_length = int(signal_length)
            segment_length = int(segment_length)
            context_length = int(context_length)
            for seg_begin in range(0, signal_length - segment_length, segment_length):
                if seg_begin + segment_length >= signal_length - segment_length:
                    seg_end = signal_length
                else:
                    seg_end = seg_begin + segment_length
                context_begin = seg_end - context_length if seg_end - context_length > 0 else 0
                context_len = seg_end - context_begin
                index = np.where(np.logical_and(context_begin <= candidates_t, candidates_t < seg_end))[0]
                first_seg_index = np.where(np.logical_and(seg_begin <= candidates_t, candidates_t < seg_end))[0][0]
                context_candi_t = candidates_t[index] - context_begin
                context_candi_e = candidates_e[index]
                if independent_prob_estimate:
                    qrs_detector_dbn = QRSDetector(init_gmm_components, positive_sensor_densities,
                                                   negative_sensor_densities,
                                                   largest_rr=largest_rr, epsilon=1e-10)
                qrs_detector_dbn.fit(context_len, context_candi_t, context_candi_e)
                if int(seg_begin / segment_length) % int(seg_update_interval) == 0:
                    qrs_detector_dbn.adjust_parameters(iterations=em_iterations, rr_prob_type=rr_prob_type)

                most_probable_seq = np.array(qrs_detector_dbn.infer_most_probable_sequence()).astype(np.int64)
                # print('most_probable_seq: ', most_probable_seq)
                # print('index: ', index)
                rpeak_indexes_context = index[most_probable_seq]
                rpeak_indexes_seg = rpeak_indexes_context[rpeak_indexes_context >= first_seg_index]
                result.append(rpeak_indexes_seg)

            if len(result) > 1:
                result = np.concatenate(result, axis=0)
            else:
                result = result[0]
    else:
        qrs_detector_dbn.fit(signal_length, candidates_t, candidates_e)
        qrs_detector_dbn.adjust_parameters(iterations=em_iterations, rr_prob_type=rr_prob_type)
        result = qrs_detector_dbn.infer_most_probable_sequence()

    print(qrs_detector_dbn.gmm_components)

    r_ans = qrs_detector.detected_peaks_locs[result]
    rec_acc, hr_acc, Se_total, PPv_total, Acc_total = score([reference], [1], [r_ans], [1], fs, thr_=0.075, sig_lens=[ecg_data.shape[0]])

    # print results
    # print(qrs_detector.detected_peaks_locs[result])

if __name__ == '__main__':

    ecg_file = r'dataset/mitdb/100'
    fs = 360
    positive_sensor_factor = 1
    k_factor = 2
    rr_prob_type = 'full'
    segment_length = 5
    context_length = 20
    
    iterations = 1
    print("#" * 50)
    print(f"Detecting the QRS in {iterations} iterations")
    time_start = time.time()
    main(ecg_file, fs, rr_prob_type, em_iterations=iterations, positive_sensor_factor=positive_sensor_factor,
         k_factor=k_factor, segment_length=segment_length, context_length=context_length)
    time_end = time.time()
    print('totally cost', time_end-time_start)

    print("#" * 50)
    iterations = 3
    time_start = time.time()
    print(f"Detecting the QRS in {iterations} iterations")
    main(ecg_file, fs, rr_prob_type, em_iterations=iterations, positive_sensor_factor=positive_sensor_factor,
         k_factor=k_factor, segment_length=segment_length, context_length=context_length)
    time_end = time.time()
    print('totally cost', time_end-time_start)