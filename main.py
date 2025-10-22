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

import numpy as np
from scipy.stats import norm

from PT_QRS_detetor import QRSDetectorOffline
from DBN_QRS_detetor import QRSDetector
import joblib
from sklearn.neighbors import KernelDensity
import scipy.io as sio
import time
from KDModel import KDModel


def main(ecg_path, ref_path, fs):

    # load data
    ecg_data = np.transpose(sio.loadmat(ecg_path)['ecg'])[0]
    reference = sio.loadmat(ref_path)['R_peak'].flatten()

    print('reference:', reference)
    signal_length = ecg_data.shape[0] / fs
    print('ecg_data shape: ', ecg_data.shape)

    # detect candidates
    qrs_detector = QRSDetectorOffline(ecg_data=ecg_data, verbose=True, frequency=fs,
                                      log_data=False, plot_data=True, show_plot=False,
                                      show_reference=True, reference=reference, score_exponents=0.5)
    candidates_t = qrs_detector.detected_peaks_locs / fs
    candidates_e = qrs_detector.detected_peaks_values / qrs_detector.detected_peaks_values.max()

    # finely detecting with DBN
    # init_gmm_components = [[0.5, 0.5, 0.2], [0.3, 1, 0.2], [0.2, 1.5, 0.5]]
    init_gmm_components = [[0.2, 0.3, 0.1], [0.2, 0.5, 0.1], [0.2, 0.7, 0.1], [0.2, 0.9, 0.1], [0.2, 2, 0.5]]
    # sensor_models = joblib.load('sensor_kde.sav')
    # positive_sensor_model = sensor_models['positive_kde']
    # negative_sensor_model = sensor_models['negative_kde']
    # positive_sensor_densities = positive_sensor_model.pdf(np.arange(0, 1.01, 0.01)) / 10
    # negative_sensor_densities = negative_sensor_model.pdf(np.arange(0, 1.01, 0.01)) / 10
    sensor_std = 0.25
    positive_sensor_model = norm(1, sensor_std)
    negative_sensor_model = norm(0, sensor_std)
    positive_sensor_densities = positive_sensor_model.pdf(np.arange(0, 1.01, 0.01))
    negative_sensor_densities = negative_sensor_model.pdf(np.arange(0, 1.01, 0.01))
    t1 = time.time()
    qrs_detector_dbn = QRSDetector(init_gmm_components, positive_sensor_densities, negative_sensor_densities, epsilon=1e-10)
    qrs_detector_dbn.fit(signal_length, candidates_t, candidates_e)
    t2 = time.time()
    print('Detector init cost: ', t2-t1)
    qrs_detector_dbn.adjust_parameters(iterations=2)
    t3 = time.time()
    print('Adjust params cost: ', t3 - t2)
    result = qrs_detector_dbn.infer_most_probable_sequence()
    t4 = time.time()
    print('Infer cost: ', t4 - t3)

    # print results
    print(qrs_detector.detected_peaks_locs[result])


if __name__ == '__main__':

    ecg_file = r'D:\Research\期刊会议\PT extending\dataset\CPSC2019\data\data_00005'
    ref_file = r'D:\Research\期刊会议\PT extending\dataset\CPSC2019\ref\R_00005'
    fs = 500
    time_start = time.time()
    main(ecg_file, ref_file, fs)
    time_end = time.time()
    print('totally cost', time_end-time_start)
