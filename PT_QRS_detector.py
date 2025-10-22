import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime
from scipy.signal import butter, lfilter, filtfilt, find_peaks
import scipy.io as sio

LOG_DIR = "logs/"
PLOT_DIR = "plots/"


class QRSDetectorOffline(object):
    """
    Python Offline ECG QRS Detector based on the Pan-Tomkins algorithm.
    
    Michał Sznajder (Jagiellonian University) - technical contact (msznajder@gmail.com)
    Marta Łukowska (Jagiellonian University)


    The module is offline Python implementation of QRS complex detection in the ECG signal based
    on the Pan-Tomkins algorithm: Pan J, Tompkins W.J., A real-time QRS detection algorithm,
    IEEE Transactions on Biomedical Engineering, Vol. BME-32, No. 3, March 1985, pp. 230-236.

    The QRS complex corresponds to the depolarization of the right and left ventricles of the human heart. It is the most visually obvious part of the ECG signal. QRS complex detection is essential for time-domain ECG signal analyses, namely heart rate variability. It makes it possible to compute inter-beat interval (RR interval) values that correspond to the time between two consecutive R peaks. Thus, a QRS complex detector is an ECG-based heart contraction detector.

    Offline version detects QRS complexes in a pre-recorded ECG signal dataset (e.g. stored in .csv format).

    This implementation of a QRS Complex Detector is by no means a certified medical tool and should not be used in health monitoring. It was created and used for experimental purposes in psychophysiology and psychology.

    You can find more information in module documentation:
    https://github.com/c-labpl/qrs_detector

    If you use these modules in a research project, please consider citing it:
    https://zenodo.org/record/583770

    If you use these modules in any other project, please refer to MIT open-source license.


    MIT License

    Copyright (c) 2017 Michał Sznajder, Marta Łukowska

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    def __init__(self, ecg_data, frequency, verbose=True, log_data=False, plot_data=False, show_plot=False,
                 show_reference=False, reference=None, score_exponents=1, k_factor=2, lowcut=1, highcut=50,
                 ):
        """
        QRSDetectorOffline class initialisation method.
        :param string ecg_data_path: path to the ECG dataset
        :param bool verbose: flag for printing the results
        :param bool log_data: flag for logging the results
        :param bool plot_data: flag for plotting the results to a file
        :param bool show_plot: flag for showing generated results plot - will not show anything if plot is not generated
        """
        # Configuration parameters.
        # self.ecg_data_path = ecg_data_path

        self.signal_frequency = frequency  # Set ECG device frequency in samples per second here.
        frequency_scale = frequency / 250.0

        self.k_factor = k_factor

        self.filter_lowcut = lowcut
        self.filter_highcut = highcut
        self.filter_order = 1

        self.integration_window = int(
            25 * frequency_scale)  # Change proportionally when adjusting frequency (in samples).

        # self.findpeaks_limit = 0.01
        self.findpeaks_spacing = int(
            25 * frequency_scale)  # Change proportionally when adjusting frequency (in samples).

        self.refractory_period = int(
            50 * frequency_scale)  # Change proportionally when adjusting frequency (in samples).
        self.qrs_peak_filtering_factor = 0.125
        self.noise_peak_filtering_factor = 0.125
        self.qrs_noise_diff_weight = 0.25

        # Loaded ECG data.
        self.ecg_data_raw = ecg_data

        # Measured and calculated values.
        self.filtered_ecg_measurements = None
        self.differentiated_ecg_measurements = None
        self.squared_ecg_measurements = None
        self.integrated_ecg_measurements = None
        self.detected_peaks_locs = None
        self.detected_peaks_values = None

        self.qrs_peak_value = 0.0
        self.noise_peak_value = 0.0
        self.threshold_value = 0.0

        self.score_exponent = score_exponents

        # Detection results.
        self.qrs_peaks_indices = np.array([], dtype=int)
        self.noise_peaks_indices = np.array([], dtype=int)

        # Final ECG data and QRS detection results array - samples with detected QRS are marked with 1 value.
        self.ecg_data_detected = None

        # Run whole detector flow.
        # self.load_ecg_data()
        self.detect_peaks()
        # self.detect_qrs()
        self.detect_qrs_adaptive_thres()

        self.show_reference = show_reference
        self.reference = reference

        if verbose:
            self.print_detection_data()

        if log_data:
            self.log_path = "{:s}QRS_offline_detector_log_{:s}.csv".format(LOG_DIR,
                                                                           strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
            self.log_detection_data()

        if plot_data:
            self.plot_path = "{:s}QRS_offline_detector_plot_{:s}.png".format(PLOT_DIR,
                                                                             strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
            self.plot_detection_data(show_plot=show_plot)

    """Loading ECG measurements data methods."""

    def load_ecg_data(self):
        """
        Method loading ECG data set from a file.
        """
        self.ecg_data_raw = np.loadtxt(self.ecg_data_path, skiprows=1, delimiter=',')

    """ECG measurements data processing methods."""

    def detect_peaks(self):
        """
        Method responsible for extracting peaks from loaded ECG measurements data through measurements processing.
        """
        # Extract measurements from loaded ECG data.
        ecg_measurements = self.ecg_data_raw

        # Measurements filtering - 0-15 Hz band pass filter.
        self.filtered_ecg_measurements = self.bandpass_filter(ecg_measurements, lowcut=self.filter_lowcut,
                                                              highcut=self.filter_highcut,
                                                              signal_freq=self.signal_frequency,
                                                              filter_order=self.filter_order)
        # self.filtered_ecg_measurements[:5] = self.filtered_ecg_measurements[5]
        self.filtered_ecg_measurements = self.filtered_ecg_measurements / np.amax(np.abs(self.filtered_ecg_measurements))

        # Derivative - provides QRS slope information.
        # self.differentiated_ecg_measurements = np.ediff1d(self.filtered_ecg_measurements)
        self.differentiated_ecg_measurements = self.derivative_filter(self.filtered_ecg_measurements,
                                                                      self.signal_frequency)
        self.differentiated_ecg_measurements = self.differentiated_ecg_measurements / np.amax(np.abs(
            self.differentiated_ecg_measurements))

        # Squaring - intensifies values received in derivative.
        # self.squared_ecg_measurements = self.differentiated_ecg_measurements ** 2
        self.squared_ecg_measurements = np.abs(self.differentiated_ecg_measurements) ** self.score_exponent

        # Moving-window integration.
        self.integrated_ecg_measurements = np.convolve(self.squared_ecg_measurements,
                                                       np.ones(self.integration_window),
                                                       mode='same')
        self.integrated_ecg_measurements = self.integrated_ecg_measurements / np.amax(self.integrated_ecg_measurements)

        # Fiducial mark - peak detection on integrated measurements.
        peaks_1 = self.findpeaks(data=self.integrated_ecg_measurements,
                                 limit=0,
                                 spacing=self.findpeaks_spacing)
        peaks_2, _ = find_peaks(self.integrated_ecg_measurements,
                                distance=self.findpeaks_spacing)
        # self.detected_peaks_locs = np.array(sorted(list(set(list(peaks_1) + list(peaks_2)))), dtype=np.int)
        self.detected_peaks_locs = peaks_1

        self.detected_peaks_values = self.integrated_ecg_measurements[self.detected_peaks_locs]

        # find the k highest peaks
        k = round(len(self.integrated_ecg_measurements) / self.signal_frequency * self.k_factor)
        k = min(k, len(self.detected_peaks_values)-1)
        k2 = len(self.detected_peaks_values) - k
        largest_k_peaks = np.partition(self.detected_peaks_values, -k)[-k:]
        smallest_k_peaks = np.partition(self.detected_peaks_values, k2-1)[0:k2]

        thres1 = np.mean(largest_k_peaks)
        # thres2 = np.mean(smallest_k_peaks)
        thres2 = 0

        # remove too low peaks
        # valid_indices = self.detected_peaks_values > np.median(largest_k_peaks)/10
        # self.detected_peaks_indices = self.detected_peaks_indices[valid_indices]
        # self.detected_peaks_values = self.detected_peaks_values[valid_indices]

        # normalize the peak values
        self.detected_peaks_values[self.detected_peaks_values > thres1] = thres1
        # self.detected_peaks_values[self.detected_peaks_values < thres2] = thres2
        self.detected_peaks_values = (self.detected_peaks_values - thres2) / (thres1 - thres2)

        self.integrated_ecg_measurements[self.integrated_ecg_measurements > thres1] = thres1
        # self.integrated_ecg_measurements[self.integrated_ecg_measurements < thres2] = thres2
        self.integrated_ecg_measurements = (self.integrated_ecg_measurements - thres2) / (thres1 - thres2)

    """QRS detection methods."""

    def detect_qrs(self):
        """
        Method responsible for classifying detected ECG measurements peaks either as noise or as QRS complex (heart beat).
        """
        for detected_peak_index, detected_peaks_value in zip(self.detected_peaks_locs, self.detected_peaks_values):

            try:
                last_qrs_index = self.qrs_peaks_indices[-1]
            except IndexError:
                last_qrs_index = 0

            # After a valid QRS complex detection, there is a 200 ms refractory period before next one can be detected.
            if detected_peak_index - last_qrs_index > self.refractory_period or not self.qrs_peaks_indices.size:
                # Peak must be classified either as a noise peak or a QRS peak.
                # To be classified as a QRS peak it must exceed dynamically set threshold value.
                if detected_peaks_value > self.threshold_value:
                    self.qrs_peaks_indices = np.append(self.qrs_peaks_indices, detected_peak_index)

                    # Adjust QRS peak value used later for setting QRS-noise threshold.
                    self.qrs_peak_value = self.qrs_peak_filtering_factor * detected_peaks_value + \
                                          (1 - self.qrs_peak_filtering_factor) * self.qrs_peak_value
                else:
                    self.noise_peaks_indices = np.append(self.noise_peaks_indices, detected_peak_index)

                    # Adjust noise peak value used later for setting QRS-noise threshold.
                    self.noise_peak_value = self.noise_peak_filtering_factor * detected_peaks_value + \
                                            (1 - self.noise_peak_filtering_factor) * self.noise_peak_value

                # Adjust QRS-noise threshold value based on previously detected QRS or noise peaks value.
                self.threshold_value = self.noise_peak_value + \
                                       self.qrs_noise_diff_weight * (self.qrs_peak_value - self.noise_peak_value)

        # Create array containing both input ECG measurements data and QRS detection indication column.
        # We mark QRS detection with '1' flag in 'qrs_detected' log column ('0' otherwise).
        measurement_qrs_detection_flag = np.zeros([len(self.ecg_data_raw)])
        measurement_qrs_detection_flag[self.qrs_peaks_indices] = 1
        self.ecg_data_detected = np.append(self.ecg_data_raw, measurement_qrs_detection_flag)

    def detect_qrs_adaptive_thres(self):
        """
        Method responsible for classifying detected ECG measurements peaks either as noise or as QRS complex (heart beat)
        using adaptive thresholds.
        """
        # init thresholds for the integrated_ecg_measurements
        THR_SIG = 0.5
        THR_NOISE = 0.1
        SIG_LEV = THR_SIG
        NOISE_LEV = THR_NOISE
        # init thresholds for the bandpass filtered ecg
        THR_SIG1 = np.amax(self.filtered_ecg_measurements[0:2 * self.signal_frequency]) * 0.25
        THR_NOISE1 = np.mean(self.filtered_ecg_measurements[0:2 * self.signal_frequency]) * 0.5
        SIG_LEV1 = THR_SIG1
        NOISE_LEV1 = THR_NOISE1

        qrs_i = []
        qrs_c = []
        qrs_i_raw = []
        qrs_amp_raw = []
        nois_c = []
        nois_i = []
        m_selected_RR = 0
        mean_RR = 0

        thres_lowing_rate_for_missed_peak = 1

        for peak_id, (detected_peak_index, detected_peaks_value) in enumerate(zip(self.detected_peaks_locs, self.detected_peaks_values)):
            ser_back = 0
            # locate the corresponding peak in the filtered signal
            if detected_peak_index - round(0.075 * self.signal_frequency) >= 0 and \
                    detected_peak_index + round(0.075 * self.signal_frequency) <= len(self.filtered_ecg_measurements):
                y_i = np.amax(self.filtered_ecg_measurements[
                              detected_peak_index - round(0.075 * self.signal_frequency): \
                              detected_peak_index + round(0.075 * self.signal_frequency)])
                x_i = np.argmax(self.filtered_ecg_measurements[
                                detected_peak_index - round(0.075 * self.signal_frequency): \
                                detected_peak_index + round(0.075 * self.signal_frequency)])
            elif detected_peak_index - round(0.075 * self.signal_frequency) < 0:
                y_i = np.amax(self.filtered_ecg_measurements[0: detected_peak_index + round(0.075 * self.signal_frequency)])
                x_i = np.argmax(self.filtered_ecg_measurements[0: detected_peak_index + round(0.075 * self.signal_frequency)])
                ser_back = 1
            else:
                y_i = np.amax(self.filtered_ecg_measurements[detected_peak_index - round(0.075 * self.signal_frequency):])
                x_i = np.argmax(self.filtered_ecg_measurements[detected_peak_index - round(0.075 * self.signal_frequency):])

            # update the heart_rate (Two heart rate means one the moste recent and the other selected)
            if len(qrs_c) >= 9:
                diffRR = np.diff(qrs_i[-9:])  # calculate RR interval
                comp = qrs_i[-1] - qrs_i[-2]  # latest RR

                if m_selected_RR > 0:
                    RR_low_limit = m_selected_RR * 0.92
                    RR_high_limit = m_selected_RR * 1.16
                    stable_RR = diffRR[np.logical_and(diffRR > RR_low_limit, diffRR < RR_high_limit)]
                    if len(stable_RR) >= 8:
                        m_selected_RR = np.mean(stable_RR[-8:])
                else:
                    m_selected_RR = np.median(diffRR)

                if comp <= 0.92 * m_selected_RR or comp >= 1.16 * m_selected_RR:
                    # lower down thresholds to detect better in the integrated signal
                    THR_SIG = 0.5 * (THR_SIG)
                    # lower down thresholds to detect better in the bandpass filtered signal
                    THR_SIG1 = 0.5 * (THR_SIG1)

            # calculate the mean of the last 8 R waves to make sure that QRS is not
            # missing(If no R detected , trigger a search back) 1.66*mean
            if m_selected_RR > 0:
                test_m = m_selected_RR
            else:
                test_m = 0

            if test_m > 0:
                if (detected_peak_index - qrs_i[-1]) >= round(1.66 * test_m): # it shows a QRS is missed

                    mediate_peaks = np.logical_and(self.detected_peaks_locs>qrs_i[-1] + round(0.200 * self.signal_frequency),
                                                   self.detected_peaks_locs<detected_peak_index - round(0.200 * self.signal_frequency))
                    mediate_peaks_locs = self.detected_peaks_locs[mediate_peaks]
                    mediate_peaks_values = self.detected_peaks_values[mediate_peaks]

                    if len(mediate_peaks_values) > 0:
                        highest_id = np.argmax(mediate_peaks_values)
                        locs_temp = mediate_peaks_locs[highest_id]
                        pks_temp = mediate_peaks_values[highest_id]

                        if pks_temp > THR_NOISE * thres_lowing_rate_for_missed_peak:
                            qrs_c.append(pks_temp)
                            qrs_i.append(locs_temp)
                            # find the location in filtered sig
                            x_i_t = np.argmax(self.filtered_ecg_measurements[locs_temp-round(0.075*self.signal_frequency):
                                                                             locs_temp+round(0.075*self.signal_frequency)])
                            y_i_t = self.filtered_ecg_measurements[locs_temp-round(0.075*self.signal_frequency) + x_i_t]
                            # take care of bandpass signal threshold
                            if y_i_t > THR_NOISE1 * thres_lowing_rate_for_missed_peak:
                                qrs_i_raw.append(locs_temp-round(0.075*self.signal_frequency) + x_i_t)
                                qrs_amp_raw.append(y_i_t)
                                SIG_LEV1 = 0.25 * y_i_t + 0.75 * SIG_LEV1

                            not_nois = 1
                            SIG_LEV = 0.25 * pks_temp + 0.75 * SIG_LEV
                else:
                    not_nois = 0

            # find noise and QRS peaks
            if detected_peaks_value >= THR_SIG:
                # if a QRS candidate occurs within 360ms of the previous QRS
                # ,the algorithm determines if its T wave or QRS
                skip = 0
                if len(qrs_c) >= 3:
                    if (detected_peak_index - qrs_i[-1]) <= round(0.3600 * self.signal_frequency):
                        if detected_peak_index + round(0.075 * self.signal_frequency) > len(self.differentiated_ecg_measurements):
                            Slope1 = np.amax(self.differentiated_ecg_measurements[
                                             detected_peak_index - round(0.075 * self.signal_frequency):])
                            Slope2 = np.amax(self.differentiated_ecg_measurements[
                                             qrs_i[-1] - round(0.075 * self.signal_frequency):
                                             qrs_i[-1] + round(0.075 * self.signal_frequency)])
                        elif qrs_i[-1]-round(0.075*self.signal_frequency) < 0:
                            Slope1 = np.amax(self.differentiated_ecg_measurements[
                                             detected_peak_index - round(0.075 * self.signal_frequency):
                                             detected_peak_index + round(0.075 * self.signal_frequency)])
                            Slope2 = np.amax(self.differentiated_ecg_measurements[
                                             0:qrs_i[-1] + round(0.075 * self.signal_frequency)])
                        else:
                            Slope1 = np.amax(self.differentiated_ecg_measurements[
                                             detected_peak_index - round(0.075 * self.signal_frequency):
                                             detected_peak_index + round(0.075 * self.signal_frequency)])
                            Slope2 = np.amax(self.differentiated_ecg_measurements[
                                             qrs_i[-1] - round(0.075 * self.signal_frequency):
                                             qrs_i[-1] + round(0.075 * self.signal_frequency)])

                        if abs(Slope1) <= abs(0.5 * (Slope2)): # slope less then 0.5 of previous R
                            nois_c.append(detected_peaks_value)
                            nois_i.append(detected_peak_index)
                            skip = 1 # T wave identification
                            # adjust noise level in both filtered and integrated signal
                            NOISE_LEV1 = 0.125 * y_i + 0.875 * NOISE_LEV1
                            NOISE_LEV = 0.125 * detected_peaks_value + 0.875 * NOISE_LEV
                        else:
                            skip = 0

                if skip == 0: # skip is 1 when a T wave is detected
                    qrs_c.append(detected_peaks_value)
                    qrs_i.append(detected_peak_index)

                #  bandpass filter check threshold
                if y_i >= THR_SIG1:
                    if ser_back:
                        qrs_i_raw.append(x_i)
                    else:
                        qrs_i_raw.append(detected_peak_index - round(0.075 * self.signal_frequency)+ (x_i - 1))
                    qrs_amp_raw.append(y_i)
                    SIG_LEV1 = 0.125 * y_i + 0.875 * SIG_LEV1

                # adjust Signal level
                SIG_LEV = 0.125*detected_peaks_value + 0.875*SIG_LEV

            elif (THR_NOISE <= detected_peaks_value) and (detected_peaks_value < THR_SIG):
                # adjust Noise level in filtered sig
                NOISE_LEV1 = 0.125 * y_i + 0.875 * NOISE_LEV1
                # adjust Noise level in integrated sig
                NOISE_LEV = 0.125 * detected_peaks_value + 0.875 * NOISE_LEV

            elif detected_peaks_value < THR_NOISE:
                nois_c.append(detected_peaks_value)
                nois_i.append(detected_peak_index)

                # noise level in filtered signal
                NOISE_LEV1 = 0.125 * y_i + 0.875 * NOISE_LEV1
                # noise level in integrated signal
                NOISE_LEV = 0.125 * detected_peaks_value + 0.875 * NOISE_LEV

            #  adjust the threshold with SNR
            if NOISE_LEV != 0 or SIG_LEV != 0:
                THR_SIG = NOISE_LEV + 0.25 * (abs(SIG_LEV - NOISE_LEV))
                THR_NOISE = 0.5 * (THR_SIG)

            # adjust the threshold with SNR for bandpassed signal
            if NOISE_LEV1 != 0 or SIG_LEV1 != 0:
                THR_SIG1 = NOISE_LEV1 + 0.25 * (abs(SIG_LEV1 - NOISE_LEV1))
                THR_NOISE1 = 0.5 * (THR_SIG1)

            skip = 0
            not_nois = 0
            ser_back = 0

        self.qrs_peaks_indices = np.array(qrs_i_raw, dtype=np.int64)
        self.noise_peaks_indices = np.array(nois_i, dtype=np.int64)

        # Create array containing both input ECG measurements data and QRS detection indication column.
        # We mark QRS detection with '1' flag in 'qrs_detected' log column ('0' otherwise).
        measurement_qrs_detection_flag = np.zeros([len(self.ecg_data_raw)])
        measurement_qrs_detection_flag[self.qrs_peaks_indices] = 1
        self.ecg_data_detected = np.append(self.ecg_data_raw, measurement_qrs_detection_flag)


    """Results reporting methods."""

    def print_detection_data(self):
        """
        Method responsible for printing the results.
        """
        print("qrs peaks indices")
        print(self.qrs_peaks_indices)
        print("noise peaks indices")
        print(self.noise_peaks_indices)

    def log_detection_data(self):
        """
        Method responsible for logging measured ECG and detection results to a file.
        """
        with open(self.log_path, "wb") as fin:
            fin.write(b"timestamp,ecg_measurement,qrs_detected\n")
            np.savetxt(fin, self.ecg_data_detected, delimiter=",")

    def plot_detection_data(self, show_plot=False):
        """
        Method responsible for plotting detection results.
        :param bool show_plot: flag for plotting the results and showing plot
        """

        def plot_data(axis, data, title='', fontsize=10):
            axis.set_title(title, fontsize=fontsize)
            axis.grid(which='both', axis='both', linestyle='--')
            axis.plot(data, color="salmon", zorder=1)

        def plot_points(axis, values, indices, color="black"):
            axis.scatter(x=indices, y=values[indices], c=color, s=50, zorder=2)

        plt.close('all')
        fig, axarr = plt.subplots(6, sharex=True, figsize=(15, 18))

        plot_data(axis=axarr[0], data=self.ecg_data_raw, title='Raw ECG measurements')
        plot_data(axis=axarr[1], data=self.filtered_ecg_measurements, title='Filtered ECG measurements')
        plot_data(axis=axarr[2], data=self.differentiated_ecg_measurements, title='Differentiated ECG measurements')
        plot_data(axis=axarr[3], data=self.squared_ecg_measurements, title='Squared ECG measurements')
        plot_data(axis=axarr[4], data=self.integrated_ecg_measurements,
                  title='Integrated ECG measurements with QRS peaks marked (black)')
        plot_points(axis=axarr[4], values=self.integrated_ecg_measurements, indices=self.qrs_peaks_indices)
        if self.show_reference and self.reference is not None:
            plot_points(axis=axarr[4], values=self.integrated_ecg_measurements, indices=self.reference, color="blue")
        plot_data(axis=axarr[5], data=self.filtered_ecg_measurements,
                  title='Raw ECG measurements with QRS peaks marked (black)')
        plot_points(axis=axarr[5], values=self.ecg_data_detected[:], indices=self.qrs_peaks_indices)
        if self.show_reference and self.reference is not None:
            plot_points(axis=axarr[5], values=self.integrated_ecg_measurements, indices=self.reference, color="blue")

        plt.tight_layout()
        fig.savefig(self.plot_path)

        if show_plot:
            plt.show()

        plt.close()

    """Tools methods."""

    def derivative_filter(self, data, signal_freq):
        if signal_freq != 200:
            int_c = (5 - 1) / (signal_freq / 40)
            b = np.interp(np.arange(1, 5.1, int_c), np.arange(1, 5.1),
                          np.array([1, 2, 0, -2, -1]) * (1 / 8) * signal_freq)
            # print(b)
        else:
            b = np.array([1, 2, 0, -2, -1]) * signal_freq / 8

        filted_data = filtfilt(b, 1, data)
        return filted_data

    def bandpass_filter(self, data, lowcut, highcut, signal_freq, filter_order):
        """
        Method responsible for creating and applying Butterworth filter.
        :param deque data: raw data
        :param float lowcut: filter lowcut frequency value
        :param float highcut: filter highcut frequency value
        :param int signal_freq: signal frequency in samples per second (Hz)
        :param int filter_order: filter order
        :return array: filtered data
        """
        nyquist_freq = 0.5 * signal_freq
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq
        b, a = butter(filter_order, [low, high], btype="bandpass", output='ba')
        y = filtfilt(b, a, data)
        return y

    def findpeaks(self, data, spacing=1, limit=None):
        """
        Janko Slavic peak detection algorithm and implementation.
        https://github.com/jankoslavic/py-tools/tree/master/findpeaks
        Finds peaks in `data` which are of `spacing` width and >=`limit`.
        :param ndarray data: data
        :param float spacing: minimum spacing to the next peak (should be 1 or more)
        :param float limit: peaks should have value greater or equal
        :return array: detected peaks indexes array
        """
        len = data.size
        x = np.zeros(len + 2 * spacing)
        x[:spacing] = data[0] - 1.e-6
        x[-spacing:] = data[-1] - 1.e-6
        x[spacing:spacing + len] = data
        peak_candidate = np.zeros(len)
        peak_candidate[:] = True
        for s in range(spacing):
            start = spacing - s - 1
            h_b = x[start: start + len]  # before
            start = spacing
            h_c = x[start: start + len]  # central
            start = spacing + s + 1
            h_a = x[start: start + len]  # after
            peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c >= h_b, h_c > h_a))

        ind = np.argwhere(peak_candidate)
        ind = ind.reshape(ind.size)
        if limit is not None:
            ind = ind[data[ind] > limit]
        else:
            limit = np.mean(data[ind]) / 2
            ind = ind[data[ind] > limit]

        return ind


if __name__ == "__main__":
    ecg_path = '../dataset/CPSC2020/data/A04'
    ecg_data = np.transpose(sio.loadmat(ecg_path)['ecg'])[0]
    ecg_data = ecg_data[round(4e6):round(5e6)]
    print('ecg_data shape: ', ecg_data.shape)
    qrs_detector = QRSDetectorOffline(ecg_data=ecg_data, verbose=True, frequency=400,
                                      log_data=True, plot_data=True, show_plot=False,
                                      show_reference=True)

    result_mat = {
        'raw_signal': ecg_data,
        'integrated_ecg_measurements': qrs_detector.integrated_ecg_measurements,
        'detected_peaks_locs': qrs_detector.detected_peaks_locs,
        'detected_peaks_values': qrs_detector.detected_peaks_values,
        'qrs_peaks_indices': qrs_detector.qrs_peaks_indices,
    }

    logpath = "{:s}QRS_offline_detector_result_{:s}.mat".format('logs/',
                                                                strftime("%Y_%m_%d_%H_%M_%S", gmtime()))

    sio.savemat(logpath, result_mat)
