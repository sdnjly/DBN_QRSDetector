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
import wfdb

np.set_printoptions(threshold=np.inf)
import os
import re
import math

def load_ans(data_path_, fs_, detector, **kwargs):
    '''
    Please modify this function when you have to load model or any other parameters in CPSC2019_challenge()
    '''
    beat_labels_all = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'F', 'e', 'j', 'n', 'E', 'f', 'Q', '?', '/']
    # beat_labels_all = ['N', 'L', 'R', 'A', 'a', 'J', 'V', 'F', 'e', 'j', 'E', 'f', 'Q', '!', 'x', '/']

    def is_dat(l):
        return l.endswith('.dat')

    ecg_files = list(filter(is_dat, os.listdir(data_path_)))
    ecg_files = sorted(ecg_files)
    HR_ref = []
    R_ref = []
    HR_ans = []
    R_ans = []
    signal_lengths = []
    for i in range(len(ecg_files)):
        rpos_file = ecg_files[i]
        index = re.split('[_.]', rpos_file)[0]

        print("{}: {}: ".format(i, index))

        ecg_path = os.path.join(data_path_, index)

        ecg_data, fields = wfdb.rdsamp(ecg_path, channels=[0, 1])
        print(fields['sig_name'])
        if 'MLII' in fields['sig_name']:
            lead_id = fields['sig_name'].index('MLII')
        else:
            lead_id = fields['sig_name'].index('V5')

        ecg_data = ecg_data[:, lead_id]

        signal_length = len(ecg_data)
        ecg_data = ecg_data.squeeze()
        ann = wfdb.rdann(ecg_path, 'atr')
        with_flutter = '!' in ann.symbol

        r_ref = [round(ann.sample[i]) for i in range(len(ann.sample)) if ann.symbol[i] in beat_labels_all]
        r_ref = np.array(r_ref)
        r_ref = r_ref[(r_ref >= 0.5 * fs_) & (r_ref <= signal_length - 0.5 * fs_)]

        r_hr = np.array([loc for loc in r_ref if (loc > 5.5 * fs_ and loc < 9.5 * fs_)])
        hr_ans, r_ans = detector(ecg_data, with_flutter=with_flutter, **kwargs)

        # print("max ecg: ", np.amax(ecg_data))
        # print("min ecg: ", np.amin(ecg_data))

        HR_ref.append(round(60 * fs_ / np.mean(np.diff(r_hr))))
        R_ref.append(r_ref)
        HR_ans.append(hr_ans)
        R_ans.append(r_ans)
        signal_lengths.append(signal_length)

    return R_ref, HR_ref, R_ans, HR_ans, signal_lengths


def load_ans_single(ecg_path, fs_, detector, with_flutter, **kwargs):
    '''
    Please modify this function when you have to load model or any other parameters in CPSC2019_challenge()
    '''
    beat_labels_all = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'F', 'e', 'j', 'n', 'E', 'f', 'Q', '?', '/']
    # beat_labels_all = ['N', 'L', 'R', 'A', 'a', 'J', 'V', 'F', 'e', 'j', 'E', 'f', 'Q', '!', 'x', '/']

    HR_ref = []
    R_ref = []
    HR_ans = []
    R_ans = []
    signal_lengths = []

    ecg_data, fields = wfdb.rdsamp(ecg_path, channels=[0])
    signal_length = len(ecg_data)
    ecg_data = ecg_data.squeeze()
    ann = wfdb.rdann(ecg_path, 'atr')

    r_ref = [round(ann.sample[i]) for i in range(len(ann.sample)) if ann.symbol[i] in beat_labels_all]
    r_ref = np.array(r_ref)
    r_ref = r_ref[(r_ref >= 0.5 * fs_) & (r_ref <= signal_length - 0.5 * fs_)]

    r_hr = np.array([loc for loc in r_ref if (loc > 5.5 * fs_ and loc < 9.5 * fs_)])
    hr_ans, r_ans = detector(ecg_data, with_flutter=with_flutter, **kwargs)

    hr_ref = round(60 * fs_ / np.mean(np.diff(r_hr)))
    # print("max ecg: ", np.amax(ecg_data))
    # print("min ecg: ", np.amin(ecg_data))

    HR_ref.append(hr_ref)
    R_ref.append(r_ref)
    HR_ans.append(hr_ans)
    R_ans.append(r_ans)
    signal_lengths.append(signal_length)

    return R_ref, HR_ref, R_ans, HR_ans, signal_lengths


def score(r_ref, hr_ref, r_ans, hr_ans, fs_, thr_, sig_lens):

    HR_score = 0
    record_flags = np.ones(len(r_ref))
    FN_total = 0
    FP_total = 0
    TP_total = 0
    for i in range(len(r_ref)):
        FN = 0
        FP = 0
        TP = 0

        if math.isnan(hr_ans[i]):
            hr_ans[i] = 0
        hr_der = abs(int(hr_ans[i]) - int(hr_ref[i]))
        if hr_der <= 0.02 * hr_ref[i]:
            HR_score = HR_score + 1
        elif hr_der <= 0.05 * hr_ref[i]:
            HR_score = HR_score + 0.75
        elif hr_der <= 0.1 * hr_ref[i]:
            HR_score = HR_score + 0.5
        elif hr_der <= 0.2 * hr_ref[i]:
            HR_score = HR_score + 0.25

        for j in range(len(r_ref[i])):
            loc = np.where(np.abs(r_ans[i] - r_ref[i][j]) <= thr_ * fs_)[0]
            if j == 0:
                err = np.where((r_ans[i] >= 0.5 * fs_ + thr_ * fs_) & (r_ans[i] <= r_ref[i][j] - thr_ * fs_))[0]
            elif j == len(r_ref[i]) - 1:
                err = np.where((r_ans[i] >= r_ref[i][j] + thr_ * fs_) & (r_ans[i] <= sig_lens[i] - 0.5 * fs_ - thr_ * fs_))[0]
            else:
                err = np.where((r_ans[i] >= r_ref[i][j] + thr_ * fs_) & (r_ans[i] <= r_ref[i][j + 1] - thr_ * fs_))[0]

            FP = FP + len(err)
            if len(loc) >= 1:
                TP += 1
                FP = FP + len(loc) - 1
            elif len(loc) == 0:
                FN += 1

        if FN + FP > 0:
            record_flags[i] = 0
            # print('{}: TP={}, FN={}, FP={}'.format(i, TP, FN, FP))
        elif FN == 1 and FP == 0:
            record_flags[i] = 0.3
        elif FN == 0 and FP == 1:
            record_flags[i] = 0.7

        print('{}: TP={}, FN={}, FP={}'.format(i, TP, FN, FP))

        FN_total += FN
        FP_total += FP
        TP_total += TP

    rec_acc = round(np.sum(record_flags) / len(r_ref), 4)
    hr_acc = round(HR_score / len(r_ref), 4)

    Se_total = TP_total / (TP_total + FN_total + 1e-10)
    PPv_total = TP_total / (TP_total + FP_total + 1e-10)
    Acc_total = TP_total / (TP_total + FP_total + FN_total + 1e-10)
    print('Total: TP={}, FN={}, FP={}'.format(TP_total, FN_total, FP_total))
    print('Se_total: {}'.format(Se_total))
    print('PPv_total: {}'.format(PPv_total))
    print('Acc_total: {}'.format(Acc_total))
    print('QRS_acc: {}'.format(rec_acc))
    print('HR_acc: {}'.format(hr_acc))
    print('Scoring complete.')

    return rec_acc, hr_acc, Se_total, PPv_total, Acc_total


if __name__ == '__main__':
    FS = 500
    THR = 0.075

    DATA_PATH = './data/'
    RPOS_PATH = './ref/'

    R_ref, HR_ref, R_ans, HR_ans, signal_lengths = load_ans(DATA_PATH, RPOS_PATH, FS)
    rec_acc, hr_acc = score(R_ref, HR_ref, R_ans, HR_ans, FS, THR, signal_lengths)

    with open('score.txt', 'w') as score_file:
        print('Total File Number: %d\n' % (np.shape(HR_ans)[0]), file=score_file)
        print('R Detection Acc: %0.4f' % rec_acc, file=score_file)
        print('HR Detection Acc: %0.4f' % hr_acc, file=score_file)

        score_file.close()
