import json
from enum import Enum
from typing import List

import numpy as np
# import torch
import tensorlayerx as tlx
from loguru import logger
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from .center_smoothing import CenterSmoothing
from .classifier_smoothing import ClassifierSmoothing
import tensorflow as tf
import paddle
import torch

def get_x_and_y(x, y, device, skip: int, get_y: bool = True, get_full_y: bool = False):
    if get_full_y:
        full_y = tlx.copy(y)

    if skip != 1:
        batch_indices = tlx.arange(0, x.shape[0], step=skip, dtype=tlx.int64)

        if tlx.BACKEND == 'tensorflow':
            import tensorflow as tf
            x = tf.gather(x, batch_indices, axis=0)
            if get_full_y:
                full_y = tf.gather(full_y, batch_indices, axis=0)
            if get_y:
                y = tf.gather(y, batch_indices, axis=0)
        
        elif tlx.BACKEND == 'torch':
            import torch
            x = x.index_select(0, batch_indices)
            if get_full_y:
                full_y = full_y.index_select(0, batch_indices)
            if get_y:
                y = y.index_select(0, batch_indices)
        
        elif tlx.BACKEND == 'paddle':
            import paddle
            x = paddle.gather(x, batch_indices, axis=0)
            if get_full_y:
                full_y = paddle.gather(full_y, batch_indices, axis=0)
            if get_y:
                y = paddle.gather(y, batch_indices, axis=0)
        
        else:
            raise NotImplementedError("Unsupported backend")
        # x = tlx.gather(x, batch_indices, axis=0)
        # if get_full_y:
        #     full_y = tlx.gather(full_y, batch_indices, axis=0)
        # if get_y:
        #     y = tlx.gather(y, batch_indices, axis=0)

    # if get_y and get_full_y:
    #     return x.to(device), y.to(device), full_y.to(device)
    # elif get_y and not get_full_y:
    #     return x.to(device), y.to(device)
    # elif not get_y and get_full_y:
    #     return x.to(device), full_y.to(device)
    # else:
    #     assert not get_y and not get_full_y
    #     return x.to(device)
    if tlx.BACKEND == 'tensorflow':
        with tf.device(device):
            if get_y and get_full_y:
                return x, y, full_y
            elif get_y and not get_full_y:
                return x, y
            elif not get_y and get_full_y:
                return x, full_y
            else:
                assert not get_y and not get_full_y
                return x
    elif tlx.BACKEND == 'torch':
        if get_y and get_full_y:
            return x.to(device), y.to(device), full_y.to(device)
        elif get_y and not get_full_y:
            return x.to(device), y.to(device)
        elif not get_y and get_full_y:
            return x.to(device), full_y.to(device)
        else:
            assert not get_y and not get_full_y
            return x.to(device)
    elif tlx.BACKEND == 'paddle':
        paddle.set_device(device)
        if get_y and get_full_y:
            return x, y, full_y
        elif get_y and not get_full_y:
            return x, y
        elif not get_y and get_full_y:
            return x, full_y
        else:
            assert not get_y and not get_full_y
            return x
    else:
        raise NotImplementedError("Unsupported backend")

class CertConsts(str, Enum):
    SAMPLE_INDEX = 'sample_index'
    GROUND_TRUTH_LABEL = 'ground_truth_label'

    CSM_RADIUS = 'csm_r'
    CSM_ERROR = 'csm_err'
    CSM_TIME = 'csm_t'

    SMOOTHED_CLS_PRED = 'smoothed_cls_p'
    CERTIFIED_CLS_RADIUS = 'certified_cls_r'
    CERTIFIED_FAIRNESS = 'certified_fair'
    COHEN_SMOOTHING_TIME = 'cls_sm_t'

    CERTIFIED_FAIRNESS_AND_ACCURACY = 'certified_fair_and_acc'

    EMPIRICALLY_FAIR = 'empirically_fair'
    CERTIFIED_FAIR_EMPIRICALLY_UNFAIR = 'certified_but_empirically_unfair'


class CertificationStatistics:

    def __init__(self):
        self.all_samples_certification_data: List[dict] = []

        self.csm_radii = []
        self.center_smoothing_errors = []

        self.ground_truth_labels = []
        self.smoothed_predictions = []

    @staticmethod
    def add_if_present(l: list, certification_data: dict, k: str):
        if k in certification_data:
            l.append(certification_data[k])

    def add(self, certification_data: dict):
        self.all_samples_certification_data.append(certification_data)

        if CertConsts.CSM_RADIUS in certification_data and certification_data[CertConsts.CSM_RADIUS] >= 0.0:
            self.csm_radii.append(certification_data[CertConsts.CSM_RADIUS])
        if CertConsts.CSM_ERROR in certification_data and certification_data[CertConsts.CSM_ERROR] >= 0.0:
            self.center_smoothing_errors.append(certification_data[CertConsts.CSM_ERROR])

        self.ground_truth_labels.append(certification_data[CertConsts.GROUND_TRUTH_LABEL])
        self.smoothed_predictions.append(
            certification_data.get(CertConsts.SMOOTHED_CLS_PRED, ClassifierSmoothing.ABSTAIN))

    def get_smoothed_predictions(self):
        return np.array(self.smoothed_predictions)

    def get_ground_truth_labels(self):
        return np.array(self.ground_truth_labels)

    def total_count(self):
        return len(self.all_samples_certification_data)

    def get_acc(self):
        return accuracy_score(self.ground_truth_labels, self.smoothed_predictions)

    def get_f1(self):
        return f1_score(self.ground_truth_labels, self.smoothed_predictions)

    def get_balanced_acc(self):
        return balanced_accuracy_score(self.ground_truth_labels, self.smoothed_predictions)

    def get_cert_fair(self):
        cert_fair = [cert_data.get(CertConsts.CERTIFIED_FAIRNESS, False) for cert_data in
                     self.all_samples_certification_data]
        return np.mean(cert_fair)

    def get_cert_fair_and_acc(self):
        cert_fair_and_acc = [cert_data.get(CertConsts.CERTIFIED_FAIRNESS_AND_ACCURACY, False) for cert_data in
                             self.all_samples_certification_data]
        return np.mean(cert_fair_and_acc)

    def get_values(self, k: str):
        return [cert_data[k] for cert_data in self.all_samples_certification_data if k in cert_data]

    def get_mean_of_values(self, k: str, default_value_if_empty=0.0):
        values = self.get_values(k)
        if values:
            return np.mean(values)
        else:
            return default_value_if_empty

    def num_of_csm_abstentions(self):
        return sum([(cert_data.get(CertConsts.CSM_RADIUS) == CenterSmoothing.ABSTAIN) for cert_data in
                    self.all_samples_certification_data])

    def num_of_classifier_abstentions(self):
        return sum([(cert_data.get(CertConsts.SMOOTHED_CLS_PRED) == ClassifierSmoothing.ABSTAIN) for cert_data in
                    self.all_samples_certification_data])

    def report(self):
        assert self.total_count() > 0
        logger.debug(f'[n = {self.total_count()}]:')

        csm_times_mean = self.get_mean_of_values(CertConsts.CSM_TIME)
        cohen_smoothing_times_mean = self.get_mean_of_values(CertConsts.COHEN_SMOOTHING_TIME)
        logger.debug('Times (sec) | csm - {:.4f}, csm + cls smoothing - {:.4f}'.format(csm_times_mean,
                                                                                       csm_times_mean + cohen_smoothing_times_mean))

        if self.csm_radii and self.center_smoothing_errors:
            logger.debug('Center smoothing average | radius = {:.5f}, error = {:.5f}'.format(np.mean(self.csm_radii),
                                                                                             np.mean(
                                                                                                 self.center_smoothing_errors)))
        logger.debug(
            'Smoothing abstentions | center - {}, classifier - {} / out of {}'.format(self.num_of_csm_abstentions(),
                                                                                      self.num_of_classifier_abstentions(),
                                                                                      self.total_count()))

        logger.debug('Accuracy (%) | smoothed classification: normal = {:.4f}, balanced = {:.4f}'.format(self.get_acc(),
                                                                                                         self.get_balanced_acc()))
        logger.debug(f'Certified fairness via smoothing (%) | {self.get_cert_fair():.4f}')
        logger.debug(f'Certifiably fair AND accurate (%) | {self.get_cert_fair_and_acc():.4f}')

        stdout_msg = {'n': self.total_count(), 'csm_sec': csm_times_mean,
                      'csm+cls_smoothing_sec': csm_times_mean + cohen_smoothing_times_mean, 'acc': self.get_acc(),
                      'balanced_acc': self.get_balanced_acc(), 'cert_fair': self.get_cert_fair(),
                      'fair_and_acc': self.get_cert_fair_and_acc()}

        if CertConsts.EMPIRICALLY_FAIR in self.all_samples_certification_data[0]:
            for cert_data in self.all_samples_certification_data:
                assert CertConsts.EMPIRICALLY_FAIR in cert_data
                assert CertConsts.CERTIFIED_FAIR_EMPIRICALLY_UNFAIR in cert_data
            logger.debug('Empirically fair (end points match and did not abstain) (%) | {:.4f}'.format(
                self.get_mean_of_values(CertConsts.EMPIRICALLY_FAIR)))
            logger.debug('Certifiably fair but empirically unfair (%) | {:.4f}'.format(
                self.get_mean_of_values(CertConsts.CERTIFIED_FAIR_EMPIRICALLY_UNFAIR)))

            stdout_msg['empirically_fair'] = self.get_mean_of_values(CertConsts.EMPIRICALLY_FAIR)
            stdout_msg['cert_fair_empirically_unfair'] = self.get_mean_of_values(
                CertConsts.CERTIFIED_FAIR_EMPIRICALLY_UNFAIR)

        print(json.dumps(stdout_msg))
