import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class LeepDataset(Dataset):
    def __init__(self, features, train=True, ids=[]):
        self.feature_all = features.get_features()
        self.train = train

        if train:
            self.target = features.get_labels()
            self.target_group0 = features.get_label0()
            self.target_group1 = features.get_label1()
            self.target_group2 = features.get_label2()
            self.target_group3 = features.get_label3()
            self.target_group4 = features.get_label4()
            self.target_group5 = features.get_label5()
            self.target_group6 = features.get_label6()

        self.feature_group0 = features.get_features0()
        self.feature_group1 = features.get_features1()
        self.feature_group2 = features.get_features2()
        self.feature_group3 = features.get_features3()
        self.feature_group4 = features.get_features4()
        self.feature_group5 = features.get_features5()
        self.feature_group6 = features.get_features6()
        self.feature_group7 = features.get_features7()
        self.feature_group8 = features.get_features8()
        self.feature_group_else = features.get_features_else()

        self.ids = ids

    def __len__(self):
        return len(self.feature_all)

    def __getitem__(self, idx):
        if self.train:
            return {
                "x": self.feature_all[idx],
                "y": self.target[idx],
                "y0": self.target_group0[idx],
                "y1": self.target_group1[idx],
                "y2": self.target_group2[idx],
                "y3": self.target_group3[idx],
                "y4": self.target_group4[idx],
                "y5": self.target_group5[idx],
                "y6": self.target_group6[idx],
                "g0": self.feature_group0[idx],
                "g1": self.feature_group1[idx],
                "g2": self.feature_group2[idx],
                "g3": self.feature_group3[idx],
                "g4": self.feature_group4[idx],
                "g5": self.feature_group5[idx],
                "g6": self.feature_group6[idx],
                "g7": self.feature_group7[idx],
                "g8": self.feature_group8[idx],
                "g_else": self.feature_group_else[idx],
            }
        else:
            return {
                "sample_ids": self.ids[idx],
                "x": self.feature_all[idx],
                "g0": self.feature_group0[idx],
                "g1": self.feature_group1[idx],
                "g2": self.feature_group2[idx],
                "g3": self.feature_group3[idx],
                "g4": self.feature_group4[idx],
                "g5": self.feature_group5[idx],
                "g6": self.feature_group6[idx],
                "g7": self.feature_group7[idx],
                "g8": self.feature_group8[idx],
                "g_else": self.feature_group_else[idx],
            }
