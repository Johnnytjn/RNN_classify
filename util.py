#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jieba
import pandas as pd
from config import stop_words_dir
from sklearn.metrics import f1_score
from tqdm import tqdm

stop_words = set()
with open(stop_words_dir,'r') as f:
    for line in tqdm(f.readlines()):
        stop_words.add(line.strip())

def load_data_from_csv(file_name, header=0, encoding="utf-8"):
    data_df = pd.read_csv(file_name, header=header, encoding=encoding)
    return data_df


def seg_words(contents):
    contents_segs = list()
    for content in tqdm(contents):
        rcontent = content.replace("\r\n", " ").replace("\n", " ")
        segs = [word for word in jieba.cut(rcontent) if word not in stop_words]
        contents_segs.append(" ".join(segs))
    return contents_segs


def get_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=[1, 0, -1, -2], average='macro')
