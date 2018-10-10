#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

data_path = "/home/wzh/PycharmProjects/tongjianing/data"
model_path = data_path + "/model/"
train_data_path = data_path + "/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv"
validate_data_path = data_path + "/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv"
test_data_path = data_path + "/test/test.csv"
test_data_predict_output_path = data_path + "/predict/test_predict.csv"
stop_words_dir = data_path +'/stop_words.txt'