#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import config
import logging
import os

import numpy as np
import time
import tensorflow as tf
from rnn import RNN
from sklearn.externals import joblib
from util import load_data_from_csv, seg_words, get_f1_score
from utils.training_utils import get_config_proto, save_hparams, load_hparams, early_stop
import tensorflow.contrib.keras as kr

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='train', help="running mode: train | eval | inference")
    parser.add_argument('-mn', '--model_name', type=str, nargs='?',
                        default='fasttext_model.pkl',
                        help='the name of model')
    parser.add_argument('-lr', '--learning_rate', type=float, nargs='?',
                        default=0.1)
    parser.add_argument('-ep', '--epoch', type=int, nargs='?',
                        default=20)
    parser.add_argument('-wn', '--word_ngrams', type=int, nargs='?',
                        default=1)
    parser.add_argument('-mc', '--min_count', type=int, nargs='?',
                        default=1)
    parser.add_argument('-num_train_epoch', '--num_train_epoch', type=int, nargs='?',
                        default=50)
    parser.add_argument('-batch_size', '--batch_size', type=int, nargs='?',
                        default=32)
    parser.add_argument('-embeddding_size', '--embeddding_size', type=int, nargs='?',
                        default=256)
    parser.add_argument('-steps_per_summary', '--steps_per_summary', type=int, nargs='?',
                        default=50)
    parser.add_argument('-steps_per_eval', '--steps_per_eval', type=int, nargs='?',
                        default=100)
    parser.add_argument('-steps_per_stats', '--steps_per_stats', type=int, nargs='?',
                        default=50)
    parser.add_argument('-checkpoint_dir', '--checkpoint_dir', type=str, nargs='?',
                        default='/data/share/tongjianing/project',)
    parser.add_argument('-seq_len', '--seq_len', type=int, nargs='?',
                        default=2000)
    parser.add_argument('-batch_num', '--batch_num', type=int, nargs='?',
                        default=64) 
    parser.add_argument('-num_layers', '--num_layers', type=int, nargs='?',
                        default=1)   
    parser.add_argument('-self_attention','--self_attention', type=bool, nargs='?',
                        default=True)                 
    parser.add_argument('-residual_layers','--residual_layers', type=int, nargs='?',
                        default=0)   
    parser.add_argument("--rnn_cell_name", type = str, default = 'LSTM',help="LSTM | GRU | Weight_LSTM")
    parser.add_argument("--attn_len", type=int, default=40, help="self attention length")   
    parser.add_argument("--optimizer", type=str, default='AdamOptimizer', help="AdamOptimizer | AdagradOptimizer")
    args = parser.parse_args()
    
    def create_hparams(flags):
        return tf.contrib.training.HParams(
            mode = args.mode,
            model_name = args.model_name,
            learning_rate = args.learning_rate,
            epoch = args.epoch,
            word_ngrams = args.word_ngrams,
            min_count = args.min_count,
            num_train_epoch = args.num_train_epoch,
            batch_size = args.batch_size,
            steps_per_summary = args.steps_per_summary,
            steps_per_stats = args.steps_per_stats,
            steps_per_eval  = args.steps_per_eval ,
            checkpoint_dir = args.checkpoint_dir,
            seq_len = args.seq_len,
            batch_num = args.batch_num
    )
    
    

    # load train data
    logger.info("start load load")
    train_data_df = load_data_from_csv(config.train_data_path)
    validate_data_df = load_data_from_csv(config.validate_data_path)

    content_train = train_data_df.iloc[:, 1]

    logger.info("start seg train data")
    train_text = seg_words(content_train)
    logger.info("complete seg train data")

    logger.info("prepare train format")
    logger.info("construct embeddings for every sentence")
    import gensim
    path='/home/wzh/PycharmProjects/tongjianing/vectors.txt'
    w2v_model=gensim.models.KeyedVectors.load_word2vec_format(fname=path)
    vocabulary = w2v_model.vocab
    embedding_train_all_text = []
    for line in train_text:
        embedding_one_text = []
        for word in line.split():
            if word in vocabulary:
                embedding_one_text.append(w2v_model[word])
                
            else:
                embedding_one_text.append([0]*50)
        if len(embedding_one_text) >= seq_len:
            embedding_one_text = embedding_one_text[:seq_len]
        else:
            for i in range(seq_len - len(embedding_one_text)):
                embedding_one_text.append([0]*50)
        embedding_train_all_text.append(embedding_one_text)
    embedding_train_all_text = np.asarray(embedding_train_all_text)
    logger.info("complete construct embeddings for every sentence")
    logger.info("complete formate train data")

    columns = train_data_df.columns.values.tolist()
    logger.info("start train model")
    classifier_dict = dict()
    hparams = create_hparams(args)
    hparams.add_hparam("vocab_size",60391)
    save_hparams(args.checkpoint_dir, hparams)
    for column in columns[2:]:
        train_label = train_data_df[column]
        train_label = np.array(train_label)
        logger.info("start train %s model" % column)
        train_graph = tf.Graph()
        with train_graph.as_default():
            train_model = RNN(hparams)
            train_model.build()
            initializer = tf.global_variables_initializer()
        train_sess = tf.Session(graph=train_graph, config=get_config_proto(log_device_placement=False))

        train_model.init_model(train_sess, initializer = initializer)
                
        print("# Start to train with learning rate {0}, {1}".format(learning_rate,time.ctime()))

        global_step = train_sess.run(train_model.global_step)
        for epoch in range(num_train_epoch):
            state  = train_sess.run(train_model.initial_state)
            checkpoint_loss, average_acc,batch_num = 0.0, 0.0,0
            best_eval = 0.0
            eval_accuracy = []
            for i in range(batch_num):
                start_id = i * batch_size
                end_id = min((i + 1) * batch_size, 10500)
                x=embedding_train_all_text[start_id:end_id], y=train_label[start_id:end_id]
                batch_num += 1
                add_summary = (global_step % steps_per_summary == 0)
                batch_loss, accuracy, global_step, batch_size = train_model.train_one_batch(train_sess, x, y,state, add_summary = add_summary) 
                checkpoint_loss += batch_loss * batch_size
                average_acc += accuracy
                if global_step == 0:
                    continue
                if global_step % steps_per_stats == 0:
                    train_accuracy = average_acc/batch_num
                    ppl_summary = tf.Summary()
                    ppl_summary.value.add(tag='accuracy', simple_value = train_accuracy)
                    train_model.summary_writer.add_summary(ppl_summary, global_step=global_step)

                    print(
                        "# Epoch %d  global step %d batch %d/%d lr %g "
                        "average accuracy %.5f " %
                        (epoch+1, global_step,i+1,batch_num, train_model.learning_rate.eval(session=train_sess),
                        train_accuracy ))


            print("# Finsh epoch {1}, global step {0}".format(global_step, epoch+1))

        classifier_dict[column] = train_model

    logger.info("complete train model")
    logger.info("start save model")
    model_path = config.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    joblib.dump(classifier_dict, model_path + model_name)
    logger.info("complete save model")

    # # validata model
    # content_validata = validate_data_df.iloc[:, 1]

    # logger.info("start seg validata data")
    # content_validata = seg_words(content_validata)
    # logger.info("complet seg validata data")

    # logger.info("prepare valid format")
    # validata_data_format = np.asarray([content(x,y) in enumerate()
    # logger.info("complete formate train data")(x,y) in enumerate()

    # logger.info("start compute f1 score for validata model")
    # f1_score_dict = dict()
    # for column in columns[2:]:
    #     true_label = np.asarray(validate_data_df[column])
    #     classifier = classifier_dict[column]
    #     pred_label = classifier.predict(validata_data_format).astype(int)
    #     f1_score = get_f1_score(true_label, pred_label)
    #     f1_score_dict[column] = f1_score

    # f1_score = np.mean(list(f1_score_dict.values()))
    # str_score = "\n"
    # for column in columns[2:]:
    #     str_score += column + ":" + str(f1_score_dict[column]) + "\n"

    # logger.info("f1_scores: %s\n" % str_score)
    # logger.info("f1_score: %s" % f1_score)
    # logger.info("complete compute f1 score for validate model")


    # def train_eval(model, sess):
    #     checkpoint_loss,  average_acc,batch_num = 0.0, 0.0,0
    #     state  = sess.run(model.initial_state)
    #     for i in range(batch_num):
    #             start_id = i * self.batch_size
    #             end_id = min((i + 1) * self.batch_size, 10500)
    #             x=train_text[start_id:end_id], y=train_label[start_id:end_id]
    #         batch_num += 1
    #         batch_loss, accuracy= model.eval_one_batch(sess,  x,  y,  state)  # batch_loss, final_state, token_num
    #         checkpoint_loss += batch_loss * flags.batch_size
    #         if (i+1) % 100 == 0:
    #             print("# batch %d/%d" %(i+1,dataset.num_batches))
    #         average_acc += accuracy
    #     average_acc = average_acc/batch_num
    #     print( "# average accuracy %.5f " % (average_acc))
    #     return average_acc