import tensorflow as tf
import os
import opennmt as onmt
from utils.model_helper import (get_device_str, residual_rnn_cell,
                                single_rnn_cell)
from utils.rnn_cells import WeightTyingWrapper, WeightTyingLayer

class RNN(object):
    """RNN model for text classify
    """


    def __init__(self, hparams):
        self.hparams = hparams

    def build(self):
        self.setup_input_placeholders()
        self.setup_embedding()
        self.setup_rnn()
        self.setup_loss()
        if self.is_training():
            self.setup_training()
            self.setup_summary()
        self.saver = tf.train.Saver(tf.global_variables())

    def init_model(self, sess, initializer=None):
        if initializer:
            sess.run(initializer)
        else:
            sess.run(tf.global_variables_initializer())
    
    def save_model(self, sess):
        return self.saver.save(sess, os.path.join(self.hparams.checkpoint_dir,
                                                  "model.ckpt"), global_step=self.global_step)

    def restore_model(self, sess, epoch=None):
        if epoch is None:
            self.saver.restore(sess, tf.train.latest_checkpoint(
                self.hparams.checkpoint_dir))
        else:
            self.saver.restore(
                sess, os.path.join(self.hparams.checkpoint_dir, "model.ckpt" + ("-%d" % epoch)))
        print("restored model")

    def setup_input_placeholders(self):
        self.source_tokens = tf.placeholder(
            tf.int32, shape=[None, self.hparams.seq_len], name='source_tokens')
        self.targets = tf.placeholder(
                tf.int32, shape=[None], name='class')
        self.batch_size = self.hparams.batch_size

        # for training and evaluation
        if self.hparams.mode in ['train', 'eval']:
            self.dropout_keep_prob = tf.placeholder(
                dtype=tf.float32, name='keep_prob')
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        self.global_step = global_step

    def is_training(self):
        return self.hparams.mode == 'train'     

    def setup_summary(self):
        self.summary_writer = tf.summary.FileWriter(
            self.hparams.checkpoint_dir, tf.get_default_graph())
        tf.summary.scalar("train_loss", self.losses)
        tf.summary.scalar("accuracy_summary",self.accuracy)
        tf.summary.scalar("learning_rate", self.learning_rate)
        tf.summary.scalar('gN', self.gradient_norm)
        tf.summary.scalar('pN', self.param_norm)
        self.summary_op = tf.summary.merge_all()

    def setup_embedding(self):
        with tf.variable_scope("embedding") as scope:
            self.embedding = tf.get_variable(name='embedding', shape=[
                                             self.hparams.vocab_size, self.hparams.embedding_size])
            self.source_embedding = tf.nn.embedding_lookup(
                self.embedding, self.source_tokens)

            if self.is_training():
                self.source_embedding = tf.nn.dropout(
                    self.source_embedding, keep_prob=self.dropout_keep_prob)
    
    def setup_rnn(self):
        with tf.variable_scope("rnn") as scope:
            if not self.hparams.weight_tying:
                self.output_layer = tf.layers.Dense(self.hparams.vocab_size)
            else:
                self.output_layer = WeightTyingLayer(self.embedding,self.hparams.vocab_size)

            cell_list = []
            residual_layers = self.hparams.residual_layers
            for i in range(self.hparams.num_layers):
                # Note: if we use weight_tying, then the num_units of the last layer of RNN should be equal to embedding size
                # This is also the independent embedding size and hidden size
                if self.hparams.weight_tying and i == self.hparams.num_layers - 1:
                    rnn_cell = single_rnn_cell(self.hparams.rnn_cell_name, self.hparams.embedding_size,
                                               self.is_training(), self.hparams.dropout_keep_prob, self.hparams.weight_keep_drop, self.hparams.variational_dropout)
                else:
                    rnn_cell = single_rnn_cell(self.hparams.rnn_cell_name, self.hparams.num_units,
                                               self.is_training(), self.hparams.dropout_keep_prob, self.hparams.weight_keep_drop, self.hparams.variational_dropout)
                if i >= self.hparams.num_layers - residual_layers:
                    # Note: in weight_tying, the num_units of the last layer is different from others
                    # we cannot add residual layer on it.
                    if self.hparams.weight_tying and i == self.hparams.num_layers - 1:
                        pass
                    else:
                        wrapper = residual_rnn_cell(self.hparams.residual_type)
                        rnn_cell = wrapper(rnn_cell)

                if self.hparams.num_gpus > 1:
                    device_str = get_device_str(i, self.hparams.num_gpus)
                    rnn_cell = tf.contrib.rnn.DeviceWrapper(
                        rnn_cell, device_str)
                cell_list.append(rnn_cell)

                if self.hparams.num_gpus > 1:
                    device_str = get_device_str(i, self.hparams.num_gpus)
                    rnn_cell = tf.contrib.rnn.DeviceWrapper(
                        rnn_cell, device_str)
                cell_list.append(rnn_cell)

            if len(cell_list) > 1:
                self.final_cell = tf.contrib.rnn.MultiRNNCell(cells=cell_list)
            else:
                self.final_cell = cell_list[0]
            self.initial_state = self.final_cell.zero_state(self.batch_size,dtype=tf.float32)
            if self.hparams.self_attention:
                self.final_cell = tf.contrib.rnn.AttentionCellWrapper(
                    self.final_cell, self.hparams.attn_len)

            outputs, _ = tf.nn.dynamic_rnn(cell=self.final_cell, inputs=self.source_embedding, dtype=tf.float32)
            outputs = tf.reduce_mean(outputs,axis=1)
            fc = tf.layers.dense(outputs, self.hparams.embedding_size, name = 'fc1')
            self.logits = tf.layers.dense(fc, self.hparams.vocab_size, name='fc2')

    def setup_loss(self):
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits+1e-10,labels=self.targets)
        self.losses = tf.reduce_mean(self.loss)
        self.prediction = tf.argmax(self.logits,1,output_type=tf.int32)
        correct_prediction = tf.equal(self.prediction,self.targets)
        self.correct_num=tf.reduce_sum(tf.cast(correct_prediction,tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name="accuracy")

    def setup_training(self):
        # learning rate decay
        if self.hparams.decay_schema == 'exp':
            self.learning_rate = tf.train.exponential_decay(self.hparams.learning_rate, self.global_step,
                                                            self.hparams.decay_steps, 0.96, staircase=True)
        else:
            self.learning_rate = tf.constant(
                self.hparams.learning_rate, dtype=tf.float32)

        opt = onmt.utils.optim.get_optimizer_class(
            self.hparams.optimizer)(self.learning_rate)
        params = tf.trainable_variables()
        get_total_param_num(params)
        # we need to enable the colocate_gradients_with_ops option in tf.gradients to parallelize the gradients computation.
        gradients = tf.gradients(self.losses, params, colocate_gradients_with_ops=True if self.hparams.num_gpus>1 else False)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, self.hparams.max_gradient_norm)
        self.gradient_norm = tf.global_norm(gradients)
        self.param_norm = tf.global_norm(params)
        self.train_op = opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step)

    def feed_state(self, feed_dict, state):
        if self.hparams.self_attention:
            state, attns, attn_states = state
            feed_dict[self.initial_state[1]] = attns
            feed_dict[self.initial_state[2]] = attn_states
            if self.hparams.num_layers == 1:
                initial_state = tuple([self.initial_state[0]])
                state = tuple([state])
            else:
                initial_state = self.initial_state
            for i, (c, h) in enumerate(self.initial_state[0]):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h
        else:
            if self.hparams.num_layers == 1:
                initial_state = tuple([self.initial_state])
                state = tuple([state])
            else:
                initial_state = self.initial_state
            for i, (c, h) in enumerate(initial_state):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h
        return feed_dict

    def train_one_batch(self, sess, source, targets, state, run_info=False, add_summary=False):
        feed_dict = {self.dropout_keep_prob: self.hparams.dropout_keep_prob}
        feed_dict = self.feed_state(feed_dict, state)
        feed_dict[self.source_tokens]= source
        feed_dict[self.targets] = targets
        batch_size = self.batch_size

        if run_info:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            _, batch_loss,accuracy, summary, global_step= sess.run(
                [self.train_op, self.losses,self.accuracy, self.summary_op,
                    self.global_step],
                feed_dict=feed_dict,
                options=run_options,
                run_metadata=run_metadata)
        else:
            _, batch_loss,accuracy, summary, global_step= sess.run(
                [self.train_op, self.losses, self.accuracy, self.summary_op, self.global_step],
                feed_dict=feed_dict)

        if run_info:
            self.summary_writer.add_run_metadata(
                run_metadata, 'step%03d' % global_step)
            print("adding run meta for", global_step)

        if add_summary:
            self.summary_writer.add_summary(summary, global_step=global_step)
        return batch_loss, accuracy, global_step, batch_size

    def eval_one_batch(self, sess, source, targets, state):
        feed_dict = {self.dropout_keep_prob: 1.0}
        feed_dict = self.feed_state(feed_dict, state)
        feed_dict[self.source_tokens] = source
        feed_dict[self.targets] = targets

        batch_loss, accuracy= sess.run(
            [self.losses,self.accuracy ], feed_dict=feed_dict)
        return batch_loss,accuracy
