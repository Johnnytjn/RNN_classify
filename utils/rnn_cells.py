"""Extension of RNN cell
"""

import tensorflow as tf

class WeightTyingLayer(tf.layers.Layer):
    def __init__(self, embedding, vocab_size, trainable=True, name=None, dtype=None,**kwargs):
        super(WeightTyingLayer,self).__init__(trainable=trainable,name=name,dtype=dtype,**kwargs)
        self.embedding = embedding
        self.embedding_size = tf.shape(self.embedding)[1]
        self.vocab_size = vocab_size

    def build(self, input_shape):
        self.softmax_b = self.add_variable("softmax_b",[self.vocab_size],dtype=tf.float32,trainable=True)
        super(WeightTyingLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        shape = inputs.get_shape().as_list()
        batch_size = shape[0]
        inputs = tf.reshape(inputs,[-1,self.embedding_size])
        softmax_w = tf.matmul(inputs,self.embedding,transpose_b=True)
        logits = tf.nn.bias_add(softmax_w,self.softmax_b)
        if len(shape) > 2:
            logits = tf.reshape(logits,[batch_size,-1,self.vocab_size])
        else:
            logits = tf.reshape(logits,[batch_size, self.vocab_size])
        return logits

    def _compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.vocab_size)


# Do not use this class, use WeightTyingLayer instead
# This class is slow and will get worse result
class WeightTyingWrapper(tf.contrib.rnn.RNNCell):
    """Using wrapper to implement weight tying
    """

    def __init__(self, cell, embedding, vocab_size, output_layer=None, cache=False):
        super(WeightTyingWrapper,self).__init__()
        self._cell = cell
        self.embedding = embedding
        self.vocab_size = vocab_size
        self.softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        self.output_layer = output_layer
        self.cache = cache

    def __call__(self, inputs, state):
        # outputs = [batch * embedding_size]
        outputs, cell_state = self._cell(inputs, state)
        if self.output_layer:
            logits = self.output_layer(outputs)
        else:
            softmax_w = tf.matmul(outputs,self.embedding,transpose_b=True)
            logits = tf.nn.bias_add(softmax_w,self.softmax_b)

        # if using cache, we need to return outputs
        if self.cache:
            logits = tf.concat([logits,outputs],axis=-1)
        return logits, cell_state

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
            It can be represented by an Integer, a TensorShape or a tuple of Integers
            or TensorShapes.
        """
        return self._cell.state_size

    @property
    def output_size(self):
        return self.vocab_size + self._cell.output_size if self.cache else self.vocab_size

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)