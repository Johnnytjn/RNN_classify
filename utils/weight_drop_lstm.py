import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops.rnn_cell_impl import _Linear

class WeightDropLSTMCell(tf.contrib.rnn.BasicLSTMCell):
    def __init__(self, num_units, weight_keep_drop=0.7, mode=tf.estimator.ModeKeys.TRAIN, 
                forget_bias=1.0, state_is_tuple=True, activation=None, reuse=None):
        """Initialize the parameters for an LSTM cell.
        """
        self.weight_keep_drop = weight_keep_drop
        self.mode = mode
        super(WeightDropLSTMCell,self).__init__( num_units, forget_bias, state_is_tuple, activation, reuse)

    def call(self, inputs, state):
        """Long short-term memory cell (LSTM).
        Args:
            inputs: `2-D` tensor with shape `[batch_size x input_size]`.
            state: An `LSTMStateTuple` of state tensors, each shaped
                `[batch_size x self.state_size]`, if `state_is_tuple` has been set to
                `True`.  Otherwise, a `Tensor` shaped
                `[batch_size x 2 * self.state_size]`.
            Returns:
            A pair containing the new hidden state, and the new state (either a
                `LSTMStateTuple` or a concatenated state, depending on
                `state_is_tuple`).
        """
        sigmoid = tf.sigmoid
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = tf.split(value=state, num_or_size_splits=2, axis=1)

        if self._linear is None:
            self._linear = _Linear([inputs, h], 4 * self._num_units, True)
            if self.mode == tf.estimator.ModeKeys.TRAIN:
                mask = tf.ones_like(self._linear._weights)
                mask_1, mask_2 = tf.split(mask,num_or_size_splits=2,axis=1)
                mask_2 = tf.nn.dropout(mask_2,keep_prob=self.weight_keep_drop) * self.weight_keep_drop
                mask = tf.concat([mask_1,mask_2],axis=1)
                self._linear._weights = self._linear._weights * mask

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(
            value=self._linear([inputs, h]), num_or_size_splits=4, axis=1)

        new_c = (
            c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
        new_h = self._activation(new_c) * sigmoid(o)

        if self._state_is_tuple:
            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
        else:
            new_state = tf.concat([new_c, new_h], 1)
        return new_h, new_state