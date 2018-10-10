python3 -m train_rnn \
--checkpoint_dir=/home/wzh/PycharmProjects/dianying/rnn_1008 \
--batch_num=50 \
--batch_size=64 \
--seq_len=32000 \
--num_layers=3 \
--embeddding_size=256 \
--self_attention=True \
--optimizer=AdamOptimizer \
--rnn_cell_name=LSTM