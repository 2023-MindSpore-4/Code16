from config import USE_DATA
import mindspore
from mindspore import Tensor, nn
from mindspore import ops
import numpy as np

SEED=1234

USER_EMBEDDING_SIZE=128
ITEM_EMBEDDING_SIZE=128
OTHER_EMBEDDING_SIZE=32
LSTM_1=128
FC1_SIZE=128
FC2_SIZE=64
FC3_SIZE=1
MAX_TO_KEEP=10

#ml-1m
USER_SIZE=6040
ITEM_SIZE=3706
GENDER_DIM=2
AGE_DIM=7
OCCUPATION_DIM=21
RATE_DIM=6
YEAR_DIM=81
GENRE_DIM=25
DIRECTOR_DIM=2186
TOT_ITEM_EMBEDDING_SIZE= ITEM_EMBEDDING_SIZE + OTHER_EMBEDDING_SIZE*4

din_all_size = 512
att_hist_embedding_size = 128
lstm_input_size = 128
tot_embedding_size = 464

if USE_DATA=='bookcrossing':
    USER_SIZE=7369
    ITEM_SIZE = 291537
    AGE_DIM = 7
    AUTHOR_DIM=102031
    YEAR_DIM=118
    PUBLISHER_DIM=16810
    TOT_ITEM_EMBEDDING_SIZE = ITEM_EMBEDDING_SIZE + OTHER_EMBEDDING_SIZE * 3
elif USE_DATA=='avazu':
    USER_SIZE=34452
    ITEM_SIZE = 1294660
    DEVICE_TYPE_SIZE = 4
    C1_DIM=7
    C14_DIM=2626
    C15_DIM=8
    C16_DIM = 9
    TOT_ITEM_EMBEDDING_SIZE = ITEM_EMBEDDING_SIZE + OTHER_EMBEDDING_SIZE * 4

def sequence_mask(nums, length):
    # key_masks = tf.sequence_mask(items_length, hist_items_length)  # [B, N]
    masks = ops.zeros((nums.shape[0], length), mindspore.bool_) 
    for i, num in enumerate(nums): 
        masks[:, :num]  = True
    return masks

class OurModel(nn.Cell):
    def __init__(self, args):
        super(OurModel, self).__init__()

        self.batches = args.batches
        self.decay_rate = args.decay_rate
        self.trainable = args.trainable
        self.drop_keep_prob = args.drop_keep_prob
        self.regularizer_weight_decay = args.regularizer_weight_decay


        global USER_EMBEDDING_SIZE,ITEM_EMBEDDING_SIZE,OTHER_EMBEDDING_SIZE,LSTM_1,TOT_ITEM_EMBEDDING_SIZE
        USER_EMBEDDING_SIZE = args.USER_EMBEDDING_SIZE
        ITEM_EMBEDDING_SIZE = args.ITEM_EMBEDDING_SIZE
        OTHER_EMBEDDING_SIZE = args.OTHER_EMBEDDING_SIZE
        TOT_ITEM_EMBEDDING_SIZE = ITEM_EMBEDDING_SIZE + OTHER_EMBEDDING_SIZE * 4

        if USE_DATA == 'bookcrossing':
            TOT_ITEM_EMBEDDING_SIZE = ITEM_EMBEDDING_SIZE + OTHER_EMBEDDING_SIZE * 3
        elif USE_DATA == 'avazu':
            TOT_ITEM_EMBEDDING_SIZE = ITEM_EMBEDDING_SIZE + OTHER_EMBEDDING_SIZE * 4

        self.user_emb_w = nn.Embedding(USER_SIZE + 1, USER_EMBEDDING_SIZE)
        self.item_emb_w = nn.Embedding(ITEM_SIZE + 1, ITEM_EMBEDDING_SIZE)
        self.gender_emb_w = nn.Embedding(GENDER_DIM + 1, OTHER_EMBEDDING_SIZE)
        self.age_emb_w = nn.Embedding(AGE_DIM + 1, OTHER_EMBEDDING_SIZE)
        self.occupation_emb_w = nn.Embedding(OCCUPATION_DIM + 1, OTHER_EMBEDDING_SIZE)
        self.item_rate_emb_w = nn.Embedding(RATE_DIM + 1, OTHER_EMBEDDING_SIZE)
        self.item_year_emb_w = nn.Embedding(YEAR_DIM + 1, OTHER_EMBEDDING_SIZE)
        self.item_genre_emb_w = nn.Embedding(GENRE_DIM + 1, OTHER_EMBEDDING_SIZE)
        self.item_director_emb_w = nn.Embedding(DIRECTOR_DIM + 1, OTHER_EMBEDDING_SIZE)

        self.FC1 = nn.Dense(din_all_size, FC1_SIZE, activation = 'relu')
        self.FC2 = nn.Dense(FC1_SIZE, FC2_SIZE, activation = 'relu')
        self.FC3 = nn.Dense(FC2_SIZE, FC3_SIZE, activation = 'relu')
        
        self.FC4 = nn.Dense(att_hist_embedding_size, TOT_ITEM_EMBEDDING_SIZE, activation = 'relu')
        
        self.FC5 = nn.Dense(tot_embedding_size, FC1_SIZE, activation = 'relu')
        self.FC6 = nn.Dense(FC1_SIZE, FC2_SIZE, activation = 'relu')
        self.FC7 = nn.Dense(FC2_SIZE, FC3_SIZE, activation = 'relu')

        self.LSTM = nn.LSTMCell(lstm_input_size, LSTM_1)

    def construct(self, user, targets, targets_labels, hists, items_length, hists_y):

        user_embedding = ops.concat([
            self.user_emb_w(user[:, 0]), 
            self.gender_emb_w(user[:, 1]), 
            self.age_emb_w(user[:, 2]), 
            self.occupation_emb_w(user[:, 3])], axis=1
        )
        
        target_embedding = ops.concat([self.item_emb_w(targets[:, 0]),
            self.item_rate_emb_w(targets[:, 1]),
            self.item_year_emb_w(targets[:, 2]),
            self.item_genre_emb_w(targets[:, 3]),
            self.item_director_emb_w(targets[:, 4])],
            axis=1
        )

        hist_embedding = ops.concat([
            self.item_emb_w(hists[:, :, 0]),
            self.item_rate_emb_w(hists[:, :, 1]),
            self.item_year_emb_w(hists[:, :, 2]),
            self.item_genre_emb_w(hists[:, :, 3]),
            self.item_director_emb_w(hists[:, :, 4])], axis=2
        )
        print("user_embedding size:", user_embedding.shape)
        print("target_embedding size:", target_embedding.shape)
        print("hist_embedding size:", hist_embedding.shape)

        # attention
        att_hist_embedding = self.attention(target_embedding, hist_embedding, items_length)
        # att_hist_embedding = tf.layers.batch_normalization(inputs=att_hist_embedding, name='bn_hist', reuse=tf.AUTO_REUSE) 
        att_hist_embedding = ops.reshape(att_hist_embedding, (-1, TOT_ITEM_EMBEDDING_SIZE))
        att_hist_embedding = self.FC4(att_hist_embedding)

        # rnn
        # rnn_hist_embedding = tf.layers.batch_normalization(inputs=hist_embedding, name='bn_hist', reuse=tf.AUTO_REUSE) 
        rnn_hist_embedding = hist_embedding
        zeros = mindspore.Tensor(np.zeros((20, 128)).astype(np.float32))
        lstm_1 = [self.LSTM(i, (zeros, zeros))[0] for i in rnn_hist_embedding]
        lstm_1 = ops.stack(tuple(lstm_1), axis=0)
        
        mask = sequence_mask(items_length, lstm_1.shape[1]).astype(mindspore.float32)  # [batch_size,N]
        mask = ops.expand_dims(mask, -1)  # [batch_size,N, 1]
        mask = ops.tile(mask, (1, 1, lstm_1.shape[2]))  # [batch_size,N, ITEM_EMBEDDING_SIZE+CATE_EMBEDDING_SIZE]
        lstm_1 *= mask  # [batch_size,N,ITEM_EMBEDDING_SIZE+CATE_EMBEDDING_SIZE]
        lstm_1 = ops.reduce_sum(lstm_1, 1)
        lstm_1 = lstm_1 / ops.cast(ops.tile(ops.expand_dims(items_length, 1), (1, LSTM_1)), mindspore.float32)

        # mlp
        tot_embedding = ops.concat((user_embedding, att_hist_embedding, lstm_1, target_embedding), axis=1)
        # tot_embedding = tf.layers.batch_normalization(inputs=tot_embedding, name='BN_MLP_input', reuse=tf.AUTO_REUSE) 
        fc1_output = self.FC5(tot_embedding)
        if self.drop_keep_prob < 1:
            fc1_output = ops.dropout(fc1_output, p=1-self.drop_keep_prob)
        fc2_output = self.FC6(fc1_output)
        if self.drop_keep_prob < 1:
            fc2_output = ops.dropout(fc2_output, p=1-self.drop_keep_prob)
        fc3_output = self.FC7(fc2_output)
        fc3_output = ops.reshape(fc3_output, [-1])

        # item_bias = tf.get_variable("item_bias", [ITEM_SIZE + 1], trainable=self.trainable,
        #                                 initializer=tf.constant_initializer(0.0))
        #     batch_bias = tf.gather(item_bias, self.target_item[:, 0])
        #     self.logits = batch_bias + fc3_output
        logits = fc3_output
        predicted = ops.sigmoid(logits)
        return predicted, 0

    def attention(self, target_item, hist_items, items_length):

        hidden_units = target_item.shape[-1]
        hist_items_length = hist_items.shape[1]
        target_item = ops.tile(target_item, (1, hist_items_length))
        target_item = ops.reshape(target_item, (-1, hist_items_length, hidden_units))

        din_all = ops.concat((target_item, hist_items, target_item - hist_items, target_item * hist_items), axis=-1)        
        d_layer_3_all = self.FC3(self.FC2(self.FC1(din_all)))
        d_layer_3_all = ops.reshape(d_layer_3_all, (-1, 1, hist_items_length))
        outputs = d_layer_3_all

        
        key_masks = sequence_mask(items_length, hist_items_length)
        key_masks = ops.expand_dims(key_masks, 1)
        

        paddings = ops.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = ops.where(key_masks, outputs, paddings) 
        outputs = outputs / (hidden_units ** 0.5) 
        outputs = nn.Softmax()(outputs)  # [B, 1, N]        
        outputs = ops.matmul(outputs, hist_items) 
        return outputs