from mindspore import nn
import mindspore
from transformers.modeling_bert import BertModel
from transformers.modeling_bert import BertPreTrainedModel
from .MI_estimators import VIB, vCLUB, InfoNCE
from .classifier import MultiNonLinearClassifier
from .model_utils import span_select
from .span_layer import SpanLayer

ANNEALING_RATIO = 0.3
SMALL = 1e-08
SAMPLE_SIZE = 5

# span embedding settings
MAX_SPAN_LEN = 4
MORPH_NUM = 5
MAX_SPAN_NUM = 502
TOKEN_LEN_DIM = 50
SPAN_LEN_DIM = 50
SPAN_MORPH_DIM = 100
BOTTLE_NECK_DIM = 50


class BertSpanNerBN(BertPreTrainedModel):
    def __init__(
        self,
        config,
        args=None,
        num_labels=None
    ):
        super(BertSpanNerBN, self).__init__(config)

        # ---------------- encoder ------------------
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cross_entropy = mindspore.nn.CrossEntropyLoss(reduction='none')
        self.n_class = 5 if not num_labels else num_labels

        # ---------------- span layer ------------------
        self.span_layer = SpanLayer(config.hidden_size, TOKEN_LEN_DIM, SPAN_LEN_DIM,
                                    SPAN_MORPH_DIM, MAX_SPAN_LEN, MORPH_NUM)
        # start + end + token len + span len + morph
        span_dim = config.hidden_size * 2 + TOKEN_LEN_DIM \
                   + SPAN_LEN_DIM + SPAN_MORPH_DIM

        # ---------------- classifier ------------------
        self.span_classifier = MultiNonLinearClassifier(
            span_dim, self.n_class, dropout_rate=0.2
        )
        self.softmax = mindspore.nn.Softmax(dim=-1)

        # ---------------- info bottleneck ------------------
        # self.sample_size = SAMPLE_SIZE
        # self.r_mean = nn.Parameter(mindspore.randn(MAX_SPAN_NUM, BOTTLE_NECK_DIM))
        # self.r_std = nn.Parameter(mindspore.randn(MAX_SPAN_NUM, BOTTLE_NECK_DIM))

        self.bn_encoder = VIB(
            embedding_dim=span_dim,
            hidden_dim=(BOTTLE_NECK_DIM + span_dim) // 2,
            tag_dim=BOTTLE_NECK_DIM,
            device=args.device
        )

        # ---------------- OOV regular ------------------
        self.beta = args.beta  # oov regular weights
        self.oov_reg = vCLUB()

        # ---------------- z infoNCE regular ------------------
        self.gama = args.gama
        self.z_reg = InfoNCE(span_dim, span_dim, args.device)

        self.init_weights()

    def forward(self, ori_feas, cont_feas):
        """
        默认有 labels 为 train, 无 labels 为 test
        """
        ori_encoding, cont_encoding = {}, {}
        ori_encoding['spans_rep'] = self.span_encoding(**ori_feas)
        cont_encoding['spans_rep'] = self.span_encoding(**cont_feas)
        ori_encoding['logits'] = self.span_classifier(ori_encoding['spans_rep'])
        # cont_encoding['logits'] = self.span_classifier(cont_encoding['spans_rep'])

        loss_dict = {}
        outputs = [self.softmax(ori_encoding['logits'])]

        if ori_feas['span_labels'] is not None:
            # TODO, test stop gradient of cont features
            # cont_encoding['spans_rep'].detach()
            loss_dict = self.compute_loss(ori_feas, ori_encoding, cont_feas, cont_encoding)

        return outputs, loss_dict  # (loss), scores

    def span_encoding(self, input_ids=None, input_mask=None, segment_ids=None,
                      span_token_idxes=None, span_lens=None, morph_idxes=None, **kwargs):
        """
        Encode tokens by Bert, and get span representations.
        """
        # encoder [batch, seq_len, hidden]
        sequence_output = self.bert(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids
        )

        span_rep = self.span_layer(sequence_output[0], span_token_idxes.long(), span_lens, morph_idxes)

        return span_rep

    def compute_loss(self, ori_feas, ori_encoding, cont_feas, cont_encoding):
        """
        计算loss, 包括: span分类loss, gi loss, si loss
        """
        # ----------compute span classification loss----------
        loss_dic = {'c': self.compute_clas_loss(ori_feas, ori_encoding),
                    # 'cc': self.compute_clas_loss(cont_feas, cont_encoding)
                    }

        # ----------compute KL(p(z_o|t), p(z_c|t)) loss----------
        x_span_cont, y_span_cont = span_select(
            ori_encoding['spans_rep'].unsqueeze(1), ori_feas['cont_span_idx'],
            cont_encoding['spans_rep'].unsqueeze(1), cont_feas['cont_span_idx']
        )

        entity_dist = self.oov_reg.update(x_span_cont, y_span_cont)
        loss_dic['si'] = self.beta * entity_dist

        # ----------compute MI(z_o, z_c) loss----------
        entity_mi = self.gama * self.z_reg(x_span_cont, y_span_cont)
        loss_dic['gi'] = entity_mi

        # ----------sum loss----------
        loss_dic['loss'] = sum([item[1] for item in loss_dic.items()])

        return loss_dic

    def compute_clas_loss(self, features, encoding):
        """
        计算分类loss.
        """
        batch_size, n_span = features['span_labels'].size()
        ori_span_rep = encoding['logits'].view(-1, self.n_class)
        ori_span_labels = features['span_labels'].view(-1)

        clas_loss = self.cross_entropy(ori_span_rep, ori_span_labels)
        clas_loss = clas_loss.view(batch_size, n_span) * features['span_weights']
        clas_loss = mindspore.masked_select(clas_loss, features['span_masks'].bool()).mean()

        return clas_loss
