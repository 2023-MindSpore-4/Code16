import os
import copy
import logging
import random
import json

from .typos import typos
from .utils_metrics import get_entities
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, words, bio_labels, tokenizer=None, max_len=128):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.

        """
        self.guid = guid
        self.words = words
        self.bio_labels = bio_labels
        # initialize tokens, valid_mask, and entities
        self.tokens, self.valid_mask, self.offsets = self._tokenize(tokenizer)
        self.entities = self._get_entities(max_len)

    def _tokenize(self, tokenizer=None):
        tokenizer = tokenizer if tokenizer else AutoTokenizer.from_pretrained('bert-base-uncased')
        valid_mask = []
        tokens = []
        offsets = []

        for word in self.words:
            subwords = tokenizer.tokenize(word)
            # [(word1_left, word1_right), (word2_left, word2_right)...]
            offsets.append([len(tokens), len(tokens) + len(subwords) - 1])
            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            for i, word_token in enumerate(subwords):
                valid_mask.append(int(i == 0))
                tokens.append(word_token)
        return tokens, valid_mask, offsets

    def _get_entities(self, max_len=128):
        bio_labels = self.bio_labels
        entities = get_entities(self.bio_labels)

        for ent_type, start, end in entities:
            # cant assure entity position not out of index
            if start <= max_len - 1 and end > max_len - 1:
                bio_labels = bio_labels[:start-1] + ['O'] * (max_len-start)

        # clip items to avoid bug
        self.tokens = self.tokens[: max_len]
        self.valid_mask = self.valid_mask[: max_len]

        return get_entities(bio_labels)


def build_contrast_examples(
        examples,
        tokenizer,
        cont_ent_idxes,
        pmi_json=None,
        preserve_ratio=0.3,
        entity_json=None,
        switch_ratio=0.5,
        max_seq_len=128
):
    """
    Provide two ways to create contrastive samples:
        1> replace entity words with similar entity words;
        2> add typos to entity words.

    Args:
        examples: list of InputExamples.
        tokenizer: tokenizer of pre-trained models.
        cont_ent_idxes: indicate replace which entities.
        pmi_json: subwords json which was ranked by PMI(subword, label).
        preserve_ratio: how much subwords were not convert to typos version.
        entity_json: entity json which contains their similar entities.
        switch_ratio: how much candidates were generated by typos operation.
            0 means generated by typos, while 1 means similar entities.
    Returns:

    """
    with open(pmi_json, "r+", encoding='utf-8') as fo:
        os.path.exists(pmi_json)
        pmi_json = json.load(fo)
        pmi_filter = {k: v[:int(len(v) * preserve_ratio)] for k, v in
                      pmi_json.items()}

    with open(entity_json, "r+", encoding='utf-8') as fo:
        os.path.exists(entity_json)
        entity_json = json.load(fo)

    contrast_examples = []

    for i, example in enumerate(examples):
        if entity_json and random.random() < switch_ratio:
            cont_example = rand_typos(example, tokenizer, cont_ent_idxes[i], pmi_filter, max_seq_len)
        else:
            cont_example = sim_ent_rep(example, tokenizer, cont_ent_idxes[i], entity_json, max_seq_len)
        contrast_examples.append(cont_example)

    return contrast_examples


def rand_replace(rep_fun):
    """
    Wrapper of contrastive sample generate functions.
    """

    def gen_neg_subwords(example, tokenizer, cont_ent_idx, auxiliary_json=None, max_seq_len=128):
        cont_example = copy.deepcopy(example)
        entities = cont_example.entities

        if not entities:
            return copy.deepcopy(example)

        # replaced entity idx should be set
        label, start, end = entities[cont_ent_idx]
        rep_words = rep_fun(example, [label, start, end], tokenizer, auxiliary_json)

        cont_words = cont_example.words[:start] + rep_words + cont_example.words[end+1:]
        cont_bio_labels = cont_example.bio_labels[:start] + \
                          ["B-" + label] + ["I-" + label] * (len(rep_words) -1) + \
                          cont_example.bio_labels[end + 1:]
        cont_example = InputExample(guid=example.guid, words=cont_words, bio_labels=cont_bio_labels,
                                    tokenizer=tokenizer, max_len=max_seq_len)

        return cont_example

    return gen_neg_subwords


@rand_replace
def sim_ent_rep(example, entity, tokenizer, auxiliary_json=None, top_n=10, **kwargs):
    """
    entity: [entity_type, start_idx, end_idx] e.g. ['LOC', 3, 4]
    """
    i, j = entity[1:]
    tmp_word = "<split>".join(example.words[i: j + 1])

    # 编辑距离 top 10
    if tmp_word in auxiliary_json[entity[0]].keys() and len(auxiliary_json[entity[0]][tmp_word]) != 0:
        # [0:top_n] means top_n
        random_entity = random.choice(auxiliary_json[entity[0]][tmp_word][0:top_n])
    else:
        logger.warning('****Error-{} is not in auxiliary_json!***'.format(tmp_word))
        random_entity = tmp_word

    new_entity_words = random_entity.split("<split>")

    return new_entity_words


@rand_replace
def rand_typos(example, entity, tokenizer, PMI_filter=None, **kwargs):
    """
    只向 PMI 值低的 subword 中插入 typos

    """
    i, j = entity[1:]
    new_entity_words = []

    for token in example.words[i: j+1]:
        subwords = tokenizer.tokenize(token)
        # 保存的是一个词对应的subword
        new_subwords = []

        for subword in subwords:
            if subword in PMI_filter[entity[0]]:
                new_subwords.append(subword)
            else:
                # 判断 ##
                sub_wo_sig = subword[2:] if '##' == subword[:2] else subword
                rep_subwords = typos.get_candidates(sub_wo_sig, n=1)

                if rep_subwords:
                    rep_subword = rep_subwords[0]
                else:
                    rep_subword = sub_wo_sig

                if subword[:2] == '##':
                    rep_subword = '##' + rep_subword
                new_subwords.append(rep_subword)
        new_entity_words.append(reverse_bert_tokenize(new_subwords))

    assert(len(new_entity_words) == j-i+1)

    return new_entity_words


def reverse_bert_tokenize(segs):
    """
        Reverse subwords to original word.
        Adapt to bert, robera and albert.
    Args:
        segs: list of subwords

    Returns:
        a single word
    """

    text = ' '.join([x for x in segs])
    # skip bert concatenate chars
    text = text.replace(' ##', '')
    # skip albert concatenate chars
    if len(text) > 1 and text[0] == '_':
        text = text[1:]

    return text
