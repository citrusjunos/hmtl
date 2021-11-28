# coding: utf-8

import os
import argparse
from typing import List, Dict, Any, Iterable
import torch
import spacy
import re
import numpy as np

from allennlp.data.fields import Field, TextField, ArrayField, SequenceLabelField, MetadataField, SpanField, ListField
from allennlp.modules.span_extractors import SpanExtractor, SelfAttentiveSpanExtractor
from allennlp.models.model import Model
from allennlp.common.params import Params
from allennlp.data import Vocabulary, Token, Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.fields import TextField, Field, ListField, SpanField
from allennlp.data.dataset import Batch
from allennlp.nn import util
from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.nn.util import get_text_field_mask, get_range_vector
from allennlp.nn.util import device_mapping, move_to_device
from allennlp.common.checks import check_for_gpu

import sys

sys.path.append("../")
from predictionFormatter import predictionFormatter

from hmtl.dataset_readers.dataset_utils import parse_conll

import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

torch.set_num_threads(1)


def load_model(model_path):
    """
    Load both vocabulary and model and create and instance of
    HMTL full model.
    """
    serialization_dir = os.path.dirname(model_path)
    params = Params.from_file(params_file=os.path.join(serialization_dir, "config.json"))

    cuda_device = params.pop("multi_task_trainer").pop_int("cuda_device", -1)

    # Load TokenIndexer
    task_keys = [key for key in params.keys() if re.search("^task_", key)]
    token_indexer_params = params.pop(task_keys[-1]).pop("data_params").pop("dataset_reader").pop("token_indexers")
    # see https://github.com/allenai/allennlp/issues/181 for better syntax
    token_indexers = {}
    for name, indexer_params in token_indexer_params.items():
        token_indexers[name] = TokenIndexer.from_params(indexer_params)

    # Load the vocabulary
    logger.info("Loading Vocavulary from %s", os.path.join(serialization_dir, "vocabulary"))
    vocab = Vocabulary.from_files(os.path.join(serialization_dir, "vocabulary"))
    logger.info("Vocabulary loaded")

    # Create model and load weights
    model_params = params.pop("model")
    model = Model.from_params(vocab=vocab, params=model_params, regularizer=None)
    model_state_path = model_path
    model_state = torch.load(model_state_path)
    model.load_state_dict(state_dict=model_state)

    model.cuda(cuda_device)

    return model, vocab, token_indexers, cuda_device


class HMTLPredictor:
    """
    Predictor class for HMTL full model.
    """

    def __init__(self, model_path=""):
        model, vocab, token_indexers, cuda_device = load_model(model_path)
        self.model = model
        self.vocab = vocab
        self.token_indexers = token_indexers
        self.formatter = predictionFormatter()
        self.nlp = spacy.load("en_core_web_sm")
        self.cuda_device = cuda_device

        sub_model = self.model._tagger_dep_parsing
        encoder_dim = sub_model.encoder.get_output_dim()
        self.extractor = sub_model.extractor

    def create_instance(self, snt_info):
        """
        Create an batch tensor from the input sentence.
        """
        merged_tokens = [Token(t) for t in snt_info["form"]]
        expanded_tokens = []
        mwe_mask = []
        primitive_spans = []
        expanded_idx = 0
        for i,t in enumerate(snt_info["form"]):
            elems = t.split("_")
            expanded_tokens += elems[:]
            if(len(elems)) < 2:
                mwe_mask.append(0)
                primitive_spans.append([expanded_idx, expanded_idx])
                expanded_idx += 1
            else:
                mwe_len = len(elems)
                mwe_mask.append(1)
                primitive_spans.append([expanded_idx, expanded_idx + mwe_len - 1])
                expanded_idx += mwe_len

        tokens = [Token(t) for t in expanded_tokens]

        pos_tags = snt_info["upos"]
        head_tags = snt_info["deprel"]
        head_indices = [int(e) for e in snt_info["head"]]

        instance = self.text_to_instance(mwe_mask, primitive_spans, merged_tokens, tokens, pos_tags, head_tags, head_indices)

        instances = [instance]
        batch = Batch(instances)
        batch.index_instances(self.vocab)
        pad_len = batch.get_padding_lengths()
        if pad_len["words"]["num_token_characters"] < 3:
            pad_len["words"]["num_token_characters"] = 3
        batch_tensor = batch.as_tensor_dict(pad_len)

        return batch_tensor

    def text_to_instance(self,
                         mwe_mask: List[int],
                         primitive_spans: List[List[int]],
			 merged_tokens: List[Token],
			 tokens: List[Token],
			 pos_tags: List[str],
                         head_tags: List[str],
                         head_indices: List[str]) -> Instance:

        label_namespace = "ontonotes_ner_labels"
        pos_tag_namespace = "pos_tags"
        head_tag_namespace = "head_tags"
        head_indices_namespace = "head_indices_labels"

        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        org_text_field = TextField(tokens, token_indexers=self.token_indexers)
        fields["words"] = org_text_field
        text_field = TextField(merged_tokens, token_indexers=self.token_indexers)
        fields["merged_words"] = text_field
        '''
		There is  a  few books  .
		   0   1  2   3    4    5
		
		primitive_spans = [ [0,0], [1,1], [2,3], [4,4], [5,5]]
        '''
        spans: List[Field] = []
        for primitive_span in primitive_spans:
            start, end = primitive_span
            spans.append(SpanField(start, end, org_text_field))
        fields["spans"] = ListField(spans)
        fields["mwe_mask"] = ArrayField(array=np.array(mwe_mask))
        fields["merged_pos_tags"] = SequenceLabelField(
            labels=pos_tags, sequence_field=text_field, label_namespace=pos_tag_namespace
        )

        meta_fields = {"words":fields["words"],"pos":fields["merged_pos_tags"]}
        fields["metadata"] = MetadataField(meta_fields)

        return Instance(fields)

    def predict(self, snt_info):
        check_for_gpu(self.cuda_device)
        with torch.no_grad():
            self.model.eval()
            required_tasks = ["dep_parsing"]
            batch_tensor = self.create_instance(snt_info)
            batch_tensor = util.move_to_device(batch_tensor, self.cuda_device)
            final_output = self.inference(batch=batch_tensor, required_tasks=required_tasks)
            return final_output

    def inference(self, batch, required_tasks):
        """
        Fast inference of HMTL.
        """
        # pylint: disable=arguments-differ

        final_output = {}

        output_parser = self.inference_parser(batch)
        decoding_dict_parser = self.decode(task_output=output_parser, task_name="dep_parsing")
        final_output["heads"] = decoding_dict_parser["predicted_heads"]
        final_output["deprels"] = decoding_dict_parser["predicted_dependencies"]

        return final_output

    def inference_parser(self, batch):
        sub_model = self.model._tagger_dep_parsing

        words = batch["words"]
        merged_words = batch["merged_words"]
        merged_pos_tags = batch["merged_pos_tags"]
        metadata = batch["metadata"]
        spans = batch["spans"]
        mwe_mask = batch["mwe_mask"]

        embedded_text_input = sub_model.text_field_embedder(words)
        if merged_pos_tags is not None and sub_model._pos_tag_embedding is not None:
            embedded_pos_tags = sub_model._pos_tag_embedding(merged_pos_tags)
            embedded_text_input = torch.cat([embedded_text_input, embedded_pos_tags], -1)
        elif sub_model._pos_tag_embedding is not None:
            raise ConfigurationError("Model uses a POS embedding, but no POS tags were passed.")

        mask = get_text_field_mask(words)
        merged_mask = get_text_field_mask(merged_words)

        embedded_text_input = sub_model._input_dropout(embedded_text_input)
        encoded_text = sub_model.encoder(embedded_text_input, mask)

        batch_size, _, encoding_dim = encoded_text.size()
        batch_size, max_merged_snt_len = merged_pos_tags.size()

        span_representations = self.extractor(encoded_text, spans)
        encoded_text = span_representations
        mask = merged_mask

        head_sentinel = sub_model._head_sentinel.expand(batch_size, 1, encoding_dim)
        # Concatenate the head sentinel onto the sentence representation.
        encoded_text = torch.cat([head_sentinel, encoded_text], 1)
        mask = torch.cat([mask.new_ones(batch_size, 1), mask], 1)
        float_mask = mask.float()
        encoded_text = sub_model._dropout(encoded_text)

        # shape (batch_size, sequence_length, arc_representation_dim)
        head_arc_representation = sub_model._dropout(sub_model.head_arc_feedforward(encoded_text))
        child_arc_representation = sub_model._dropout(sub_model.child_arc_feedforward(encoded_text))

        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag_representation = sub_model._dropout(sub_model.head_tag_feedforward(encoded_text))
        child_tag_representation = sub_model._dropout(sub_model.child_tag_feedforward(encoded_text))
        # shape (batch_size, sequence_length, sequence_length)
        attended_arcs = sub_model.arc_attention(head_arc_representation,
                                           child_arc_representation)

        minus_inf = -1e8
        minus_mask = (1 - float_mask) * minus_inf
        attended_arcs = attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        if sub_model.training or not sub_model.use_mst_decoding_for_validation:
            predicted_heads, predicted_head_tags = sub_model._greedy_decode(head_tag_representation,
                                                                       child_tag_representation,
                                                                       attended_arcs,
                                                                       mask)
        else:
            predicted_heads, predicted_head_tags = sub_model._mst_decode(head_tag_representation,
                                                                    child_tag_representation,
                                                                    attended_arcs,
                                                                    mask)

        output_dict = {
                "heads": predicted_heads,
                "head_tags": predicted_head_tags,
                "mask": mask,
                "words": [meta["words"] for meta in metadata],
                "pos": [meta["pos"] for meta in metadata]
                }

        return output_dict

    def inference_ner(self, batch):
        submodel = self.model._tagger_ner

        ### Fast inference of NER ###
        tokens = batch["tokens"]
        embedded_text_input_base = submodel.text_field_embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        encoded_text_ner = submodel.encoder(embedded_text_input_base, mask)

        logits = submodel.tag_projection_layer(encoded_text_ner)
        best_paths = submodel.crf.viterbi_tags(logits, mask)

        predicted_tags = [x for x, y in best_paths]

        output = {"tags": predicted_tags}

        return output, embedded_text_input_base, encoded_text_ner, mask


    def decode(self, task_output, task_name: str = "dep_parsing"):
        """
        Decode the predictions.
        """
        tagger = getattr(self.model, "_tagger_%s" % task_name)
        return tagger.decode(task_output)

def write_prediction(output, snt_info, f_out):
    pred_heads = output["heads"][0]
    pred_deprels = output["deprels"][0]

    ids = snt_info["id"]
    forms = snt_info["form"]
    lemmas = snt_info["lemma"]
    upos_seq = snt_info["upos"]
    xpos_seq = snt_info["xpos"]

    for token_id, form, lemma, upos, xpos, head, deprel in zip(ids, forms, lemmas, upos_seq, xpos_seq, pred_heads, pred_deprels):
        cols = [token_id, form, lemma, upos, xpos, "_", head, deprel, "_"]
        f_out.write("\t".join([ str(e) for e in cols])+"\n")
    f_out.write("\n")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_path", required=False, type=str, help="Model path"
    )

    parser.add_argument("-i","--test_conll_path",type=str)
    parser.add_argument("-o","--pred_test_outpath",type=str)

    args = parser.parse_args()

    hmtl_predictor = HMTLPredictor(model_path=args.model_path)
    hmtl = hmtl_predictor

    f_out = open(args.pred_test_outpath, "w")

    snt_infos = parse_conll(args.test_conll_path,
                            header_d = {"id":0,
                                        "form" : 1,
                                        "lemma" : 2,
                                        "upos" : 3,
                                        "xpos" : 4,
                                        "head" : 6,
                                        "deprel" : 7})

    for snt_info in snt_infos:
        output = hmtl.predict(snt_info)
        write_prediction(output, snt_info, f_out)

