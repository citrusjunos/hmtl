# coding: utf-8

import os
import argparse
from typing import List, Dict, Any, Iterable
import torch
import spacy
import re

from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.models.model import Model
from allennlp.common.params import Params
from allennlp.data import Vocabulary, Token, Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.fields import TextField, Field, ListField, SpanField
from allennlp.data.dataset import Batch
from allennlp.nn import util
from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.nn.util import get_text_field_mask, get_range_vector
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

    def create_instance(self, snt_info):
        """
        Create an batch tensor from the input sentence.
        """
        tokens = [Token(t) for t in snt_info["form"]]

        pos_tags = snt_info["upos"]
        bio_tags = snt_info["bio_tags"]

        instance = self.text_to_instance(tokens, pos_tags, bio_tags)

        instances = [instance]
        batch = Batch(instances)
        batch.index_instances(self.vocab)
        pad_len = batch.get_padding_lengths()
        if pad_len["tokens"]["num_token_characters"] < 3:
            pad_len["tokens"]["num_token_characters"] = 3
        batch_tensor = batch.as_tensor_dict(pad_len)

        return batch_tensor

    def text_to_instance(self, tokens: List[Token], pos_tags: List[str],
                         bio_tags: List[str]) -> Instance:

        fields: Dict[str, Field] = {}
        text_field = TextField(tokens, token_indexers=self.token_indexers)
        fields["tokens"] = text_field

        return Instance(fields)

    def predict(self, snt_info):
        check_for_gpu(self.cuda_device)
        with torch.no_grad():
            self.model.eval()
            required_tasks = ["ner"]
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

        ### Fast inference of NER ###
        output_ner, embedded_text_input_base, encoded_text_ner, mask = self.inference_ner(batch)
        decoding_dict_ner = self.decode(task_output=output_ner, task_name="ner")
        final_output["ner"] = decoding_dict_ner["tags"]

        return final_output

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

    def decode(self, task_output, task_name: str = "ner"):
        """
        Decode the predictions.
        """
        tagger = getattr(self.model, "_tagger_%s" % task_name)
        return tagger.decode(task_output)

def write_prediction(output, snt_info, f_out):
    pred_bio_tags = output["ner"][0]

    ids = snt_info["id"]
    forms = snt_info["form"]
    lemmas = snt_info["lemma"]
    upos_seq = snt_info["upos"]
    xpos_seq = snt_info["xpos"]
    heads = snt_info["head"]
    deprels = snt_info["deprel"]

    for token_id, form, lemma, upos, xpos, head, deprel, pred_bio_tag in \
            zip(ids, forms, lemmas, upos_seq, xpos_seq, heads, deprels, pred_bio_tags):
        cols = [token_id, form, lemma, upos, xpos, "_", head, deprel, pred_bio_tag]
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
                                        "deprel" : 7,
                                        "bio_tags": 8})

    for snt_info in snt_infos:
        output = hmtl.predict(snt_info)
        write_prediction(output, snt_info, f_out)

