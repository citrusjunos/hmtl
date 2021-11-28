# coding: utf-8

import os
import sys
import logging
from typing import Dict
from overrides import overrides

import torch

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import RegularizerApplicator, InitializerApplicator
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules import FeedForward
from allennlp.models.biaffine_dependency_parser import BiaffineDependencyParser
from allennlp.modules.token_embedders import Embedding


from hmtl.modules.text_field_embedders import ShortcutConnectTextFieldEmbedder
from hmtl.modules.taggers import FixedMweCrfTagger

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("hmtl_mwe_aware_parsing_on_fixed_mwer")
class HMTL_mwe_aware_parsing_on_fixed_mwer(Model):
    """
    A class that implement the full HMTL model.
    
    Parameters
    ----------
    vocab: ``allennlp.data.Vocabulary``, required.
        The vocabulary fitted on the data.
    params: ``allennlp.common.Params``, required
        Configuration parameters for the multi-task model.
    regularizer: ``allennlp.nn.RegularizerApplicator``, optional (default = None)
        A reguralizer to apply to the model's layers.
    """

    def __init__(self, vocab: Vocabulary, params: Params, regularizer: RegularizerApplicator = None):

        super(HMTL_mwe_aware_parsing_on_fixed_mwer, self).__init__(vocab=vocab, regularizer=regularizer)

        # Base text Field Embedder
        text_field_embedder_params = params.pop("text_field_embedder")
        text_field_embedder = BasicTextFieldEmbedder.from_params(vocab=vocab, params=text_field_embedder_params)
        self._text_field_embedder = text_field_embedder

        ############
        # NER Stuffs
        ############

        ner_params = params.pop("ner", None)
        tmp_cuda_device = int(ner_params.pop("cuda_device"))
        logger.info("--- cuda_device:%d"%(tmp_cuda_device))

        # Encoder
        encoder_ner_params = ner_params.pop("encoder")
        encoder_ner = Seq2SeqEncoder.from_params(encoder_ner_params)
        self._encoder_ner = encoder_ner

        # Tagger NER - CRF Tagger
        tagger_ner_params = ner_params.pop("tagger")

        tagger_ner = FixedMweCrfTagger(
            vocab=vocab,
            text_field_embedder=self._text_field_embedder,
            encoder=self._encoder_ner,
            calculate_span_f1 = True,
            label_encoding = "BIOUL",
            label_namespace=tagger_ner_params.pop("label_namespace", "labels"),
            constraint_type=tagger_ner_params.pop("constraint_type", None),
            dropout=tagger_ner_params.pop("dropout", None),
            regularizer=regularizer,
            cuda_device=tmp_cuda_device
        )

        self._tagger_ner = tagger_ner

        ############################
        # Dependency Parsing Stuffs
        ############################

        dep_parsing_params = params.pop("dep_parsing", None)

        # Encoder
        encoder_dep_parsing_params = dep_parsing_params.pop("encoder")
        encoder_dep_parsing = Seq2SeqEncoder.from_params(encoder_dep_parsing_params)
        self._encoder_dep_parsing = encoder_dep_parsing

        shortcut_text_field_embedder_dep_parsing = ShortcutConnectTextFieldEmbedder(
           base_text_field_embedder=self._text_field_embedder, previous_encoders=[self._encoder_ner]
        )
        self._shortcut_text_field_embedder_dep_parsing = shortcut_text_field_embedder_dep_parsing

        # Tagger: Dependency Parsing

        tagger_dep_parsing = BiaffineDependencyParser(
            vocab=vocab,
            text_field_embedder=self._shortcut_text_field_embedder_dep_parsing,
            encoder=self._encoder_dep_parsing,
            tag_representation_dim = 100,
            arc_representation_dim = 500,
            tag_feedforward = None,
            arc_feedforward = None,
            pos_tag_embedding = None,
            dropout = 0.33,
            input_dropout = 0.33,
            initializer = InitializerApplicator(),
            regularizer = None
        )
        self._tagger_dep_parsing = tagger_dep_parsing


        logger.info("Multi-Task Learning Model has been instantiated.")

    @overrides
    def forward(self, tensor_batch, for_training: bool = False, task_name: str = "ner") -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ

        tagger = getattr(self, "_tagger_%s" % task_name)

        if task_name == "coref" and tagger._eval_on_gold_mentions:
            if for_training:
                tagger._use_gold_mentions = False
            else:
                tagger._use_gold_mentions = True

        return tagger.forward(**tensor_batch)

    @overrides
    def get_metrics(self, task_name: str, reset: bool = False, full: bool = False) -> Dict[str, float]:

        task_tagger = getattr(self, "_tagger_" + task_name)
        if full and task_name == "coref":
            return task_tagger.get_metrics(reset=reset, full=full)
        else:
            return task_tagger.get_metrics(reset)

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params, regularizer: RegularizerApplicator) -> "HMTL":
        return cls(vocab=vocab, params=params, regularizer=regularizer)
