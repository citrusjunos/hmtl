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
from allennlp.models.crf_tagger import CrfTagger
from allennlp.modules.token_embedders import Embedding

from hmtl.modules.text_field_embedders import ShortcutConnectTextFieldEmbedder
from hmtl.modules.parsers.biaffine_dependency_parser_for_pipeline import BiaffineDependencyParserForPipeline

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("biaffine_for_pipeline")
class BiaffineDependencyParserForPipelineWrapper(Model):
    def __init__(self, vocab: Vocabulary, params: Params, regularizer: RegularizerApplicator = None):

        super(BiaffineDependencyParserForPipelineWrapper, self).__init__(vocab=vocab, regularizer=regularizer)

        # Base text Field Embedder
        text_field_embedder_params = params.pop("text_field_embedder")
        text_field_embedder = BasicTextFieldEmbedder.from_params(vocab=vocab, params=text_field_embedder_params)
        self._text_field_embedder = text_field_embedder

        ############################
        # Dependency Parsing Stuffs
        ############################

        dep_parsing_params = params.pop("dep_parsing", None)

        cuda_device = int(dep_parsing_params.pop("cuda_device"))

        span_emb_calc_method = dep_parsing_params.pop("span_emb_calc_method")

        # Encoder
        encoder_dep_parsing_params = dep_parsing_params.pop("encoder")
        encoder_dep_parsing = Seq2SeqEncoder.from_params(encoder_dep_parsing_params)
        self._encoder_dep_parsing = encoder_dep_parsing

        # Tagger: Dependency Parsing

        n_pos_tag_vocab = 26

        tagger_dep_parsing = BiaffineDependencyParserForPipeline(
            vocab=vocab,
            text_field_embedder=self._text_field_embedder,
            encoder=self._encoder_dep_parsing,
            tag_representation_dim = 100,
            arc_representation_dim = 500,
            tag_feedforward = None,
            arc_feedforward = None,
            pos_tag_embedding = None,
            dropout = 0.33,
            input_dropout = 0.33,
            initializer = InitializerApplicator(),
            regularizer = None,
            cuda_device = cuda_device,
            span_emb_calc_method = span_emb_calc_method
        )
        self._tagger_dep_parsing = tagger_dep_parsing

        logger.info("Biaffine Model has been instantiated.")

    @overrides
    def forward(self, tensor_batch, for_training: bool = False, task_name: str = "ner") -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ

        tagger = getattr(self, "_tagger_%s" % task_name)

        return tagger.forward(**tensor_batch)

    @overrides
    def get_metrics(self, task_name: str, reset: bool = False, full: bool = False) -> Dict[str, float]:

        task_tagger = getattr(self, "_tagger_" + task_name)
        return task_tagger.get_metrics(reset)

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params, regularizer: RegularizerApplicator) -> "HMTL":
        return cls(vocab=vocab, params=params, regularizer=regularizer)
