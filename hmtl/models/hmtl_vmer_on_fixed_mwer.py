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
from hmtl.modules.taggers import VmweCrfTagger, FixedMweCrfTagger

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("hmtl_vmer_on_fixed_mwer")
class HMTL_vmer_on_fixed_mwer(Model):
    def __init__(self, vocab: Vocabulary, params: Params, regularizer: RegularizerApplicator = None):

        super(HMTL_vmer_on_fixed_mwer, self).__init__(vocab=vocab, regularizer=regularizer)

        # Base text Field Embedder
        text_field_embedder_params = params.pop("text_field_embedder")
        text_field_embedder = BasicTextFieldEmbedder.from_params(vocab=vocab, params=text_field_embedder_params)
        self._text_field_embedder = text_field_embedder

        ############
        # NER Stuffs
        ############

        ner_params = params.pop("ner", None)

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
            cuda_device=ner_params.pop("cuda_device")
        )

        self._tagger_ner = tagger_ner

        ### VMWE tagger
        vmwer_params = params.pop("vmwer", None)

        # Encoder
        encoder_vmwer_params = vmwer_params.pop("encoder")
        encoder_vmwer = Seq2SeqEncoder.from_params(encoder_vmwer_params)
        self._encoder_vmwer = encoder_vmwer

        tagger_vmwer_params = vmwer_params.pop("tagger")

        shortcut_text_field_embedder_vmwer = ShortcutConnectTextFieldEmbedder(
           base_text_field_embedder=self._text_field_embedder, previous_encoders=[self._encoder_ner]
        )
        self._shortcut_text_field_embedder_vmwer = shortcut_text_field_embedder_vmwer

        tagger_vmwer = VmweCrfTagger(
            vocab=vocab,
            text_field_embedder=self._shortcut_text_field_embedder_vmwer,
            encoder=self._encoder_vmwer,
            calculate_span_f1 = True,
            label_namespace=tagger_vmwer_params.pop("label_namespace", "labels"),
            constraint_type=tagger_vmwer_params.pop("constraint_type", None),
            dropout=tagger_vmwer_params.pop("dropout", None),
            regularizer=regularizer,
            cuda_device=vmwer_params.pop("cuda_device")
        )

        self._tagger_vmwer = tagger_vmwer

        logger.info("Multi-Task Learning Model has been instantiated.")

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
