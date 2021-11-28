# coding: utf-8

import logging
from typing import Dict, List, Iterable, Iterator

from overrides import overrides
import codecs

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, OntonotesSentence

from hmtl.dataset_readers.dataset_utils import parse_conll


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("fixed_mwe_ud")
class FixedMweUdConllReader(DatasetReader):
    """
    A dataset reader to read BIOUL tags of fixed-mwe spans from a UD dataset
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        label_namespace: str = "fixed_mwe_labels",
        pos_tag_namespace: str = "pos_tags",
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._label_namespace = label_namespace
        self._pos_tag_namespace = pos_tag_namespace

    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)  # if `file_path` is a URL, redirect to the cache
        logger.info("Reading fixed-MWE annotated sentences from dataset files at: %s", file_path)

        snt_infos = parse_conll(file_path,
                                        header_d = {"id":0,
                                                    "form" : 1,
                                                    "upos" : 3,
                                                    "bio_tags": 13})

        for snt_info in snt_infos:
            tokens = [Token(t) for t in snt_info["form"]]
            pos_tags = snt_info["upos"]
            bio_tags = snt_info["bio_tags"]

            yield self.text_to_instance(tokens, pos_tags, bio_tags)

    def text_to_instance(self, tokens: List[Token], pos_tags: List[str], bio_tags: List[str]) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields["tokens"] = text_field

        fields["tags"] = SequenceLabelField(
            labels=bio_tags, sequence_field=text_field, label_namespace=self._label_namespace
        )

        return Instance(fields)
