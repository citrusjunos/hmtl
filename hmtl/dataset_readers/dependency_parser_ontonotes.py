# coding: utf-8

import logging
from typing import Dict, List, Iterable

from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import iob1_to_bioul
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, OntonotesSentence

from hmtl.dataset_readers.dataset_utils import parse_conll

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("dep_parsing_ontonotes")
class DependencyOntonotesReader(DatasetReader):
    """
    An ``allennlp.data.dataset_readers.dataset_reader.DatasetReader`` for reading
    NER annotations in CoNll-formatted OntoNotes dataset.
    
    NB: This DatasetReader was implemented before the current implementation of 
    ``OntonotesNamedEntityRecognition`` in AllenNLP. It is thought doing pretty much the same thing.
    
    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Map a token to an id.
    domain_identifier : ``str``, optional (default = None)
        The subdomain to load. If None is specified, the whole dataset is loaded.
    label_namespace : ``str``, optional (default = "ontonotes_ner_labels")
        The tag/label namespace for the task/dataset considered.
    lazy : ``bool``, optional (default = False)
        Whether or not the dataset should be loaded in lazy way. 
        Refer to https://github.com/allenai/allennlp/blob/master/tutorials/getting_started/laziness.md
        for more details about lazyness.
    coding_scheme: ``str``, optional (default=``IOB1``)
        Specifies the coding scheme for ``ner_labels`` and ``chunk_labels``.
        Valid options are ``IOB1`` and ``BIOUL``.  The ``IOB1`` default maintains
        the original IOB1 scheme in the CoNLL data.
        In the IOB1 scheme, I is a token inside a span, O is a token outside
        a span and B is the beginning of span immediately following another
        span of the same type.
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        domain_identifier: str = None,
        label_namespace: str = "ontonotes_ner_labels",
        pos_tag_namespace: str = "pos_tags",
        head_tag_namespace: str = "head_tags",
        head_indices_namespace: str = "head_indices_labels",
        dict_feats_namespace: str = "dict_feat_tags",
        lazy: bool = False,
        coding_scheme: str = "IOB1",
    ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._domain_identifier = domain_identifier
        self._pos_tag_namespace = pos_tag_namespace
        self._label_namespace = label_namespace
        self._head_tag_namespace = head_tag_namespace
        self._head_indices_namespace = head_indices_namespace
        self._dict_feats_namespace = dict_feats_namespace

    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)  # if `file_path` is a URL, redirect to the cache
        logger.info("Reading dependency trees from dataset files at: %s", file_path)

        snt_infos = parse_conll(file_path,
                                        header_d = {"id":0,
                                                    "form" : 1,
                                                    "lemma" : 2,
                                                    "upos" : 3,
                                                    "xpos" : 4,
                                                    "head" : 6,
                                                    "deprel" : 7
                                                   })

        for snt_info in snt_infos:
            tokens = [Token(t) for t in snt_info["form"]]

            pos_tags = snt_info["upos"]
            head_tags = snt_info["deprel"]
            head_indices = [int(e) for e in snt_info["head"]]

            yield self.text_to_instance(tokens, pos_tags, head_tags, head_indices)

    @staticmethod
    def _ontonotes_subset(
        ontonotes_reader: Ontonotes, file_path: str, domain_identifier: str
    ) -> Iterable[OntonotesSentence]:
        for conll_file in ontonotes_reader.dataset_path_iterator(file_path):
            yield from ontonotes_reader.sentence_iterator(conll_file)

    def text_to_instance(self, tokens: List[Token], pos_tags: List[str],
                         head_tags: List[str],
                         head_indices: List[str]) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields["words"] = text_field
        fields["pos_tags"] = SequenceLabelField(
            labels=pos_tags, sequence_field=text_field, label_namespace=self._pos_tag_namespace
        )

        fields["head_tags"] = SequenceLabelField(
            labels=head_tags, sequence_field=text_field, label_namespace=self._head_tag_namespace
        )

        fields["head_indices"] = SequenceLabelField(
            labels=head_indices, sequence_field=text_field, label_namespace=self._head_indices_namespace
        )

        meta_fields = {"words":fields["words"],"pos":fields["pos_tags"]} 
        fields["metadata"] = MetadataField(meta_fields)

        return Instance(fields)
