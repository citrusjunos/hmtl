# coding: utf-8

import logging
from typing import Dict, List, Iterable, Iterator

from overrides import overrides
import codecs
import re

import numpy as np

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import iob1_to_bioul
from allennlp.data.fields import Field, TextField, SequenceLabelField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, OntonotesSentence

from hmtl.dataset_readers.dataset_utils import parse_conll

RE_TAGGING = re.compile(r'^(O|B(o|b(i[_~])+|I[_~])*(I[_~])+)+$')
# don't support plain I and i
STRENGTH = {'I_': '_', 'I~': '~', 'i_': '_', 'i~': '~', 'B': None, 'b': None, 'O': None, 'o': None}
# don't support plain I and i

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def form_groups(links):
    """
    >>> form_groups([(1, 2), (3, 4), (2, 5), (6, 8), (4, 7)])==[{1,2,5},{3,4,7},{6,8}]
    True
    """
    groups = []
    groupMap = {} # offset -> group containing that offset
    for a,b,mwepos in links:
        assert a is not None and b is not None,links
        assert b not in groups,'Links not sorted left-to-right: '+repr((a,b))
        if a not in groupMap: # start a new group
            groups.append({"mwepos":mwepos, "elems":{a}})
            groupMap[a] = groups[-1]
        else:
            if groupMap[a]["mwepos"] != mwepos:
                print("inconsistent mwepos! : (%s, %s)"%(groupMap[a]["mwepos"], mwepos))
        assert b not in groupMap[a],'Redundant link?: '+repr((a,b))
        groupMap[a]["elems"].add(b)
        groupMap[b] = groupMap[a]
    return groups

def get_groups(goldmwetags):
    #gold_mwe_tags_without_mwepos = [ e.split("-")[0] for e in goldmwetags ]
    glinks = []
    g_last_BI = None
    g_last_bi = None
    for j,(goldTag) in enumerate(goldmwetags):
        g_last_BI, g_last_bi, glinks = update_info(j, goldTag, g_last_BI, g_last_bi, glinks)

    glinks1 = [(a,b,mwepos) for a,b,mwepos, s in glinks]
    ggroups1 = form_groups(glinks1)
    return ggroups1


def update_info(j, orgTag, last_BI, last_bi, links):

    elems = orgTag.split("-")
    if len(elems)>1:
        mwepos = elems[1]
        tmpTag = elems[0]
    else:
        tmpTag = orgTag

    if tmpTag in {'I','I_','I~'}:
        links.append((last_BI, j, mwepos, STRENGTH[elems[0]]))
        last_BI = j
    elif tmpTag=='B':
        last_BI = j
    elif tmpTag in {'i','i_','i~'}:
        links.append((last_bi, j, mwepos, STRENGTH[elems[0]]))
        last_bi = j
    elif tmpTag=='b':
        last_bi = j

    return last_BI, last_bi, links


@DatasetReader.register("vmwe")
class VerbalMweConllReader(DatasetReader):
    """
    A dataset reader to read extended BIO tags of VMWE gappy spans from an Ontonotes dataset
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        label_namespace: str = "vmwe_labels",
        pos_tag_namespace: str = "pos_tags",
        lazy: bool = False,
        bio_tag_index: int = 9,
    ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._label_namespace = label_namespace
        self._pos_tag_namespace = pos_tag_namespace
        self.bio_tag_index = bio_tag_index

    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)  # if `file_path` is a URL, redirect to the cache
        logger.info("Reading VMWE annotated sentences from dataset files at: %s", file_path)

        snt_infos = parse_conll(file_path,
                                        header_d = {"id":0,
                                                    "form" : 1,
                                                    "upos" : 3,
                                                    "extended_bio_tags": self.bio_tag_index})

        for snt_info in snt_infos:
            #tokens = [ Token(text = form, pos = upos) for form, upos in zip(snt_info["form"], snt_info["upos"]) ]
            tokens = [Token(t) for t in snt_info["form"]]

            pos_tags = snt_info["upos"]
            extended_bio_tags = snt_info["extended_bio_tags"]

            yield self.text_to_instance(tokens, pos_tags, extended_bio_tags)

    def text_to_instance(self, tokens: List[Token], pos_tags: List[str], extended_bio_tags: List[str]) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields["tokens"] = text_field

        fields["tags"] = SequenceLabelField(
            labels=extended_bio_tags, sequence_field=text_field, label_namespace=self._label_namespace
        )

        if len(get_groups(extended_bio_tags)) > 0:
            flg_vmwe = 1
        else:
            flg_vmwe = 0

        fields["vmwe_flgs"] = ArrayField(array=np.array([flg_vmwe]))

        return Instance(fields)
