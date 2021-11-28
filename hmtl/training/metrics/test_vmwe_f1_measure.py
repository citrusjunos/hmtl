import sys,os
import re
from typing import Dict, List, Optional, Set, Callable
from collections import defaultdict, Counter


RE_TAGGING = re.compile(r'^(O|B(o|b(i[_~])+|I[_~])*(I[_~])+)+$')
# don't support plain I and i
STRENGTH = {'I_': '_', 'I~': '~', 'i_': '_', 'i~': '~', 'B': None, 'b': None, 'O': None, 'o': None}
# don't support plain I and i

flg_link_based = False

def getIntersection(pgroups1, ggroups1):
    tmpIntersection = sum(1 for pgrp in pgroups1
                            for ggrp in ggroups1
                                if pgrp["mwepos"] == ggrp["mwepos"]
                                    and pgrp["elems"] == ggrp["elems"]
                          )

    return tmpIntersection

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

def eval_sent_links(goldmwetags, predmwetags, counts):
   # Verify the MWE tag sequence is valid
    assert len(goldmwetags)==len(predmwetags)>0

    gold_mwe_tags_without_mwepos = [ e.split("-")[0] for e in goldmwetags ]
    pred_mwe_tags_without_mwepos = [ e.split("-")[0] for e in predmwetags ]

    print(gold_mwe_tags_without_mwepos)
    assert RE_TAGGING.match(''.join(gold_mwe_tags_without_mwepos))

    print(pred_mwe_tags_without_mwepos)
    if RE_TAGGING.match(''.join(pred_mwe_tags_without_mwepos)):
        flg_valid_prediction = True
    else:
        flg_valid_prediction = False

    #assert RE_TAGGING.match(''.join(predmwetags))
    # Sequences such as B I~ O I~ and O b i_ O are invalid.
    if flg_valid_prediction:
        # Construct links from BIO tags
        glinks, plinks = [], []
        g_last_BI, p_last_BI = None, None
        g_last_bi, p_last_bi = None, None
        for j,(goldTag,predTag) in enumerate(zip(goldmwetags, predmwetags)):
            assert goldTag.split("-")[0] in STRENGTH and predTag.split("-")[0] in STRENGTH
            g_last_BI, g_last_bi, glinks = update_info(j, goldTag, g_last_BI, g_last_bi, glinks)
            p_last_BI, p_last_bi, plinks = update_info(j, predTag, p_last_BI, p_last_bi, plinks)

        # Count link overlaps
        for d in ('Link+', 'Link-'):    # Link+ = strengthen weak links, Link- = remove weak links
            print(d)

            # for strengthened or weakened scores
            glinks1 = [(a,b,mwepos) for a,b,mwepos, s in glinks if d=='Link+' or s=='_']
            plinks1 = [(a,b,mwepos) for a,b,mwepos, s in plinks if d=='Link+' or s=='_']
            ggroups1 = form_groups(glinks1)
            print("ggroups1")
            print(ggroups1)
            print("+++++")

            pgroups1 = form_groups(plinks1)
            print("pgroups1")
            print(pgroups1)
            print("+++++")

            c = counts['MWE','Tags'][d]

            if flg_link_based:
                # soft matching (in terms of links)
                # precision and recall are defined structurally, not simply in terms of
                # set overlap (PNumer does not necessarily equal RNumer), so compare_sets_PRF doesn't apply
                c['PNumer'] += sum(1 for a,b,mwepos in plinks1 if any(a in grp["elems"] and b in grp["elems"] and grp["mwepos"]==mwepos for grp in ggroups1))
                c['PDenom'] += len(plinks1)
                c['RNumer'] += sum(1 for a,b,mwepos in glinks1 if any(a in grp["elems"] and b in grp["elems"] and grp["mwepos"]==mwepos for grp in pgroups1))
                c['RDenom'] += len(glinks1)

                c['PNumerVB'] += sum(1 for a,b,mwepos in plinks1 if any(a in grp["elems"] and b in grp["elems"] and grp["mwepos"]==mwepos and mwepos=="VB" for grp in ggroups1))
                c['PDenomVB'] += len([ a for a,b,mwepos in plinks1 if mwepos=="VB"])
                c['RNumerVB'] += sum(1 for a,b,mwepos in glinks1 if any(a in grp["elems"] and b in grp["elems"] and grp["mwepos"]==mwepos and mwepos=="VB" for grp in pgroups1))
                c['RDenomVB'] += len([ a for a,b,mwepos in glinks1 if mwepos=="VB"])

                c['PNumerCONT'] += sum(1 for a,b,mwepos in plinks1 if any(a in grp["elems"] and b in grp["elems"] and grp["mwepos"]==mwepos and mwepos!="VB" for grp in ggroups1))
                c['PDenomCONT'] += len([ a for a,b,mwepos in plinks1 if mwepos!="VB"])
                c['RNumerCONT'] += sum(1 for a,b,mwepos in glinks1 if any(a in grp["elems"] and b in grp["elems"] and grp["mwepos"]==mwepos and mwepos!="VB" for grp in pgroups1))
                c['RDenomCONT'] += len([ a for a,b,mwepos in glinks1 if mwepos!="VB"])
            else:
                # group-based F1
                tmpIntersection = getIntersection(pgroups1, ggroups1)
                c['PNumer'] += tmpIntersection
                c['PDenom'] += len(pgroups1)
                c['RNumer'] += tmpIntersection
                c['RDenom'] += len(ggroups1)

                ggroups1_vb = [tmp_d for tmp_d in ggroups1 if tmp_d["mwepos"] == "VB"]
                pgroups1_vb = [tmp_d for tmp_d in pgroups1 if tmp_d["mwepos"] == "VB"]
                tmpIntersection_VB = getIntersection(pgroups1_vb, ggroups1_vb)
                c['PNumerVB'] += tmpIntersection_VB
                c['PDenomVB'] += len(pgroups1_vb)
                c['RNumerVB'] += tmpIntersection_VB
                c['RDenomVB'] += len(ggroups1_vb)

                ggroups1_cont = [tmp_d for tmp_d in ggroups1 if tmp_d["mwepos"] != "VB"]
                pgroups1_cont = [tmp_d for tmp_d in pgroups1 if tmp_d["mwepos"] != "VB"]
                tmpIntersection_CONT = getIntersection(pgroups1_cont, ggroups1_cont)
                c['PNumerCONT'] += tmpIntersection_CONT
                c['PDenomCONT'] += len(pgroups1_cont)
                c['RNumerCONT'] += tmpIntersection_CONT
                c['RDenomCONT'] += len(ggroups1_cont)

            print("----- various counts ---")
            print(c)

            c = counts['GappyMWE','Tags'][d]
            # cross-gap links only
            c['PNumer'] += sum((1 if b-a>1 else 0) for a,b,mwepos in plinks1 if any(a in grp["elems"] and b in grp["elems"] and grp["mwepos"]==mwepos for grp in ggroups1))
            c['PDenom'] += sum((1 if b-a>1 else 0) for a,b,mwepos in plinks1)
            c['RNumer'] += sum((1 if b-a>1 else 0) for a,b,mwepos in glinks1 if any(a in grp["elems"] and b in grp["elems"] and grp["mwepos"]==mwepos for grp in pgroups1))
            c['RDenom'] += sum((1 if b-a>1 else 0) for a,b,mwepos in glinks1)

    else:
        glinks = []
        g_last_BI = None
        g_last_bi = None
        for j,(goldTag) in enumerate(goldmwetags):
            assert goldTag.split("-")[0] in STRENGTH
            g_last_BI, g_last_bi, glinks = update_info(j, goldTag, g_last_BI, g_last_bi, glinks)

        # Count link overlaps
        for d in ('Link+', 'Link-'):    # Link+ = strengthen weak links, Link- = remove weak links

            # for strengthened or weakened scores
            glinks1 = [(a,b,mwepos) for a,b,mwepos, s in glinks if d=='Link+' or s=='_']
            ggroups1 = form_groups(glinks1)
            c = counts['MWE','Tags'][d]

            if flg_link_based:
                # soft matching (in terms of links)
                # precision and recall are defined structurally, not simply in terms of
                # set overlap (PNumer does not necessarily equal RNumer), so compare_sets_PRF doesn't apply
                c['RNumer'] += 0
                c['RDenom'] += len(glinks1)

                c = counts['GappyMWE','Tags'][d]
                # cross-gap links only
                c['RNumer'] += 0
                c['RDenom'] += sum((1 if b-a>1 else 0) for a,b,mwepos in glinks1)
            else:
                # group-based F1
                c['PNumer'] += 0
                c['PDenom'] += 0
                c['RNumer'] += 0
                c['RDenom'] += len(ggroups1)

                ggroups1_vb = [tmp_d for tmp_d in ggroups1 if tmp_d["mwepos"] == "VB"]
                tmpIntersection_VB = getIntersection(pgroups1_vb, ggroups1_vb)
                c['PNumerVB'] += 0
                c['PDenomVB'] += 0
                c['RNumerVB'] += 0
                c['RDenomVB'] += len(ggroups1_vb)

                ggroups1_cont = [tmp_d for tmp_d in ggroups1 if tmp_d["mwepos"] != "VB"]
                c['PNumerCONT'] += 0
                c['PDenomCONT'] += 0
                c['RNumerCONT'] += 0
                c['RDenomCONT'] += len(ggroups1_cont)

def f1(prec, rec):
    #return 2*prec*rec/(prec+rec) if float(prec+rec)>0 else float('nan')
    return 2*prec*rec/(prec+rec) if float(prec+rec)>0 else 0.0

class Ratio(object):
    '''
    Fraction that prints both the ratio and the float value.
    (fractions.Fraction reduces e.g. 378/399 to 18/19. We want to avoid this.)
    '''
    def __init__(self, numerator, denominator):
        self._n = numerator
        self._d = denominator
    def __float__(self):
        return self._n / self._d if self._d!=0 else float('nan')
    def __str__(self):
        return f'{float(self):.1%}'
    def __repr__(self):
        return f'{self.numeratorS}/{self.denominatorS}={self:.1%}'
    def __add__(self, v):
        if v==0:
            return self
        if isinstance(v,Ratio) and self._d==v._d:
            return Ratio(self._n + v._n, self._d)
        return float(self)+float(v)
    def __mul__(self, v):
        return Ratio(self._n * float(v), self._d)
    def __truediv__(self, v):
        return Ratio(self._n / float(v) if float(v)!=0 else float('nan'), self._d)
    __rmul__ = __mul__
    @property
    def numerator(self):
        return self._n
    @property
    def numeratorS(self):
        return f'{self._n:.2f}' if isinstance(self._n, float) else f'{self._n}'
    @property
    def denominator(self):
        return self._d
    @property
    def denominatorS(self):
        return f'{self._d:.2f}' if isinstance(self._d, float) else f'{self._d}'


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


def f1(prec, rec):
    #return 2*prec*rec/(prec+rec) if float(prec+rec)>0 else float('nan')
    return 2*prec*rec/(prec+rec) if float(prec+rec)>0 else 0.0


def calc_prec_recall(p_numer, p_denom, r_numer, r_denom):

    if p_denom > 0:
        tmp_prec = float(p_numer / p_denom)
    else:
        tmp_prec = 0.0

    if r_denom > 0:
        tmp_recall = float(r_numer / r_denom)
    else:
        tmp_recall = 0.0

    tmp_f1 = f1(tmp_prec, tmp_recall)

    return tmp_f1, tmp_prec, tmp_recall


def get_metric(counts):

        for k in counts:
            if k[1] =='Tags':
                if k[0] in ('MWE'):
                    for subscore in ('Link+', 'Link-'):
                        c = counts[k][subscore]

                        c['F'], c['P'], c['R'] = calc_prec_recall(
                            c['PNumer'], c['PDenom'],
                            c['RNumer'], c['RDenom']
                        )

                        # VMWE
                        c['F-VB'], c['P-VB'], c['R-VB'] = calc_prec_recall(
                            c['PNumerVB'], c['PDenomVB'],
                            c['RNumerVB'], c['RDenomVB']
                        )

                        # Continuous MWE
                        c['F-CONT'], c['P-CONT'], c['R-CONT'] = calc_prec_recall(
                            c['PNumerCONT'], c['PDenomCONT'],
                            c['RNumerCONT'], c['RDenomCONT']
                        )

                    for m in ('P', 'R', 'F', 'P-VB', 'R-VB', 'F-VB', 'P-CONT', 'R-CONT', 'F-CONT'):
                        # strength averaging
                        avg = (counts[k]['Link+'][m]+counts[k]['Link-'][m])/2   # float
                        # construct a ratio by averaging the denominators (this gives insight into underlying recall-denominators)
                        counts[k]['LinkAvg'][m] = avg


        all_metrics = {}
        all_metrics["precision-overall"] = counts['MWE','Tags']['LinkAvg']['P']
        all_metrics["recall-overall"] = counts['MWE','Tags']['LinkAvg']['R']
        all_metrics["f1-measure-overall"] = counts['MWE','Tags']['LinkAvg']['F']

        all_metrics["precision-overall-VB"] = counts['MWE','Tags']['LinkAvg']['P-VB']
        all_metrics["recall-overall-VB"] = counts['MWE','Tags']['LinkAvg']['R-VB']
        all_metrics["f1-measure-overall-VB"] = counts['MWE','Tags']['LinkAvg']['F-VB']

        all_metrics["precision-overall-CONT"] = counts['MWE','Tags']['LinkAvg']['P-CONT']
        all_metrics["recall-overall-CONT"] = counts['MWE','Tags']['LinkAvg']['R-CONT']
        all_metrics["f1-measure-overall-CONT"] = counts['MWE','Tags']['LinkAvg']['F-CONT']

        print("----- prec ------")
        print(all_metrics["precision-overall"])
        print("----- recall ------")
        print(all_metrics["recall-overall"])
        print("----- f1 ------")
        print(all_metrics["f1-measure-overall"])

        print("----- prec (VMWE) ------")
        print(all_metrics["precision-overall-VB"])
        print("----- recall (VMWE) ------")
        print(all_metrics["recall-overall-VB"])
        print("----- f1 (VMWE) ------")
        print(all_metrics["f1-measure-overall-VB"])

        print("----- prec (Cont) ------")
        print(all_metrics["precision-overall-CONT"])
        print("----- recall (Cont) ------")
        print(all_metrics["recall-overall-CONT"])
        print("----- f1 (Continuous MWE) ------")
        print(all_metrics["f1-measure-overall-CONT"])


if __name__ == "__main__":
    ptns=[]
    ptns.append(["O","B-DT","I_-DT","O"])
    ptns.append(["O","B-VB","I_-VB","O"])
    ptns.append(["O","B-VB","o","I_-VB","O"])
    ptns.append(["O","B-VB","o","I_-VB","I_-VB","o","I_-VB","O"])
    ptns.append(["O","B-VB","o","I_-VB","B-DT","I_-DT","O"])
    ptns.append(["O","B-VB","o","b-DT","i_-DT","o", "I_-VB","o", "I_-VB","O"])

    pred_ptns=[]
    pred_ptns.append(["O","B-DT","I_-DT","O"])
    pred_ptns.append(["O","B-VB","I_-VB","O"])
    pred_ptns.append(["O","B-VB","o","I_-VB","O"])
    pred_ptns.append(["O","B-VB","o","I_-VB","I_-VB","o","I_-VB","O"])
    pred_ptns.append(["O","B-VB","o","I_-VB","O","O","O"])
    pred_ptns.append(["O","B-VB","o","b-DT","i_-DT","o", "I_-VB","o", "I_-VB","O"])

    cnt=0
    for gold_tags, pred_tags in zip(ptns, pred_ptns):
        print("---------")
        print("cnt:%d"%(cnt))
        #if cnt!=4:
        #    cnt+=1
        #    continue
        print(gold_tags)
        counts=defaultdict(lambda: defaultdict(Counter))
        eval_sent_links(gold_tags, pred_tags, counts)
        get_metric(counts)
        cnt+=1
        continue
        for k,v in counts.items():
            print("++++")
            print(k)
            for in_k, in_v in v.items():
                print("---")
                print(in_k)
                print(in_v)



