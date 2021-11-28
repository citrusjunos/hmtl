# coding: utf-8

from collections import defaultdict
import sys,os

def regist(snt_infos, snt_lines, header_d):
    snt_infos.append(
        {key:[tmp_cols[col_idx] for tmp_cols in snt_lines] for key, col_idx in header_d.items()})

def parse_conll(tagged_snts_path, header_d = {}, min_cols = 1):
    snt_infos = []
    snt_lines = []
    f_in = open(tagged_snts_path)
    for i,line in enumerate(f_in.readlines()):
        cols = line.strip().split("\t")
        if len(cols)==0 or (len(cols)==1 and len(cols[0])==0):
            regist(snt_infos, snt_lines, header_d)
            snt_lines=[]
        elif "." in cols[0]:
            # empty nodes (https://universaldependencies.org/format.html)
            # e.g., 24.1
            continue
        elif len(cols) >= min_cols:
            snt_lines.append(cols)
        else:
            print("err!!")
            print(len(cols))
            print(cols)
            print(" ".join(cols))
            exit(1)
    if len(snt_lines)>0:
        regist(snt_infos, snt_lines, header_d)

    return snt_infos


