#!/usr/bin/env python3
"""Reads wespeaker/bin/score.py output and prints EER, minDCF, and cosine threshold at EER."""

from __future__ import annotations

import os
import sys

import fire
import numpy as np

from wespeaker.utils.score_metrics import (
    compute_c_norm,
    compute_eer,
    compute_pmiss_pfa_rbst,
)


def report_metrics(scores_file, p_target=0.01, c_miss=1, c_fa=1):
    scores = []
    labels = []

    with open(scores_file, encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 4:
                continue
            scores.append(float(tokens[2]))
            labels.append(tokens[3] == "target")

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)

    fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
    eer, thres = compute_eer(fnr, fpr, scores)
    min_dcf = compute_c_norm(
        fnr, fpr, p_target=p_target, c_miss=c_miss, c_fa=c_fa
    )

    base = os.path.basename(scores_file)
    print("---- {} -----".format(base))
    print("EER = {:.3f}%".format(100 * eer))
    print("Cosine similarity threshold at EER = {:.6f}".format(thres))
    print(
        "  (decision rule aligned with wespeaker/bin/score.py: same-speaker if "
        "sklearn cosine_similarity >= threshold)"
    )
    print(
        "minDCF (p_target:{} c_miss:{} c_fa:{}) = {:.3f}".format(
            p_target, c_miss, c_fa, min_dcf
        )
    )


def main(p_target=0.01, c_miss=1, c_fa=1, *scores_files):
    if not scores_files:
        print("Usage: report_metrics.py [--p_target=0.01] <trial.score> ...", file=sys.stderr)
        sys.exit(1)
    for sf in scores_files:
        report_metrics(sf, p_target, c_miss, c_fa)


if __name__ == "__main__":
    fire.Fire(main)
