from typing import List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy import stats
from anndata import AnnData


def logr_ecidx(
    X1: AnnData,
    X2: AnnData,
    ecidx: List[int]) -> float:
    if len(ecidx) == 0:
        return -1
    else:
        X1_X = X1.X
        X2_X = X2.X

        N1 = X1_X.shape[0]
        N2 = X2_X.shape[0]
        logr_labels = np.concatenate((np.ones(N1), np.zeros(N2)), axis=0)
        logr = LogisticRegression(C=1000)
        p_of_1 = N1 / float(N1 + N2)
        llnull = (N1 + N2) * (p_of_1 * np.log(p_of_1) + (1 - p_of_1) * np.log(1 - p_of_1))
        k = len(ecidx)

        X1_ecidx = X1_X[:, ecidx].todense()
        X2_ecidx = X2_X[:, ecidx].todense()
        c = np.concatenate([X1_ecidx, X2_ecidx])

        logr.fit(c, logr_labels)
        pred = np.array(logr.predict_proba(c)[:, 1])
        gene_score = log_loss(logr_labels, pred)
        llf = -gene_score * (N1 + N2)
        llr = llf - llnull
        llr_pval = stats.chi2.sf(2 * llr, k)  # survival function defined as 1-cdf

        return llr_pval
