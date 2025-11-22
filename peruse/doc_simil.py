from collections.abc import Mapping, Callable
from functools import partial
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.base import BaseEstimator
from scipy.spatial.distance import cdist

from slang import dflt_snips_to_str as snips_to_str


class Utils:
    array_mean = staticmethod(partial(np.mean, axis=0))
    array_min = staticmethod(partial(np.min, axis=0))


_u = Utils()


@dataclass
class DocSimilScorer:
    """
    >>> dss = DocSimilScorer()
    >>> snips_to_learn_with = [[2, 0, 1, 0], [2, 0, 2, 0]]
    >>> _ = dss.fit_vectorizer(snips_to_learn_with)
    >>>
    >>> vect = dss.docs_to_vect([[2, 0, 1, 0]]).toarray()
    >>> assert np.isclose(vect, np.array([[0.75726441, 0.53215436, 0.37863221]])).all()
    >>>
    >>> simil_scores = dss.simil_scores([[2, 0, 2, 0]],
    ...                                 [[2, 0, 2, 0], [2, 0, 1, 0], [2, 2, 2, 2]])
    >>> assert np.isclose(simil_scores,
    ...                   np.array([0., 0.1967998, 0.29289322])).all()
    """
    vectorizer: TfidfVectorizer = TfidfVectorizer(token_pattern=None, tokenizer=list, preprocessor=snips_to_str)
    u = _u

    def fit_vectorizer(self, docs):
        self.vectorizer.fit(docs)
        return self

    def docs_to_vect(self, docs):
        return self.vectorizer.transform(docs)

    def simil_scores(self, query_docs, peruse_docs, dist_aggreg=u.array_min):
        return self.peruser(query_docs, dist_aggreg)(peruse_docs)

    def peruser(self, query_docs, dist_aggreg=u.array_min):
        query_vects = self.docs_to_vect(query_docs).toarray()

        def peruse_docs(peruse_docs):
            peruse_vects = self.docs_to_vect(peruse_docs).toarray()
            doc_sims = cdist(query_vects, peruse_vects, metric='cosine')
            return dist_aggreg(doc_sims)

        return peruse_docs


from slang import FittableSnipper, Snipper


# SnipDocSimilScorer = DocSimilScorer

@dataclass
class SnipDocSimilScorer(DocSimilScorer):
    """
    """
    snipper: Snipper = FittableSnipper()
    vectorizer: TfidfVectorizer = TfidfVectorizer(token_pattern=None, tokenizer=list, preprocessor=snips_to_str)

    def wfs_to_docs(self, wfs):
        return map(list, map(self.snipper.wf_to_snips, wfs))

    def fit_vectorizer(self, wfs):
        snips_collection = self.wfs_to_docs(wfs)
        super().fit_vectorizer(snips_collection)
        return self

    def fit_snipper(self, wfs):
        self.snipper.fit(wfs)
        return self

    def fit(self, wfs):
        self.fit_snipper(wfs)
        self.fit_vectorizer(wfs)
        return self

    def docs_to_vect(self, wfs):
        snips_collection = self.wfs_to_docs(wfs)
        return super().docs_to_vect(snips_collection)

    # def simil_scores(self, query_wfs, peruse_wfs, dist_aggreg=_u.array_min):
    #     query_docs = self.wfs_to_docs(query_wfs)
    #     peruse_docs = self.wfs_to_docs(peruse_wfs)
    #     return super().simil_scores(query_docs, peruse_docs, dist_aggreg)


from py2store import KvReader
from dataclasses import dataclass
from collections.abc import Sequence
from slang import dflt_snips_to_str


@dataclass
class SnipSeqStore(KvReader):
    snips: Sequence
    doc_size: int = 20

    def __iter__(self):
        yield from range(len(self.snips) - self.doc_size)

    def __getitem__(self, k):
        #         return self.snips[k:(k + self.doc_size)]
        #         return ' '.join(dflt_snips_to_str(self.snips[k:(k + self.doc_size)]))
        return ''.join(dflt_snips_to_str(self.snips[k:(k + self.doc_size)]))

    def __len__(self):
        return len(self.snips)
