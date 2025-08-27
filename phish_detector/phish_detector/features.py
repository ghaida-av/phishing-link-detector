from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from typing import Optional, Dict

def get_feature_pipeline(max_tfidf_features: int = 5000,
                         min_df: int = 2,
                         tld_freq: Optional[Dict[str, int]] = None) -> Pipeline:
    """
    Returns a sklearn pipeline for extracting features from URLs.
    """
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=max_tfidf_features,
        min_df=min_df
    )

    pipeline = Pipeline([
        ("vectorizer", vectorizer)
    ])

    return pipeline

