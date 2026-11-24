"""Only transforms fitted on original training data; never fit at inference."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from .data import validate_query

def transform(predictor, query):
    validate_query(query)
    lexical=predictor.tfidf.transform([query])
    latent_raw=predictor.lsa.transform(lexical)
    latent=normalize(latent_raw)
    if not np.isfinite(lexical.data).all() or not np.isfinite(latent).all():
        raise ValueError('Non-finite transformed features.')
    return lexical, latent_raw, latent

def feature_table(predictor, lexical):
    names=predictor.tfidf.get_feature_names_out()
    return pd.DataFrame({'Term / n-gram':names[lexical.indices], 'TF-IDF weight':lexical.data}).sort_values('TF-IDF weight',ascending=False)
