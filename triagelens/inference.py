"""Run original notebook inference and expose actual classifier distributions."""
import time
import numpy as np
from .preprocessing import transform

def predict(predictor, query):
    lexical, raw, latent=transform(predictor,query)
    start=time.perf_counter()
    result=predictor.predict(query)
    result['elapsed_ms']=(time.perf_counter()-start)*1000
    primary=predictor.primary.predict_proba(lexical)[0]
    secondary=predictor.lsa_classifier.predict_proba(latent)[0]
    for values in (primary,secondary,[result['trust_score']],list(result['signals'].values())):
        if not np.isfinite(values).all():
            raise ValueError('Model returned non-finite values.')
    for values in (primary,secondary):
        if np.any(values<0) or np.any(values>1) or not np.isclose(sum(values),1):
            raise ValueError('Invalid model probability distribution.')
    if not 0<=result['trust_score']<=1:
        raise ValueError('Invalid trust probability.')
    result.update(classes=predictor.primary.classes_.tolist(),primary_probabilities=primary.tolist(),
                  secondary_probabilities=secondary.tolist(),secondary_intent=str(predictor.lsa_classifier.classes_[np.argmax(secondary)]),
                  zero_vocabulary=lexical.nnz==0)
    return result
