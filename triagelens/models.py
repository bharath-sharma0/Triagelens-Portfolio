"""Validated loading of trusted local notebook exports."""
import json
import warnings
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.exceptions import InconsistentVersionWarning
from .config import FEATURES, missing_artifacts
from .notebook_inference import TriageLensPredictor

class ArtifactError(ValueError):
    pass

def load_predictor(root):
    root=Path(root)
    missing=missing_artifacts(root)
    if missing:
        raise ArtifactError('Missing artifacts: '+', '.join(missing))
    try:
        config=json.loads((root/'triagelens_results/experiment_config.json').read_text())
        for key in ('confidence_threshold','trust_threshold'):
            value=config[key]
            if isinstance(value,bool) or not isinstance(value,(float,int)) or not np.isfinite(value) or not 0<=value<=1:
                raise ArtifactError(f'Invalid {key}. Export full-precision frozen thresholds.')
        with warnings.catch_warnings():
            warnings.simplefilter('error',InconsistentVersionWarning)
            p=TriageLensPredictor(root)
        if list(p.feature_columns)!=FEATURES:
            raise ArtifactError('Trust feature order differs from notebook.')
        if not sparse.issparse(p.training_tfidf):
            raise ArtifactError('Expected sparse training TF-IDF matrix.')
        if p.training_tfidf.shape[0]!=15000 or np.asarray(p.training_lsa).shape!=(15000,300):
            raise ArtifactError('Reference data must have 15,000 rows and 300 LSA dimensions.')
        if np.asarray(p.training_labels).shape!=(15000,):
            raise ArtifactError('Training labels and reference rows do not align.')
        width=len(p.tfidf.get_feature_names_out())
        if p.training_tfidf.shape[1]!=width or p.primary.n_features_in_!=width or p.lsa.n_features_in_!=width:
            raise ArtifactError('TF-IDF dimensions differ across artifacts.')
        if p.lsa.components_.shape!=(300,width) or p.lsa_classifier.n_features_in_!=300 or p.trust_model.n_features_in_!=7:
            raise ArtifactError('LSA or trust input dimensions differ.')
        if len(p.primary.classes_)!=150 or set(p.training_labels)!=set(p.primary.classes_):
            raise ArtifactError('Expected matching 150-class intent labels.')
        if not np.array_equal(p.trust_model.classes_,[0,1]):
            raise ArtifactError('Trust positive class must mean primary prediction is correct.')
        if hasattr(p.trust_model,'feature_names_in_') and list(p.trust_model.feature_names_in_)!=FEATURES:
            raise ArtifactError('Trust model feature names differ.')
        for values in (p.training_tfidf.data,p.training_lsa,p.lsa.components_,p.primary.coef_,p.lsa_classifier.coef_):
            if not np.isfinite(values).all():
                raise ArtifactError('Artifacts contain non-finite values.')
        if not np.allclose(np.asarray(p.training_tfidf.multiply(p.training_tfidf).sum(axis=1)).ravel(),1,atol=1e-5):
            raise ArtifactError('Training TF-IDF references are not unit normalized.')
        if not np.allclose(np.linalg.norm(p.training_lsa,axis=1),1,atol=1e-5):
            raise ArtifactError('Training LSA references are not unit normalized.')
        return p
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError('Unable to load compatible notebook artifacts. Check versions, config and export integrity.') from exc
