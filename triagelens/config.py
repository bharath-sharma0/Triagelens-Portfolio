"""Paths and research invariants; no rounded policy thresholds are used for inference."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path(os.environ.get('TRIAGELENS_PROJECT_DIR', ROOT)).expanduser().resolve()
FEATURES = ['confidence', 'tfidf_nn_similarity', 'knn_agreement', 'lsa_nn_similarity',
            'lsa_knn_agreement', 'model_agreement_score', 'classifier_lsa_knn_match']
MODEL_FILES = ['tfidf_vectorizer', 'primary_classifier', 'lsa_transformer', 'lsa_classifier',
               'trust_model', 'training_tfidf', 'training_lsa', 'training_labels', 'feature_columns']
SAMPLES = ['How can I change the PIN on my debit card?',
           'I need to report my stolen credit card', 'Please cancel my flight reservation']
SIGNAL_HELP = dict(zip(FEATURES, [
    'Maximum primary-classifier probability; not a guarantee of correctness.',
    'Largest cosine similarity to a training TF-IDF vector.',
    'Fraction of nonnegative similarity weight voting for the winning intent among five lexical neighbors.',
    'Largest cosine similarity to a normalized training LSA vector.',
    'Fraction of nonnegative similarity weight voting for the winning intent among five LSA neighbors.',
    'Negative Jensen–Shannon distance, using natural logarithms. Closer to zero means closer class distributions.',
    'One when primary intent matches the weighted LSA neighbor winner, zero otherwise.'
]))

def missing_artifacts(root=ARTIFACT_ROOT):
    paths = [Path(root)/'triagelens_models'/f'{f}.joblib' for f in MODEL_FILES]
    paths += [Path(root)/'triagelens_results/experiment_config.json']
    return [str(p.relative_to(root)) for p in paths if not p.is_file()]
