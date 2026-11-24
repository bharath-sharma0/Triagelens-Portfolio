
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy import sparse
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import normalize


class TriageLensPredictor:
    """
    Reusable inference pipeline for TriageLens.

    Loads trained models and applies the frozen
    confidence-only and seven-signal trust policies.
    """

    def __init__(self, project_dir=None):

        if project_dir is None:
            self.project_dir = Path(__file__).resolve().parent
        else:
            self.project_dir = Path(project_dir).resolve()

        model_dir = self.project_dir / "triagelens_models"

        config_path = (
            self.project_dir
            / "triagelens_results"
            / "experiment_config.json"
        )

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.confidence_threshold = float(
            config["confidence_threshold"]
        )

        self.trust_threshold = float(
            config["trust_threshold"]
        )

        # Load all trained components
        self.tfidf = joblib.load(
            model_dir / "tfidf_vectorizer.joblib"
        )

        self.primary = joblib.load(
            model_dir / "primary_classifier.joblib"
        )

        self.lsa = joblib.load(
            model_dir / "lsa_transformer.joblib"
        )

        self.lsa_classifier = joblib.load(
            model_dir / "lsa_classifier.joblib"
        )

        self.trust_model = joblib.load(
            model_dir / "trust_model.joblib"
        )

        self.training_tfidf = joblib.load(
            model_dir / "training_tfidf.joblib"
        )

        self.training_lsa = joblib.load(
            model_dir / "training_lsa.joblib"
        )

        self.training_labels = np.asarray(
            joblib.load(
                model_dir / "training_labels.joblib"
            )
        )

        self.feature_columns = joblib.load(
            model_dir / "feature_columns.joblib"
        )

        # Ensure probability columns refer to the same intents
        if not np.array_equal(
            self.primary.classes_,
            self.lsa_classifier.classes_
        ):
            raise ValueError("Classifier class orders differ.")

        print("TriageLens models loaded successfully!")

    def _knn_signals(self, query_vector, training_vectors, k=5):

        similarities = query_vector @ training_vectors.T

        if sparse.issparse(similarities):
            similarities = similarities.toarray()

        similarities = np.asarray(similarities).ravel()

        top_indices = np.argpartition(
            -similarities,
            kth=k - 1
        )[:k]

        votes = {}

        for idx in top_indices:

            intent = self.training_labels[idx]
            weight = max(float(similarities[idx]), 0.0)

            votes[intent] = votes.get(intent, 0.0) + weight

        total_weight = sum(votes.values())

        if total_weight > 0:

            majority_intent = max(votes, key=votes.get)

            agreement = (
                votes[majority_intent] / total_weight
            )

        else:
            majority_intent = None
            agreement = 0.0

        return (
            float(similarities.max()),
            float(agreement),
            majority_intent
        )

    def predict(self, query):

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be non-empty text.")

        # Generate TF-IDF and LSA representations
        X_query = self.tfidf.transform([query])

        X_query_lsa = normalize(
            self.lsa.transform(X_query)
        )

        # Primary classifier
        probabilities = self.primary.predict_proba(
            X_query
        )[0]

        predicted_intent = self.primary.classes_[
            np.argmax(probabilities)
        ]

        confidence = float(probabilities.max())

        # Secondary classifier
        lsa_probabilities = (
            self.lsa_classifier.predict_proba(
                X_query_lsa
            )[0]
        )

        # TF-IDF neighbor features
        tfidf_nn, tfidf_knn, _ = self._knn_signals(
            X_query,
            self.training_tfidf
        )

        # LSA neighbor features
        lsa_nn, lsa_knn, lsa_majority = self._knn_signals(
            X_query_lsa,
            self.training_lsa
        )

        # Model agreement
        model_agreement = -float(
            jensenshannon(
                probabilities,
                lsa_probabilities
            )
        )

        # Classifier-neighbor match
        knn_match = int(
            predicted_intent == lsa_majority
        )

        # Assemble all seven features
        signals = {
            "confidence": confidence,
            "tfidf_nn_similarity": tfidf_nn,
            "knn_agreement": tfidf_knn,
            "lsa_nn_similarity": lsa_nn,
            "lsa_knn_agreement": lsa_knn,
            "model_agreement_score": model_agreement,
            "classifier_lsa_knn_match": knn_match
        }

        feature_df = pd.DataFrame(
            [signals],
            columns=self.feature_columns
        )

        # Learned trust score
        trust_score = float(
            self.trust_model.predict_proba(
                feature_df
            )[0, 1]
        )

        # Apply frozen policies
        confidence_decision = (
            "AUTOMATE"
            if confidence >= self.confidence_threshold
            else "HUMAN REVIEW"
        )

        triagelens_decision = (
            "AUTOMATE"
            if trust_score >= self.trust_threshold
            else "HUMAN REVIEW"
        )

        return {
            "query": query,
            "predicted_intent": str(predicted_intent),
            "confidence": confidence,
            "trust_score": trust_score,
            "confidence_decision": confidence_decision,
            "triagelens_decision": triagelens_decision,
            "signals": signals
        }
