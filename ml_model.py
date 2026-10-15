"""
ml_model.py – ML-based strategy selector using RandomForestClassifier.

Trains on synthetic swarm data to predict the optimal strategy:
  0 → TIT_FOR_TAT
  1 → STRICT_MODE
  2 → EXPLORATION_MODE

Falls back to rule-based selection if the model is not available.
"""

import os
import pickle
import logging
from typing import Optional, List, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'strategy_model.pkl'
)

# Strategy constants
STRATEGY_TIT_FOR_TAT = 0
STRATEGY_STRICT_MODE = 1
STRATEGY_EXPLORATION_MODE = 2

# Feature indices
FEAT_AVG_RATIO = 0
FEAT_LOW_PCT = 1
FEAT_CHURN = 2
FEAT_TIMEOUT = 3
FEAT_FAILURE = 4


def generate_training_data(n: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data for the strategy selector.

    Features: [avg_ratio, low_contributor_pct, churn_rate, timeout_rate, failure_rate]
    Labels:   0=TIT_FOR_TAT, 1=STRICT_MODE, 2=EXPLORATION_MODE

    Heuristic labeling rules:
      - High freerider % (>40%) OR high failure rate (>30%) → STRICT_MODE
      - Low avg ratio (<0.4) AND low freerider % (<20%) → EXPLORATION_MODE
      - High churn (>5.0) AND moderate freerider % → EXPLORATION_MODE
      - Otherwise → TIT_FOR_TAT (default cooperative strategy)

    Args:
        n: Number of synthetic samples.

    Returns:
        (X, y) tuple of numpy arrays.
    """
    rng = np.random.default_rng(seed=42)

    # NOTE: pickles are only safe from trusted sources; the model file is
    # generated locally by train_model() and is deliberately not committed.
    X = np.zeros((n, 5))
    y = np.zeros(n, dtype=int)

    for i in range(n):
        avg_ratio = rng.uniform(0.0, 1.5)
        low_pct = rng.uniform(0.0, 1.0)
        churn = rng.uniform(0.0, 10.0)
        timeout = rng.uniform(0.0, 0.5)
        failure = rng.uniform(0.0, 0.5)

        X[i] = [avg_ratio, low_pct, churn, timeout, failure]

        # Label based on heuristic rules
        if low_pct > 0.4 or failure > 0.3:
            # Many freeriders or high failure rate → strict enforcement
            y[i] = STRATEGY_STRICT_MODE
        elif avg_ratio < 0.4 and low_pct < 0.2:
            # Low overall contribution, few identified freeriders → explore
            y[i] = STRATEGY_EXPLORATION_MODE
        elif churn > 5.0 and low_pct < 0.3:
            # High churn with few freeriders → explore new peers
            y[i] = STRATEGY_EXPLORATION_MODE
        else:
            # Normal conditions → cooperative tit-for-tat
            y[i] = STRATEGY_TIT_FOR_TAT

    return X, y


def train_model(save_path: Optional[str] = None) -> Tuple[Any, float]:
    """
    Train a RandomForestClassifier on synthetic data and save it.

    Args:
        save_path: Path to save model pickle (default: strategy_model.pkl).

    Returns:
        Tuple of (trained_model, accuracy). Model is None if sklearn unavailable.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
    except ImportError:
        logger.warning("scikit-learn not installed. Using rule-based fallback.")
        return None, 0.0

    if save_path is None:
        save_path = MODEL_PATH

    X, y = generate_training_data(n=500)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    logger.info(f"Model trained with accuracy: {accuracy:.2%}")

    tmp_path = save_path + '.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(model, f)
    os.replace(tmp_path, save_path)  # atomic: never leave a half-written model
    logger.info(f"Model saved to {save_path}")

    return model, accuracy


def load_model(model_path: Optional[str] = None) -> Any:
    """
    Load a pre-trained model from disk.

    Args:
        model_path: Path to model pickle file.

    Returns:
        Loaded model, or None if not found.
    """
    if model_path is None:
        model_path = MODEL_PATH

    if not os.path.exists(model_path):
        logger.warning(f"Model not found at {model_path}")
        return None

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:  # corrupt/incompatible pickle -> caller retrains
        logger.error(f"Failed to load model: {e}")
        return None


def predict_strategy(model: Any, feature_vector: List[float]) -> int:
    """
    Predict the optimal strategy using the ML model.

    Args:
        model: Trained sklearn model (or None for fallback).
        feature_vector: [avg_ratio, low_pct, churn, timeout, failure]

    Returns:
        Strategy int (0=TIT_FOR_TAT, 1=STRICT_MODE, 2=EXPLORATION_MODE)
    """
    if model is None:
        return rule_based_fallback(feature_vector)

    try:
        X = np.array(feature_vector).reshape(1, -1)
        prediction = int(model.predict(X)[0])
        return prediction
    except Exception as e:
        logger.error(f"ML prediction failed: {e}, using fallback")
        return rule_based_fallback(feature_vector)


def predict_with_confidence(model: Any, feature_vector: List[float]) -> Tuple[int, List[float]]:
    """
    Predict a strategy plus per-class probabilities (for explainability and
    for the confidence gate in the adaptive loop).

    Falls back to the rule-based choice with a one-hot distribution.
    """
    one_hot = lambda c: [1.0 if i == c else 0.0 for i in range(3)]
    if model is None or not hasattr(model, 'predict_proba'):
        choice = rule_based_fallback(feature_vector)
        return choice, one_hot(choice)
    try:
        X = np.array(feature_vector, dtype=float).reshape(1, -1)
        proba = model.predict_proba(X)[0]
        full = [0.0, 0.0, 0.0]
        for cls, p in zip(model.classes_, proba):
            full[int(cls)] = float(p)
        return int(np.argmax(full)), full
    except Exception as e:
        logger.error(f"ML probability prediction failed: {e}, using fallback")
        choice = rule_based_fallback(feature_vector)
        return choice, one_hot(choice)


def rule_based_fallback(feature_vector: List[float]) -> int:
    """
    Rule-based strategy selection when ML model is unavailable.

    Uses the same heuristics as the training data generation:
      - High freerider % or failure rate → STRICT_MODE
      - Low avg ratio with few identified freeriders → EXPLORATION_MODE
      - High churn with few freeriders → EXPLORATION_MODE
      - Otherwise → TIT_FOR_TAT

    Args:
        feature_vector: [avg_ratio, low_pct, churn, timeout, failure]

    Returns:
        Strategy int (0, 1, or 2).
    """
    avg_ratio = feature_vector[FEAT_AVG_RATIO]
    low_pct = feature_vector[FEAT_LOW_PCT]
    churn = feature_vector[FEAT_CHURN]
    _timeout = feature_vector[FEAT_TIMEOUT]
    failure = feature_vector[FEAT_FAILURE]

    if low_pct > 0.4 or failure > 0.3:
        logger.info("  Fallback → STRICT_MODE (high freerider%/failure)")
        return STRATEGY_STRICT_MODE
    elif avg_ratio < 0.4 and low_pct < 0.2:
        logger.info("  Fallback → EXPLORATION_MODE (low ratio, few freeriders)")
        return STRATEGY_EXPLORATION_MODE
    elif churn > 5.0 and low_pct < 0.3:
        logger.info("  Fallback → EXPLORATION_MODE (high churn)")
        return STRATEGY_EXPLORATION_MODE
    else:
        logger.info("  Fallback → TIT_FOR_TAT (normal conditions)")
        return STRATEGY_TIT_FOR_TAT


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("=== Training ML Strategy Selector ===")
    model, accuracy = train_model()

    if model is not None:
        print(f"\nAccuracy: {accuracy:.2%}")
        print("\n=== Test Predictions ===")

        strategy_names = {
            0: 'TIT_FOR_TAT',
            1: 'STRICT_MODE',
            2: 'EXPLORATION_MODE'
        }

        test_cases = [
            ([0.8, 0.1, 1.0, 0.05, 0.02], "Normal swarm → TIT_FOR_TAT"),
            ([0.3, 0.6, 2.0, 0.1, 0.1],  "Many freeriders → STRICT_MODE"),
            ([0.2, 0.1, 0.5, 0.0, 0.0],  "Low contribution → EXPLORATION"),
            ([0.5, 0.5, 5.0, 0.3, 0.4],  "High failure → STRICT_MODE"),
            ([0.6, 0.2, 7.0, 0.1, 0.1],  "High churn → EXPLORATION"),
        ]

        for features, desc in test_cases:
            pred = predict_strategy(model, features)
            print(f"  {desc}: {strategy_names[pred]}")
            print(f"    features={features}")
    else:
        print("\nNo sklearn available, testing fallback:")
        pred = rule_based_fallback([0.3, 0.6, 2.0, 0.1, 0.1])
        print(f"  Fallback prediction: {pred}")
