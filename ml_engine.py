"""
Machine Learning Engine for DNA Sequence Classification.
Provides training, evaluation, feature importance analysis, decision tree visualization,
and unknown sequence prediction with biological interpretations.
"""

from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for servers and scripts
import matplotlib.pyplot as plt
import io
import base64

FEATURE_COLUMNS = [
    "GC_Content_Pct",
    "Count_A",
    "Count_T",
    "Count_G",
    "Count_C",
    "Length"
]

FEATURE_DISPLAY_NAMES = {
    "GC_Content_Pct": "GC Content (%)",
    "Count_A": "Nucleotide A Count",
    "Count_T": "Nucleotide T Count",
    "Count_G": "Nucleotide G Count",
    "Count_C": "Nucleotide C Count",
    "Length": "Total Sequence Length (bp)"
}


class GenomeClassifierModel:
    def __init__(self, model_type: str = "Decision Tree", max_depth: int = 3, n_estimators: int = 50, random_state: int = 42):
        self.model_type = model_type
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        self.classes_ = None
        self.feature_names = FEATURE_COLUMNS
        self.is_trained = False
        self.training_summary = {}

    def _init_estimator(self):
        if self.model_type == "Decision Tree":
            return DecisionTreeClassifier(
                max_depth=self.max_depth,
                criterion="gini",
                random_state=self.random_state
            )
        elif self.model_type == "Random Forest":
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
        else:
            return DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=self.random_state
            )

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Trains the classifier on the provided DataFrame.
        DataFrame must contain the 6 feature columns and 'Group'.
        """
        # Ensure we have both groups represented
        unique_groups = df["Group"].unique()
        if len(unique_groups) < 2:
            raise ValueError(
                f"Training requires at least two distinct groups (e.g. Group A and Group B). "
                f"Currently found: {list(unique_groups)}"
            )

        X = df[self.feature_names].values
        y = df["Group"].values

        self.model = self._init_estimator()
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.is_trained = True

        # In-sample predictions
        y_pred = self.model.predict(X)
        train_acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred, labels=self.classes_)
        report_dict = classification_report(y, y_pred, labels=self.classes_, output_dict=True, zero_division=0)
        report_str = classification_report(y, y_pred, labels=self.classes_, zero_division=0)

        # Feature importances
        importances = self.model.feature_importances_
        feat_imp_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Display_Name": [FEATURE_DISPLAY_NAMES[f] for f in self.feature_names],
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

        # Decision rules text (for decision tree)
        tree_rules_text = ""
        if self.model_type == "Decision Tree":
            tree_rules_text = export_text(self.model, feature_names=self.feature_names)

        # Group-level feature statistics (baseline for interpretability)
        group_stats = df.groupby("Group")[self.feature_names].mean().to_dict(orient="index")

        self.training_summary = {
            "model_type": self.model_type,
            "sample_count": len(df),
            "classes": list(self.classes_),
            "train_accuracy": round(train_acc * 100.0, 2),
            "confusion_matrix": cm,
            "classification_report_dict": report_dict,
            "classification_report_str": report_str,
            "feature_importance_df": feat_imp_df,
            "tree_rules_text": tree_rules_text,
            "group_stats": group_stats
        }

        return self.training_summary

    def predict_unknown(self, feature_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts the group for an unknown DNA sequence from its extracted features.
        Returns predicted label, confidence/probabilities, and biological reasoning.
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet. Train the model first.")

        input_vector = np.array([[feature_dict[col] for col in self.feature_names]])
        prediction = self.model.predict(input_vector)[0]

        probabilities = {}
        confidence = 100.0
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(input_vector)[0]
            probabilities = {cls: round(prob * 100.0, 2) for cls, prob in zip(self.classes_, probs)}
            confidence = probabilities.get(prediction, 100.0)

        # Build qualitative reasoning based on dominant features
        reasoning = self._generate_prediction_reasoning(feature_dict, prediction)

        return {
            "predicted_group": prediction,
            "confidence_pct": confidence,
            "probabilities": probabilities,
            "input_features": feature_dict,
            "reasoning": reasoning
        }

    def _generate_prediction_reasoning(self, feature_dict: Dict[str, Any], predicted_group: str) -> List[str]:
        """
        Generates readable, biologically grounded explanations of why the model reached its decision.
        """
        reasons = []
        gc_val = feature_dict.get("GC_Content_Pct", 0.0)
        g_val = feature_dict.get("Count_G", 0)
        c_val = feature_dict.get("Count_C", 0)
        a_val = feature_dict.get("Count_A", 0)
        t_val = feature_dict.get("Count_T", 0)

        # Top feature from training
        top_feat = None
        if "feature_importance_df" in self.training_summary:
            top_feat = self.training_summary["feature_importance_df"].iloc[0]["Feature"]

        reasons.append(
            f"The sequence possesses a GC content of **{gc_val:.1f}%** (with G: {g_val}, C: {c_val}, A: {a_val}, T: {t_val})."
        )

        if gc_val >= 60.0:
            reasons.append(
                "High GC content (>= 60%) is characteristic of Group B sequences (frequently observed in bacterial genomes, CpG islands, and thermophile adaptations)."
            )
        elif gc_val <= 45.0:
            reasons.append(
                "Moderate to low GC content (<= 45%) with balanced or AT-dominant nucleotide counts closely mirrors the Group A baseline profile."
            )
        else:
            reasons.append(
                f"The sequence has an intermediate GC content of {gc_val:.1f}%. The decision boundary separated this sample into **{predicted_group}** based on individual base counts."
            )

        if top_feat:
            top_display = FEATURE_DISPLAY_NAMES.get(top_feat, top_feat)
            top_val = feature_dict.get(top_feat, "N/A")
            reasons.append(
                f"The model's highest weighted decision feature is **{top_display}** (value: {top_val})."
            )

        return reasons

    def render_tree_diagram(self) -> Optional[plt.Figure]:
        """
        Renders a matplotlib figure showing the Decision Tree structure.
        """
        if self.model_type != "Decision Tree" or not self.is_trained:
            return None

        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        plot_tree(
            self.model,
            feature_names=self.feature_names,
            class_names=[str(c) for c in self.classes_],
            filled=True,
            rounded=True,
            fontsize=9,
            ax=ax
        )
        ax.set_title("Decision Tree Boundary Architecture", fontsize=12, fontweight="bold", pad=12)
        plt.tight_layout()
        return fig
