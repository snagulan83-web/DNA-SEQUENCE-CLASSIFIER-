"""
Unit and integration tests for DNA Engine and ML Engine.
"""
from dna_engine import validate_sequence, extract_dna_features, get_default_sequences, build_feature_dataframe
from ml_engine import GenomeClassifierModel

def test_engines():
    print("Testing DNA engine...")
    is_valid, cleaned, msg = validate_sequence("ATGCGATC")
    assert is_valid, f"Validation failed: {msg}"
    assert cleaned == "ATGCGATC"

    # Test features
    feats = extract_dna_features("ATGCGATC")
    print("Extracted features for ATGCGATC:", feats)
    assert feats["Count_A"] == 2
    assert feats["Count_T"] == 2
    assert feats["Count_G"] == 2
    assert feats["Count_C"] == 2
    assert feats["Length"] == 8
    assert feats["GC_Content_Pct"] == 50.0

    # Test unknown sequence from prompt: ATGCGCGATC
    unk_feats = extract_dna_features("ATGCGCGATC")
    print("Extracted features for ATGCGCGATC:", unk_feats)
    assert unk_feats["Count_A"] == 2
    assert unk_feats["Count_T"] == 2
    assert unk_feats["Count_G"] == 3
    assert unk_feats["Count_C"] == 3
    assert unk_feats["Length"] == 10
    assert unk_feats["GC_Content_Pct"] == 60.0

    # Build dataframe for assignment default dataset
    core_seqs = get_default_sequences("assignment_core")
    df = build_feature_dataframe(core_seqs)
    print("Feature dataframe:")
    print(df[["Seq_ID", "Sequence", "Group", "GC_Content_Pct", "Count_G", "Count_C"]])

    # Train model
    print("\nTraining Decision Tree classifier...")
    model = GenomeClassifierModel(model_type="Decision Tree", max_depth=3)
    summary = model.train(df)
    print(f"Training accuracy: {summary['train_accuracy']}%")
    print("Feature importances:\n", summary["feature_importance_df"])
    print("Decision tree rules:\n", summary["tree_rules_text"])

    # Predict unknown sequence
    pred = model.predict_unknown(unk_feats)
    print("\nPrediction for ATGCGCGATC:")
    print("Predicted Group:", pred["predicted_group"])
    print("Confidence:", pred["confidence_pct"])
    print("Probabilities:", pred["probabilities"])
    print("Reasoning:", pred["reasoning"])

    # Test Random Forest model
    print("\nTraining Random Forest classifier...")
    rf_model = GenomeClassifierModel(model_type="Random Forest", n_estimators=20)
    rf_summary = rf_model.train(df)
    rf_pred = rf_model.predict_unknown(unk_feats)
    print("RF Predicted Group:", rf_pred["predicted_group"])
    print("RF Probabilities:", rf_pred["probabilities"])

    print("\nALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_engines()
