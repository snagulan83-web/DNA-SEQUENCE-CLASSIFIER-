"""
Predict the Genome Pattern: AI/ML-based classification of DNA sequences
Streamlit Web Application for University-Level Computational Biology.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from dna_engine import (
    validate_sequence,
    extract_dna_features,
    get_default_sequences,
    build_feature_dataframe,
    VALID_NUCLEOTIDES
)
from ml_engine import GenomeClassifierModel, FEATURE_COLUMNS, FEATURE_DISPLAY_NAMES

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Genome Pattern Classifier | AI/ML DNA Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom Styling (Bioinformatics Aesthetic)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700;
        color: #1E3A8A;
    }
    
    /* DNA nucleotide badge styling */
    .base-badge {
        display: inline-block;
        padding: 2px 7px;
        font-family: monospace;
        font-weight: bold;
        border-radius: 4px;
        margin: 1px;
    }
    .base-A { background-color: #D1FAE5; color: #065F46; border: 1px solid #10B981; }
    .base-T { background-color: #FEE2E2; color: #991B1B; border: 1px solid #EF4444; }
    .base-G { background-color: #FEF3C7; color: #92400E; border: 1px solid #F59E0B; }
    .base-C { background-color: #DBEAFE; color: #1E40AF; border: 1px solid #3B82F6; }

    /* Custom Callout Box */
    .bio-card {
        background: #F8FAFC;
        border-left: 5px solid #2563EB;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .bio-alert {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 8px;
        padding: 12px 16px;
        color: #1E40AF;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def format_dna_html(seq: str) -> str:
    """Renders formatted HTML nucleotide badges for a DNA string."""
    html_parts = []
    for char in seq.upper():
        if char in "ATGC":
            html_parts.append(f'<span class="base-badge base-{char}">{char}</span>')
        else:
            html_parts.append(f'<span class="base-badge" style="background:#E2E8F0; color:#475569;">{char}</span>')
    return "".join(html_parts)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "dataset_records" not in st.session_state:
    st.session_state.dataset_records = get_default_sequences("assignment_core")

if "trained_model" not in st.session_state:
    st.session_state.trained_model = None

if "model_summary" not in st.session_state:
    st.session_state.model_summary = None

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


# -----------------------------------------------------------------------------
# Sidebar: Configuration & Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🧬 Control Hub")
    st.markdown("**Predict the Genome Pattern**  \n*AI/ML Sequence Classification*")
    st.divider()

    st.subheader("📁 Dataset Presets")
    preset_choice = st.selectbox(
        "Load Predefined Dataset",
        options=["assignment_core", "assignment_extended", "cpg_islands"],
        format_func=lambda x: {
            "assignment_core": "Assignment Default (4 seqs: S1-S4)",
            "assignment_extended": "Assignment Extended (12 sequences)",
            "cpg_islands": "CpG Islands vs Intergenic (10 seqs)"
        }[x],
        help="Select standard assignment sequences or an extended dataset."
    )

    if st.button("🔄 Reset / Apply Selected Preset", use_container_width=True):
        st.session_state.dataset_records = get_default_sequences(preset_choice)
        st.session_state.trained_model = None
        st.session_state.model_summary = None
        st.session_state.last_prediction = None
        st.success(f"Loaded dataset: {preset_choice}")
        st.rerun()

    st.divider()

    st.subheader("⚙️ Classifier Settings")
    model_choice = st.selectbox(
        "Machine Learning Algorithm",
        options=["Decision Tree", "Random Forest"],
        index=0,
        help="Decision Tree gives transparent rule splits. Random Forest is an ensemble of trees."
    )

    max_depth = st.slider("Max Tree Depth", min_value=1, max_value=6, value=3, help="Limits tree complexity to prevent severe overfitting.")
    
    n_estimators = 50
    if model_choice == "Random Forest":
        n_estimators = st.slider("Number of Trees", min_value=10, max_value=100, value=50, step=10)

    st.divider()
    st.markdown("### 📘 Assignment Specifications")
    st.info(
        """
        **6 Target Features:**
        1. Count(A)
        2. Count(T)
        3. Count(G)
        4. Count(C)
        5. Sequence Length
        6. GC Content (%)
        """
    )


# -----------------------------------------------------------------------------
# Header Banner
# -----------------------------------------------------------------------------
st.markdown("""
# 🧬 Predict the Genome Pattern
### AI/ML-based Classification of DNA Sequences
*Computational Biology & Machine Learning Laboratory Dashboard*
""")

st.markdown("""
<div class="bio-alert">
    <strong>Dashboard Overview:</strong> This application demonstrates automated genomic feature engineering 
    and machine learning classification to distinguish DNA patterns between <strong>Group A</strong> (balanced/AT-rich motifs) 
    and <strong>Group B</strong> (GC-rich motifs).
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Tabs Interface
# -----------------------------------------------------------------------------
tab_data, tab_features, tab_train, tab_predict, tab_analysis, tab_discussion = st.tabs([
    "1. 📋 Dataset & Input",
    "2. 🧮 Feature Extraction",
    "3. 🤖 Model Training",
    "4. 🔮 Predict Unknown Sequence",
    "5. 🔬 Applications & Limitations",
    "6. 💡 Discussion & Q&A"
])


# =============================================================================
# TAB 1: DATASET & INPUT SECTION
# =============================================================================
with tab_data:
    st.header("1. Dataset & Sequence Input Manager")
    st.markdown(
        "View existing sequences, edit entries directly, or add custom sequences for Group A and Group B."
    )

    col_data_view, col_add_seq = st.columns([1.5, 1])

    with col_data_view:
        st.subheader("Current Dataset Records")
        current_df = pd.DataFrame(st.session_state.dataset_records)
        
        # Interactive Editor
        edited_df = st.data_editor(
            current_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Seq_ID": st.column_config.TextColumn("Sequence ID", required=True),
                "Sequence": st.column_config.TextColumn("DNA Sequence (5' -> 3')", required=True),
                "Group": st.column_config.SelectboxColumn("Group Label", options=["Group A", "Group B"], required=True)
            },
            key="sequence_editor"
        )

        # Sync changes from data_editor
        if st.button("💾 Save Table Edits", type="secondary"):
            cleaned_records = []
            for _, row in edited_df.iterrows():
                if pd.notna(row["Seq_ID"]) and pd.notna(row["Sequence"]) and pd.notna(row["Group"]):
                    cleaned_records.append({
                        "Seq_ID": str(row["Seq_ID"]).strip(),
                        "Sequence": str(row["Sequence"]).strip().upper(),
                        "Group": str(row["Group"]).strip()
                    })
            st.session_state.dataset_records = cleaned_records
            st.session_state.trained_model = None
            st.success("Dataset successfully updated!")
            st.rerun()

    with col_add_seq:
        st.subheader("Add Single Sequence")
        with st.form("add_sequence_form", clear_on_submit=True):
            new_id = st.text_input("Sequence Identifier", value=f"S{len(st.session_state.dataset_records) + 1}")
            new_seq = st.text_input("DNA Sequence (A, T, G, C)", placeholder="e.g. ATGCGATCGATC")
            new_group = st.selectbox("Assign Group", options=["Group A", "Group B"])
            submit_btn = st.form_submit_button("➕ Append Sequence")

            if submit_btn:
                is_valid, cleaned, val_msg = validate_sequence(new_seq)
                if not is_valid:
                    st.error(f"Validation failed: {val_msg}")
                elif not new_id.strip():
                    st.error("Please provide a valid Sequence ID.")
                else:
                    st.session_state.dataset_records.append({
                        "Seq_ID": new_id.strip(),
                        "Sequence": cleaned,
                        "Group": new_group
                    })
                    st.session_state.trained_model = None
                    st.success(f"Added {new_id} ({new_group}) successfully!")
                    st.rerun()

    st.divider()
    st.subheader("Formatted Sequence Visualizer")
    seq_cards = []
    for item in st.session_state.dataset_records:
        seq_cards.append({
            "ID": item["Seq_ID"],
            "Group": item["Group"],
            "Visual": format_dna_html(item["Sequence"]),
            "Raw": item["Sequence"]
        })
    
    st.write("Formatted sequences (A: Green, T: Red, G: Yellow/Amber, C: Blue):")
    html_table = """
    <table style="width:100%; border-collapse: collapse; margin-top: 10px;">
        <thead>
            <tr style="border-bottom: 2px solid #E2E8F0; text-align: left; padding: 8px;">
                <th style="padding: 8px;">Seq ID</th>
                <th style="padding: 8px;">Group</th>
                <th style="padding: 8px;">Sequence Representation</th>
                <th style="padding: 8px;">Length</th>
            </tr>
        </thead>
        <tbody>
    """
    for sc in seq_cards:
        grp_badge = f"<span style='background:#E0E7FF; color:#3730A3; padding: 2px 8px; border-radius: 4px; font-weight:600;'>{sc['Group']}</span>" if sc['Group'] == "Group A" else f"<span style='background:#FCE7F3; color:#9D174D; padding: 2px 8px; border-radius: 4px; font-weight:600;'>{sc['Group']}</span>"
        html_table += f"""
            <tr style="border-bottom: 1px solid #F1F5F9;">
                <td style="padding: 8px; font-weight: bold;">{sc['ID']}</td>
                <td style="padding: 8px;">{grp_badge}</td>
                <td style="padding: 8px;">{sc['Visual']}</td>
                <td style="padding: 8px; color: #64748B;">{len(sc['Raw'])} bp</td>
            </tr>
        """
    html_table += "</tbody></table>"
    st.markdown(html_table, unsafe_allow_html=True)


# =============================================================================
# TAB 2: FEATURE EXTRACTION ENGINE
# =============================================================================
with tab_features:
    st.header("2. Genomic Feature Extraction Engine")
    st.markdown("""
    Automatically computes six numerical features for every sequence in the dataset:
    * **Count of A, T, G, C**: Absolute nucleotide frequencies.
    * **Total Length ($L$)**: Number of base pairs.
    * **GC Content (%)**: Percentage of Guanine and Cytosine bases:
    $$\\text{GC Content (\\%)} = \\frac{\\text{Count}(G) + \\text{Count}(C)}{L} \\times 100$$
    """)

    feat_df = build_feature_dataframe(st.session_state.dataset_records)

    st.subheader("Extracted Feature Table")
    # Display styled feature table
    display_cols = [
        "Seq_ID", "Sequence", "Group", "Count_A", "Count_T", "Count_G", "Count_C",
        "Length", "GC_Content_Pct"
    ]
    st.dataframe(
        feat_df[display_cols].style.background_gradient(
            subset=["GC_Content_Pct"], cmap="Blues"
        ).format({"GC_Content_Pct": "{:.2f}%"}),
        use_container_width=True
    )

    # Download features as CSV
    csv_bytes = feat_df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Feature Table (CSV)",
        data=csv_bytes,
        file_name="genomic_features.csv",
        mime="text/csv"
    )

    st.divider()
    st.subheader("Feature Distributions by Group")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        fig_gc = px.box(
            feat_df,
            x="Group",
            y="GC_Content_Pct",
            color="Group",
            points="all",
            title="GC Content (%) Distribution between Groups",
            color_discrete_map={"Group A": "#3B82F6", "Group B": "#EC4899"}
        )
        fig_gc.update_layout(yaxis_title="GC Content (%)", template="plotly_white")
        st.plotly_chart(fig_gc, use_container_width=True)

    with col_chart2:
        # Group averages for the 4 bases
        base_means = feat_df.groupby("Group")[["Count_A", "Count_T", "Count_G", "Count_C"]].mean().reset_index()
        melted_bases = pd.melt(base_means, id_vars=["Group"], var_name="Nucleotide", value_name="Mean_Count")
        melted_bases["Nucleotide"] = melted_bases["Nucleotide"].str.replace("Count_", "")

        fig_bases = px.bar(
            melted_bases,
            x="Nucleotide",
            y="Mean_Count",
            color="Group",
            barmode="group",
            title="Average Nucleotide Counts by Group",
            color_discrete_map={"Group A": "#3B82F6", "Group B": "#EC4899"}
        )
        fig_bases.update_layout(yaxis_title="Mean Base Count", template="plotly_white")
        st.plotly_chart(fig_bases, use_container_width=True)


# =============================================================================
# TAB 3: MACHINE LEARNING TRAINING
# =============================================================================
with tab_train:
    st.header("3. Machine Learning Classifier Training")
    st.markdown("""
    Train an interpretable classifier on the six extracted features to learn the decision boundary separating
    **Group A** and **Group B**.
    """)

    feat_df = build_feature_dataframe(st.session_state.dataset_records)
    unique_groups = feat_df["Group"].unique()

    train_col_action, train_col_info = st.columns([1, 2])
    with train_col_action:
        st.write(f"**Total Samples:** {len(feat_df)}")
        st.write(f"**Groups Detected:** {', '.join(unique_groups)}")
        
        train_btn = st.button("🚀 Train Classifier Now", type="primary", use_container_width=True)

    with train_col_info:
        if len(unique_groups) < 2:
            st.error("⚠️ Cannot train: At least two distinct groups (e.g. Group A and Group B) are required. Please add samples in Tab 1.")
        else:
            st.success("Ready for training. Features: `Count_A`, `Count_T`, `Count_G`, `Count_C`, `Length`, `GC_Content_Pct`.")

    if train_btn and len(unique_groups) >= 2:
        with st.spinner("Training classification model..."):
            model = GenomeClassifierModel(
                model_type=model_choice,
                max_depth=max_depth,
                n_estimators=n_estimators
            )
            summary = model.train(feat_df)
            st.session_state.trained_model = model
            st.session_state.model_summary = summary
            st.toast("Model trained successfully!", icon="✅")

    # Display results if model is trained
    if st.session_state.model_summary is not None:
        summary = st.session_state.model_summary
        st.divider()
        st.subheader("Model Evaluation & Performance Metrics")

        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Model Architecture", summary["model_type"])
        m_col2.metric("Training Accuracy", f"{summary['train_accuracy']}%")
        m_col3.metric("Sample Size", f"{summary['sample_count']} sequences")

        eval_col1, eval_col2 = st.columns(2)

        with eval_col1:
            st.markdown("#### Confusion Matrix")
            cm = summary["confusion_matrix"]
            classes = summary["classes"]
            fig_cm = px.imshow(
                cm,
                x=classes,
                y=classes,
                text_auto=True,
                labels=dict(x="Predicted Group", y="Actual Group", color="Count"),
                color_continuous_scale="Blues"
            )
            fig_cm.update_layout(title="Confusion Matrix Heatmap", template="plotly_white")
            st.plotly_chart(fig_cm, use_container_width=True)

        with eval_col2:
            st.markdown("#### Feature Importance Ranking")
            imp_df = summary["feature_importance_df"]
            fig_imp = px.bar(
                imp_df,
                x="Importance",
                y="Display_Name",
                orientation="h",
                title="Relative Feature Importance in Decision Splitting",
                color="Importance",
                color_continuous_scale="Viridis"
            )
            fig_imp.update_layout(yaxis=dict(autorange="reversed"), template="plotly_white")
            st.plotly_chart(fig_imp, use_container_width=True)

        # Classification Report Table
        st.markdown("#### Detailed Classification Report")
        rep_dict = summary["classification_report_dict"]
        rep_rows = []
        for cls in summary["classes"]:
            if cls in rep_dict:
                rep_rows.append({
                    "Group": cls,
                    "Precision": f"{rep_dict[cls]['precision']:.2f}",
                    "Recall": f"{rep_dict[cls]['recall']:.2f}",
                    "F1-Score": f"{rep_dict[cls]['f1-score']:.2f}",
                    "Support": rep_dict[cls]['support']
                })
        st.table(pd.DataFrame(rep_rows))

        # Decision Tree Logic Inspection
        if summary["model_type"] == "Decision Tree":
            st.markdown("#### 🌲 Decision Tree Split Logic")
            with st.expander("View Decision Tree Diagram & Textual Rules", expanded=True):
                tree_fig = st.session_state.trained_model.render_tree_diagram()
                if tree_fig:
                    st.pyplot(tree_fig)
                st.markdown("**Machine Learning Rule Representation:**")
                st.code(summary["tree_rules_text"], language="text")
    else:
        st.info("Click **'🚀 Train Classifier Now'** above to generate evaluation metrics, confusion matrix, and feature rankings.")


# =============================================================================
# TAB 4: UNKNOWN SEQUENCE PREDICTION
# =============================================================================
with tab_predict:
    st.header("4. Unknown DNA Sequence Prediction")
    st.markdown("""
    Enter a new, unclassified DNA sequence to predict whether it belongs to **Group A** or **Group B**.
    The sequence is automatically validated, feature-extracted, and evaluated using the trained classifier.
    """)

    # Quick test sequence buttons
    st.write("**Quick Sequence Presets:**")
    preset_cols = st.columns(4)
    if preset_cols[0].button("Assignment Preset (ATGCGCGATC)"):
        st.session_state.unknown_input = "ATGCGCGATC"
    if preset_cols[1].button("GC-Rich Probe (GCGCGCATCG)"):
        st.session_state.unknown_input = "GCGCGCATCG"
    if preset_cols[2].button("AT-Rich Probe (ATATAGCTAT)"):
        st.session_state.unknown_input = "ATATAGCTAT"
    if preset_cols[3].button("Mixed Oligo (CCCCGGGGAA)"):
        st.session_state.unknown_input = "CCCCGGGGAA"

    default_unknown = st.session_state.get("unknown_input", "ATGCGCGATC")
    unknown_seq_input = st.text_input(
        "Enter Unknown DNA Sequence (A, T, G, C):",
        value=default_unknown,
        help="Default preset: ATGCGCGATC as specified in assignment requirements."
    )

    is_valid, cleaned_seq, val_msg = validate_sequence(unknown_seq_input)

    if not is_valid:
        st.error(f"Invalid sequence: {val_msg}")
    else:
        # Extract features for this unknown sequence
        unknown_feats = extract_dna_features(cleaned_seq)

        st.subheader("Extracted Features for Unknown Sequence")
        st.markdown(f"**Sequence Preview:** {format_dna_html(cleaned_seq)}", unsafe_allow_html=True)

        fc1, fc2, fc3, fc4, fc5, fc6 = st.columns(6)
        fc1.metric("Count A", unknown_feats["Count_A"])
        fc2.metric("Count T", unknown_feats["Count_T"])
        fc3.metric("Count G", unknown_feats["Count_G"])
        fc4.metric("Count C", unknown_feats["Count_C"])
        fc5.metric("Length", f"{unknown_feats['Length']} bp")
        fc6.metric("GC Content", f"{unknown_feats['GC_Content_Pct']}%")

        st.divider()

        predict_btn = st.button("🔮 Classify Sequence", type="primary", use_container_width=True)

        if predict_btn:
            if st.session_state.trained_model is None:
                # Auto-train if not yet trained
                feat_df = build_feature_dataframe(st.session_state.dataset_records)
                model = GenomeClassifierModel(model_type=model_choice, max_depth=max_depth)
                model.train(feat_df)
                st.session_state.trained_model = model
                st.session_state.model_summary = model.training_summary

            prediction_result = st.session_state.trained_model.predict_unknown(unknown_feats)
            st.session_state.last_prediction = prediction_result

        if st.session_state.last_prediction is not None:
            res = st.session_state.last_prediction
            pred_group = res["predicted_group"]
            conf = res["confidence_pct"]

            st.markdown("### Prediction Outcome")
            
            badge_color = "#3B82F6" if pred_group == "Group A" else "#EC4899"
            bg_color = "#EFF6FF" if pred_group == "Group A" else "#FDF2F8"
            
            st.markdown(f"""
            <div style="background-color: {bg_color}; border: 2px solid {badge_color}; border-radius: 10px; padding: 20px; margin-bottom: 20px;">
                <h2 style="color: {badge_color}; margin: 0;">Predicted Class: {pred_group}</h2>
                <h4 style="color: #475569; margin-top: 6px;">Classification Confidence: <strong>{conf:.1f}%</strong></h4>
            </div>
            """, unsafe_allow_html=True)

            # Probabilities distribution
            if res["probabilities"]:
                prob_col1, prob_col2 = st.columns(2)
                with prob_col1:
                    prob_df = pd.DataFrame([
                        {"Group": grp, "Probability (%)": p} for grp, p in res["probabilities"].items()
                    ])
                    fig_prob = px.bar(
                        prob_df,
                        x="Group",
                        y="Probability (%)",
                        color="Group",
                        text="Probability (%)",
                        color_discrete_map={"Group A": "#3B82F6", "Group B": "#EC4899"},
                        title="Class Probability Distribution"
                    )
                    fig_prob.update_layout(yaxis_range=[0, 100], template="plotly_white")
                    st.plotly_chart(fig_prob, use_container_width=True)

                with prob_col2:
                    st.markdown("#### 🔬 Biological & Algorithmic Reasoning")
                    for reason in res["reasoning"]:
                        st.markdown(f"- {reason}")
                    
                    st.info(
                        f"For assignment sequence **ATGCGCGATC**, total length is 10 bp with 3 Gs and 3 Cs, "
                        f"yielding a GC content of **60.0%**. Compared to Group A's typical 37.5% - 50.0% GC content, "
                        f"this elevated GC content aligns strongly with the **Group B profile**."
                    )


# =============================================================================
# TAB 5: APPLICATIONS & LIMITATIONS
# =============================================================================
with tab_analysis:
    st.header("5. Genomic Analysis, Applications & Limitations")
    st.markdown("""
    In-depth computational biology analysis addressing model performance, real-world translation, 
    and systemic limitations of ML in genomics.
    """)

    sub_tab1, sub_tab2, sub_tab3 = st.tabs([
        "📊 Model Results & Interpretation",
        "🌍 Real-World Genomics Application",
        "⚠️ Key Limitations of AI/ML in Genomics"
    ])

    with sub_tab1:
        st.subheader("Model Results and Genomic Interpretation")
        st.markdown("""
        #### 1. Why Did the Model Succeed on These Sequences?
        In DNA sequence classification, nucleotide frequency distributions—specifically **GC content percentage**—frequently serve 
        as the primary discriminative axis between distinct genomic elements.

        - **Group A (e.g., S1: ATGCGATC, S2: ATGCGTAC):**
          - Length: 8 bp | G: 2, C: 2 | **GC Content = 50.0%**
          - Equal ratio of AT (4 bases) to GC (4 bases).
        - **Group B (e.g., S3: GCGCGTAT, S4: GCGCGCGC):**
          - S3: G: 3, C: 2 | Length: 8 bp | **GC Content = 62.5%**
          - S4: G: 4, C: 4 | Length: 8 bp | **GC Content = 100.0%**
          - Noticeably enriched in G and C nucleotides.
        
        #### 2. Decision Tree Boundary Logic
        When scikit-learn's `DecisionTreeClassifier` minimizes Gini impurity:
        $$Gini(D) = 1 - \\sum_{i=1}^{C} p_i^2$$
        it establishes a single threshold split on `GC_Content_Pct` (typically at $\\approx 56.25\\%$) or on `Count_G`.
        Any sequence with $\\text{GC\\%} \\le 55.0\\%$ falls into Group A, while $\\text{GC\\%} > 55.0\\%$ falls into Group B.
        Consequently, the unknown test sequence **`ATGCGCGATC`** (60.0% GC content) cleanly crosses the threshold into **Group B**.
        """)

    with sub_tab2:
        st.subheader("Real-World Genomics Application: CpG Island Detection & Epigenetic Gene Regulation")
        st.markdown("""
        <div class="bio-card">
            <h4>Application Case Study: Automated Epigenetic Annotation of CpG Islands</h4>
            <p>
            In vertebrate genomes, cytosine nucleotides occurring next to guanines (<strong>CpG dinucleotides</strong>) 
            are frequently targeted by DNA methyltransferases to form 5-methylcytosine (5mC). Over evolutionary timescales, 
            methylated cytosines spontaneously undergo deamination into thymine, resulting in a systemic depletion of CpG 
            dinucleotides across eukaryotic DNA.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        #### How AI/ML Classifiers are Applied in This Domain:
        1. **Promoter Annotation**: Areas that escape methylation—termed **CpG Islands (CGIs)**—typically exhibit high GC content ($>50\\%$) 
           and elevated CpG observed/expected ratios ($>0.60$). CGIs are clustered in $\\sim 70\\%$ of mammalian gene promoters.
        2. **Cancer Diagnostics**: Aberrant hypermethylation of promoter CpG islands causes transcriptional silencing of critical 
           **tumor suppressor genes** (such as *BRCA1*, *MLH1*, and *CDKN2A*).
        3. **Machine Learning Pipeline**: Just like our classifier distinguishes Group A vs Group B via GC content and nucleotide frequencies, 
           real-world algorithms (e.g., *CpGcluster*, *Epigenome-wide Hidden Markov Models*, and *Random Forests*) scan newly assembled chromosomes 
           in sliding windows to delineate active promoters from quiescent intergenic heterochromatin.
        """)

    with sub_tab3:
        st.subheader("Key Limitation: Loss of Positional & Spatial Context ('Bag-of-Bases' Flaw)")
        st.markdown("""
        <div class="bio-card" style="border-left-color: #DC2626;">
            <h4 style="color: #991B1B;">The Spatial Blindness of Simple Frequency Models</h4>
            <p>
            While nucleotide counts and GC percentage are computationally lightweight, they treat the DNA sequence as an 
            <strong>unordered multiset ("bag-of-bases")</strong>. In molecular biology, sequence order and spatial spacing dictate function.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        #### Critical Manifestations of this Limitation:
        1. **Identical Composition, Divergent Biology:**
           - Consider the sequences `ATGCGCGATC` and `CGTACGTAGT`. Both share identical lengths (10 bp), identical base counts (2 A, 2 T, 3 G, 3 C), 
             and identical GC content (60.0%).
           - Under this feature extraction engine, our model treats them as **identical data points**.
           - In reality, `ATGCGCGATC` contains an exact recognition palindrome for restriction enzymes (*SfoI* / *NarI*), 
             while `CGTACGTAGT` does not.
        2. **Regulatory Motifs Depend on Specific Positional Grammar:**
           - Transcription factor binding sites (e.g., the **TATA box** consensus `TATAWAW`) must reside at precise distances 
             (typically $-25$ to $-30$ base pairs upstream of the Transcription Start Site).
           - Global nucleotide counts completely fail to capture this spatial positioning.
        3. **Modern Bioinformatics Solutions:**
           - To resolve this limitation, state-of-the-art computational biology employs:
             * **k-mer spectrums** (counting 3-mers, 6-mers to capture local dependencies)
             * **Position Weight Matrices (PWMs)**
             * **1D Convolutional Neural Networks (CNNs)** (e.g., *DeepSEA*, *Basset*)
             * **Genomic Transformers** (e.g., *DNABERT*, *HyenaDNA*, *Nucleotide Transformer*).
        """)


# =============================================================================
# TAB 6: CRITICAL DISCUSSION PROMPTS
# =============================================================================
with tab_discussion:
    st.header("6. Critical Discussion Prompts & Guided Q&A")
    st.markdown("Academic reflection questions addressing model behavior, biological fidelity, and edge cases.")

    with st.expander("❓ Discussion Prompt 1: Which features were the most useful, and why?", expanded=True):
        st.markdown("""
        **Answer & Biological Analysis:**
        * **GC Content (%)** and **Count of G** proved to be the most decisive features.
        * **Why?** Group A sequences were designed with an approximately balanced AT:GC ratio (GC content $\\approx 50.0\\%$), 
          whereas Group B sequences represent GC-rich genomic patterns (GC content $\\ge 62.5\\%$ to $100\\%$).
        * Because the target groups split neatly along the GC axis, the tree algorithm assigns nearly 100% of its feature 
          importance split criterion to `GC_Content_Pct` or `Count_G`.
        * Features such as `Length` had zero discriminative value in the default 4-sequence set because all training sequences 
          shared the exact same length (8 base pairs).
        """)

    with st.expander("❓ Discussion Prompt 2: What would happen if an unknown sequence is very different from the training data?", expanded=True):
        st.markdown("""
        **Answer & Distribution Shift Analysis:**
        * **Length Divergence:** If an unknown sequence is 500 base pairs long, raw nucleotide counts (`Count_A = 120`, `Count_G = 150`) 
          will be far outside the training distribution range ($[0, 8]$).
        * If a model relies on raw nucleotide counts rather than normalized percentages, the decision thresholds break down completely.
        * **Ratio Resilience:** Conversely, normalized `GC_Content_Pct` scales well with length. However, a 500 bp sequence with 60% GC 
          is biologically distinct from an 8 bp oligonucleotide with 60% GC, as long stretches may encompass multiple distinct functional domains.
        * **Out-of-Distribution (OOD) Edge Cases:** Poly-A tails (`AAAAAAAAAAAA`) will have `GC_Content_Pct = 0%`, classifying into Group A 
          purely because it is non-Group B, even though a poly-A tail has no biological kinship with the balanced motif of Group A.
        """)

    with st.expander("❓ Discussion Prompt 3: How should ambiguous bases (e.g., 'N' or IUPAC degenerate codes) be handled?", expanded=False):
        st.markdown("""
        **Answer & Sequencing Error Handling:**
        * In next-generation sequencing (NGS), high-throughput sequencers emit the character `'N'` when a base call is ambiguous 
          due to low fluorescence signal or sequencing errors.
        * If unhandled, `'N'` deflates both AT and GC counts. In rigorous pipelines, users can:
          1. Mask and exclude reads with $>5\\%$ ambiguous bases.
          2. Fractionally assign `'N'` as $0.25$ to each base.
          3. Impute base calls using surrounding k-mer context.
        """)

    with st.expander("❓ Discussion Prompt 4: Small Dataset Vulnerability & Overfitting Risks ($N=4$)", expanded=False):
        st.markdown("""
        **Answer & Statistical Learning Perspective:**
        * The default assignment dataset contains only 4 sequences ($N=4$, with $p=6$ features).
        * In statistical learning, when $p \\ge N$, models suffer from the **curse of dimensionality** and can achieve 100% training accuracy 
          by memorizing artifactual noise.
        * While this is pedagogically clear for illustrating decision tree mechanics, a real-world genomic model requires thousands 
          of diverse sequences along with cross-validation to guarantee true generalization.
        """)


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.divider()
st.markdown("""
<div style="text-align: center; color: #94A3B8; font-size: 0.85rem;">
    Predict the Genome Pattern: AI/ML-based classification of DNA sequences | Computational Biology Assignment Solution<br>
    Built with Python, Streamlit, Scikit-learn, and Plotly.
</div>
""", unsafe_allow_html=True)
