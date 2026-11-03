"""
DNA Sequence Processing and Feature Extraction Engine.
Assignment: Predict the Genome Pattern: AI/ML-based classification of DNA sequences
"""

import re
from typing import Dict, Any, Tuple, List
import pandas as pd


VALID_NUCLEOTIDES = set("ATGC")


def validate_sequence(sequence: str) -> Tuple[bool, str, str]:
    """
    Validates and normalizes a DNA sequence string.
    Returns:
        (is_valid, cleaned_seq, error_or_warning_msg)
    """
    if not sequence or not isinstance(sequence, str):
        return False, "", "Sequence is empty."
    
    cleaned = "".join(sequence.split()).upper()
    if not cleaned:
        return False, "", "Sequence contains no valid characters."
    
    invalid_chars = set(cleaned) - VALID_NUCLEOTIDES
    if invalid_chars:
        return False, cleaned, f"Invalid characters detected: {', '.join(sorted(invalid_chars))}. Only A, T, G, C are permitted."
    
    return True, cleaned, "Valid DNA sequence."


def extract_dna_features(sequence: str) -> Dict[str, Any]:
    """
    Extracts the six required numerical features for a given DNA sequence:
    1. Count of nucleotide A
    2. Count of nucleotide T
    3. Count of nucleotide G
    4. Count of nucleotide C
    5. Total sequence length
    6. GC content percentage (%)

    Also returns complementary biological ratios for comprehensive visualization.
    """
    # Clean & normalize
    cleaned = "".join(sequence.split()).upper()
    length = len(cleaned)
    
    if length == 0:
        return {
            "Count_A": 0,
            "Count_T": 0,
            "Count_G": 0,
            "Count_C": 0,
            "Length": 0,
            "GC_Content_Pct": 0.0,
            "AT_Content_Pct": 0.0,
            "Purine_Pct": 0.0,
            "Pyrimidine_Pct": 0.0
        }
    
    count_a = cleaned.count("A")
    count_t = cleaned.count("T")
    count_g = cleaned.count("G")
    count_c = cleaned.count("C")
    
    # Calculate percentages
    gc_pct = round(((count_g + count_c) / length) * 100.0, 2)
    at_pct = round(((count_a + count_t) / length) * 100.0, 2)
    purine_pct = round(((count_a + count_g) / length) * 100.0, 2)
    pyrimidine_pct = round(((count_c + count_t) / length) * 100.0, 2)
    
    return {
        "Count_A": count_a,
        "Count_T": count_t,
        "Count_G": count_g,
        "Count_C": count_c,
        "Length": length,
        "GC_Content_Pct": gc_pct,
        "AT_Content_Pct": at_pct,
        "Purine_Pct": purine_pct,
        "Pyrimidine_Pct": pyrimidine_pct
    }


def get_default_sequences(preset: str = "assignment_core") -> List[Dict[str, str]]:
    """
    Returns curated datasets.
    'assignment_core': The exact four sequences mentioned in the problem prompt.
    'assignment_extended': 10 representative sequences demonstrating balanced AT-rich vs GC-rich groups.
    'cpg_islands': Realistic biological promoter vs non-promoter motif fragments.
    """
    if preset == "assignment_core":
        return [
            {"Seq_ID": "S1", "Sequence": "ATGCGATC", "Group": "Group A"},
            {"Seq_ID": "S2", "Sequence": "ATGCGTAC", "Group": "Group A"},
            {"Seq_ID": "S3", "Sequence": "GCGCGTAT", "Group": "Group B"},
            {"Seq_ID": "S4", "Sequence": "GCGCGCGC", "Group": "Group B"},
        ]
    elif preset == "assignment_extended":
        return [
            # Group A: AT-dominant / Balanced genomic patterns
            {"Seq_ID": "S1", "Sequence": "ATGCGATC", "Group": "Group A"},
            {"Seq_ID": "S2", "Sequence": "ATGCGTAC", "Group": "Group A"},
            {"Seq_ID": "S5", "Sequence": "ATATGCAT", "Group": "Group A"},
            {"Seq_ID": "S6", "Sequence": "TACGATCA", "Group": "Group A"},
            {"Seq_ID": "S7", "Sequence": "AATTGCCA", "Group": "Group A"},
            {"Seq_ID": "S8", "Sequence": "TTACGGAT", "Group": "Group A"},
            # Group B: GC-dominant / CpG island-like patterns
            {"Seq_ID": "S3", "Sequence": "GCGCGTAT", "Group": "Group B"},
            {"Seq_ID": "S4", "Sequence": "GCGCGCGC", "Group": "Group B"},
            {"Seq_ID": "S9", "Sequence": "CGGCCGGC", "Group": "Group B"},
            {"Seq_ID": "S10", "Sequence": "GCGCAGCC", "Group": "Group B"},
            {"Seq_ID": "S11", "Sequence": "CCGGCGCG", "Group": "Group B"},
            {"Seq_ID": "S12", "Sequence": "GGCCGCTA", "Group": "Group B"},
        ]
    elif preset == "cpg_islands":
        return [
            # CpG-poor / Standard intergenic DNA (Low GC content)
            {"Seq_ID": "IG_1", "Sequence": "AAATTTTTAAGCT", "Group": "Group A"},
            {"Seq_ID": "IG_2", "Sequence": "TTAATAAACTTGA", "Group": "Group A"},
            {"Seq_ID": "IG_3", "Sequence": "TATATATGCATAT", "Group": "Group A"},
            {"Seq_ID": "IG_4", "Sequence": "AAACCTTTAATTG", "Group": "Group A"},
            {"Seq_ID": "IG_5", "Sequence": "TTGAAATTCTAAT", "Group": "Group A"},
            # CpG-rich Promoter elements (High GC content)
            {"Seq_ID": "PR_1", "Sequence": "CGCGCCCCGCGGC", "Group": "Group B"},
            {"Seq_ID": "PR_2", "Sequence": "GCGCCGGCCGCGC", "Group": "Group B"},
            {"Seq_ID": "PR_3", "Sequence": "CCGCGCCGCGGCG", "Group": "Group B"},
            {"Seq_ID": "PR_4", "Sequence": "GGCGCGCGGCGCC", "Group": "Group B"},
            {"Seq_ID": "PR_5", "Sequence": "CGGCCGCGCGCGG", "Group": "Group B"},
        ]
    else:
        return get_default_sequences("assignment_core")


def build_feature_dataframe(records: List[Dict[str, str]]) -> pd.DataFrame:
    """
    Transforms a list of sequence dicts into a full DataFrame with the 6 core features.
    """
    rows = []
    for item in records:
        seq_id = item.get("Seq_ID", "").strip()
        raw_seq = item.get("Sequence", "").strip().upper()
        group = item.get("Group", "").strip()
        
        is_valid, cleaned, msg = validate_sequence(raw_seq)
        feat = extract_dna_features(cleaned if is_valid else raw_seq)
        
        row = {
            "Seq_ID": seq_id,
            "Sequence": cleaned if is_valid else raw_seq,
            "Group": group,
            "Count_A": feat["Count_A"],
            "Count_T": feat["Count_T"],
            "Count_G": feat["Count_G"],
            "Count_C": feat["Count_C"],
            "Length": feat["Length"],
            "GC_Content_Pct": feat["GC_Content_Pct"],
            "Is_Valid": is_valid,
            "Validation_Note": msg
        }
        rows.append(row)
        
    return pd.DataFrame(rows)
