# 🧬 Predict the Genome Pattern: AI/ML-based Classification of DNA Sequences

> **Course / Assignment Title:** Predict the Genome Pattern: AI/ML-based classification of DNA sequences  
> **Domain:** Computational Biology, Bioinformatics & Applied Machine Learning  
> **Tech Stack:** Python 3.13, Streamlit, Scikit-learn, Pandas, NumPy, Plotly, Matplotlib  

---

## 📖 Executive Summary

This project provides an interactive computational biology web application designed to classify DNA sequences into distinct functional groups (**Group A** vs **Group B**) using automated feature extraction and machine learning classifiers (Decision Trees & Random Forests). 

It features an end-to-end biological machine learning pipeline:
1. **Interactive Dataset Management:** View, edit, add, and export sequence records.
2. **Genomic Feature Extraction Engine:** Real-time computation of 6 core numerical features:
   - Nucleotide counts: $\text{Count}(A), \text{Count}(T), \text{Count}(G), \text{Count}(C)$
   - Total sequence length ($L$)
   - GC Content percentage ($\text{GC}\% = \frac{G+C}{L} \times 100\%$)
3. **Interpretable Machine Learning Training:** Decision Tree and Random Forest classifiers with confusion matrix heatmaps, classification metrics, feature importance rankings, and graphical decision tree diagrams.
4. **Unknown Sequence Prediction Engine:** Classification of test sequences (default preset: `ATGCGCGATC`) with class probabilities and biological reasoning.
5. **Academic Analysis & Literature Review:**
   - Model results and mathematical interpretation.
   - Real-world application: **CpG Island Detection & Epigenetic Regulation in Cancer**.
   - Fundamental limitation: **Loss of Positional & Spatial Context ("Bag-of-Bases" limitation)**.
   - Guided critical discussion prompts addressing distribution shift, sequence length divergence, and ambiguous base handling.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure Python 3.10+ (tested on Python 3.13) is installed.

### 2. Launch the Application

#### Option A: One-Click Launch (Windows)
Double-click `run.bat` or run:
```powershell
.\run.bat
```
or via PowerShell:
```powershell
.\run.ps1
```

#### Option B: Terminal Command
From the project directory:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📁 Repository Structure

```
genome_pattern_classifier/
├── app.py              # Main Streamlit web application & user interface
├── dna_engine.py       # Sequence validation, normalization & feature extraction engine
├── ml_engine.py        # Machine learning wrapper (training, evaluation, visualization & prediction)
├── test_app.py         # Automated test suite for feature extraction and model prediction
├── requirements.txt    # Python dependencies
├── run.bat             # Windows batch launch script
├── run.ps1             # Windows PowerShell launch script
└── README.md           # Project documentation and academic report
```

---

## 🧬 Feature Extraction Details

For every DNA sequence $S = s_1 s_2 \dots s_L$ where $s_i \in \{A, T, G, C\}$, the engine extracts six features:

| Feature | Notation | Mathematical Definition | Biological Relevance |
| :--- | :--- | :--- | :--- |
| **Count of A** | $\text{Count}(A)$ | $\sum_{i=1}^{L} \mathbb{I}(s_i = \text{'A'})$ | AT-rich promoter elements (TATA boxes, poly-A signals) |
| **Count of T** | $\text{Count}(T)$ | $\sum_{i=1}^{L} \mathbb{I}(s_i = \text{'T'})$ | Transcription termination signals, melting point depression |
| **Count of G** | $\text{Count}(G)$ | $\sum_{i=1}^{L} \mathbb{I}(s_i = \text{'G'})$ | G-quadruplex structures, CpG islands |
| **Count of C** | $\text{Count}(C)$ | $\sum_{i=1}^{L} \mathbb{I}(s_i = \text{'C'})$ | Epigenetic methylation targets (5mC) |
| **Sequence Length** | $L$ | $|S|$ | Normalization denominator & motif scale |
| **GC Content (%)** | $\text{GC}\%$ | $\frac{\text{Count}(G) + \text{Count}(C)}{L} \times 100\%$ | DNA duplex thermal stability ($T_m$), genomic density |

---

## 🔬 Computational Biology Academic Analysis

### 1. Model Results and Interpretation
When evaluated on the core assignment dataset:
- **Group A:** Balanced / AT-rich sequences
  - $S_1$ (`ATGCGATC`): Length = 8, $\text{Count}(G)=2, \text{Count}(C)=2$, $\text{GC} = 50.0\%$
  - $S_2$ (`ATGCGTAC`): Length = 8, $\text{Count}(G)=2, \text{Count}(C)=2$, $\text{GC} = 50.0\%$
- **Group B:** GC-enriched sequences
  - $S_3$ (`GCGCGTAT`): Length = 8, $\text{Count}(G)=3, \text{Count}(C)=2$, $\text{GC} = 62.5\%$
  - $S_4$ (`GCGCGCGC`): Length = 8, $\text{Count}(G)=4, \text{Count}(C)=4$, $\text{GC} = 100.0\%$

**Classification Logic:**  
The decision tree achieves $100\%$ training accuracy by identifying a single decision threshold on **$\text{GC Content (\%)} \le 56.25\%$** (or alternatively $\text{Count}(G) \le 2.5$). 
For the unknown sequence **`ATGCGCGATC`**:
- Length: 10 bp
- $\text{Count}(A)=2, \text{Count}(T)=2, \text{Count}(G)=3, \text{Count}(C)=3$
- $\text{GC Content} = \frac{3 + 3}{10} \times 100 = 60.0\%$

Because $60.0\% > 56.25\%$ (and $\text{Count}(G) = 3 > 2.5$), the sequence traverses to the right branch and is classified as **Group B**.

---

### 2. Real-World Genomics Application: CpG Island Detection & Epigenetics
* **Biological Context:** In vertebrate genomes, the dinucleotide CpG is predominantly methylated at the 5-position of cytosine (5mC). Spontaneous deamination of 5mC converts it to thymine ($T$), which leads to a progressive evolutionary depletion of CpGs across the genome.
* **The CpG Island (CGI) Exception:** Unmethylated genomic islands with GC content $> 50\%$ and observed-to-expected CpG ratio $> 0.60$ escape this depletion and remain clustered around the transcription start sites (TSS) of $\sim 70\%$ of mammalian genes.
* **Clinical Significance:** In cancer cells, aberrant hypermethylation of promoter CpG islands causes pathological silencing of key tumor suppressor genes (e.g., *p16/CDKN2A*, *BRCA1*, *MLH1*).
* **Machine Learning Role:** Automated sequence classifiers scan newly assembled chromosomes in sliding windows to pinpoint putative CpG islands and predict transcriptional promoter competence.

---

### 3. Key Limitation: Loss of Positional & Spatial Grammar ("Bag-of-Bases")
* **The Limitation:** Representing DNA by 1-mer nucleotide counts collapses sequence order into an unordered multiset.
* **Concrete Example:** The sequences `ATGCGCGATC` and `CGTACGTAGT` possess identical lengths (10 bp), identical base counts ($2A, 2T, 3G, 3C$), and identical GC content ($60.0\%$). To our 6-feature classifier, they are indistinguishable.
* **Biological Reality:** `ATGCGCGATC` contains the restriction enzyme cleavage palindrome `GGCGCC` / `CGCG`, whereas `CGTACGTAGT` does not. Similarly, functional promoter motifs such as the **TATA box** (`TATAAA`) depend on their exact spatial location ($-25$ to $-30$ bp upstream of TSS).
* **Bioinformatics Solution:** Modern computational pipelines transition from simple base counts to $k$-mer spectrums ($k=3, 6$), Position Weight Matrices (PWMs), 1D Convolutional Neural Networks (CNNs like *DeepSEA*), and Genomic Foundation Models (*DNABERT*, *HyenaDNA*).

---

### 4. Critical Discussion Prompts

#### Q1: Which features were the most useful, and why?
> **Answer:** **GC Content (%)** and **Count of G** were the most useful features. Group A is characterized by balanced composition ($\text{GC}\approx 50\%$), while Group B represents GC-rich elements ($\text{GC}\ge 62.5\%$). Features like `Length` provided zero discriminative information in the default training set because all sequences were of identical length (8 bp).

#### Q2: How does the model behave with sequences very different from the training data?
> **Answer:** 
> 1. **Length Shifts:** For an unknown sequence of 500 bp, raw nucleotide counts (e.g., $\text{Count}(G)=150$) wildly exceed the training set boundaries ($[0, 4]$), causing count-based decision thresholds to fail. Normalized features like `GC Content (%)` are robust to scale differences.
> 2. **Extreme Out-of-Distribution Sequences:** Repetitive motifs (e.g., poly-A tail `AAAAAAAAAA`, $\text{GC}=0\%$) would be misclassified into Group A purely because $\text{GC} \le 56.25\%$, despite bearing no biological functional relation to Group A.

#### Q3: How should ambiguous bases ('N') be handled?
> **Answer:** Ambiguous bases resulting from low-confidence sequencer reads should either be masked, excluded if exceeding a threshold (e.g., $>5\%$), or fractionally assigned ($0.25$ per nucleotide) to avoid deflating GC content estimates.
