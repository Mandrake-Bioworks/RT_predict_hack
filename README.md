# Mandrake Open Problems #1: The Retroviral Wall

**Can you predict which reverse transcriptases will work for prime editing — without memorising their evolutionary family?**

A machine learning challenge by [Mandrake Bioworks](https://www.mandrakebio.com) — an AI-first gene editing company building the next generation of gene editing tools.

📅 **Submissions close:** March 31, 2026
🏆 **Details & prizes:** https://www.mandrakebio.com/Challenges/retroviral-wall/
📧 **Submit to:** challenges@mandrakebio.com

---

## The Problem

Prime editing is arguably the most precise genome editing technology ever built. It can write any of the 12 possible point mutations, make small insertions and deletions, and do it all without double-strand breaks or donor DNA templates. Where earlier CRISPR tools cut and hope, prime editing writes exactly what you want. It has the potential to correct the majority of known pathogenic mutations in the human genome.

At the heart of every prime editor is a **reverse transcriptase (RT)** enzyme fused to a Cas9 nickase. The RT reads an RNA template and writes the desired edit directly into DNA. It determines whether the edit works, how efficiently it works, and in which cell types it works. Today's best prime editors use MMLV-derived RTs — large enzymes (~700 amino acids) that are difficult to package into viral delivery vectors and challenging to deliver therapeutically. Smaller, more efficient RTs would unlock prime editing for a far wider range of applications.

Nature has evolved thousands of RTs across retroviruses, bacteria, mobile genetic elements, and more. Some will work for prime editing. Most won't. Experimentally screening thousands of candidates is expensive and slow — each round takes months of cloning, expression, and activity measurement. This is the fundamental bottleneck in enzyme engineering: the dry-lab to wet-lab loop. Lab testing is rate-limiting, and the datasets that come back are inevitably small and sparse. **Every candidate you can eliminate computationally is a month saved at the bench.**

Unlike antibody engineering, where binding affinity provides a single, well-defined metric to optimise against, enzyme function has no universal objective. An RT that works for prime editing must satisfy multiple constraints simultaneously — catalytic efficiency, processivity, fidelity, thermostability, structural compatibility with the Cas9 fusion, and more. There is no single number to hill-climb. Prediction requires learning from an ensemble of biophysical signals, many of which interact in ways that are poorly understood.

**This benchmark asks:** given a reverse transcriptase protein sequence and its computed biophysical/structural properties, can you predict whether it will be active for prime editing, and how efficient it will be?

The dataset comes from Doman et al. (2023), who experimentally tested 57 diverse RT enzymes for prime editing activity.

---

## Why This Is Hard

This looks like a simple binary classification problem (57 samples, 98 features). It isn't.

Yes, 57 samples with 98 features is a small-data problem. Standard ML approaches will overfit. That's the point — we want to see who can extract real signal from limited, high-dimensional biological data. This mirrors the reality of experimental biology, where datasets are expensive and small. **Cracking small-data biological prediction is the actual challenge.**

The dataset has pathologies that will eat naive approaches alive:

### Failure Mode 1: Family Memorisation

The 57 RTs come from 7 evolutionary families. The active/inactive ratio is heavily confounded with family membership. Retroviral RTs are mostly active (12/18). Non-retroviral families are mostly inactive. A model that learns "Retroviral → active" gets ~75% accuracy but may have learned family identity rather than the underlying biophysics of what makes an RT work.

The goal is to ensure your model's performance isn't purely explained by family distribution. We use leave-one-family-out cross-validation as a sanity check alongside standard evaluation — not as the only metric, but as evidence that the model has learned something real.

### Failure Mode 2: Class Imbalance Masking

21 active, 36 inactive. A model that predicts "inactive" for everything gets 63% accuracy. **Do not report accuracy alone.** F1 on the positive class is what matters.

### Failure Mode 3: The Retroviral Wall

The Retroviral family contains 12 of 21 active RTs (57%). When held out during cross-validation, your model must predict Retroviral activity from patterns learned on non-retroviral families — which have only 9 active RTs across 3 families. **Breaking through this wall is the central challenge.**

### Failure Mode 4: AUC on Single-Class Folds

Three families (CRISPR-associated, Other, Unclassified) have zero active members. AUC is undefined for these folds. A model predicting "inactive" for all gets Acc=1.0 on these folds. Don't count these folds as successes.

---

## Dataset

This dataset goes beyond raw activity labels. We've enriched every RT using our computational platform to maximise the signal available from 57 experimentally tested enzymes.

```
data/
├── rt_sequences.csv          # 57 RTs: name, sequence, active, PE efficiency, family
├── handcrafted_features.csv  # 98 biophysical features per RT
├── esm2_embeddings.npz       # ESM-2 1280-dim mean-pooled embeddings
├── family_splits.csv         # Family membership and class balance
├── feature_dictionary.csv    # What each of the 98 features means
└── structures/               # ESMFold predicted 3D structures (57 PDB files)
```

You are free to use any additional tools, models, or databases — protein language models, structural predictors, external RT datasets, novel featurisation approaches, anything. Document what you use in your writeup.

### rt_sequences.csv

| Column | Description |
|--------|-------------|
| `rt_name` | Unique identifier |
| `sequence` | Full amino acid sequence |
| `active` | 1 = active for prime editing, 0 = inactive |
| `pe_efficiency_pct` | Prime editing efficiency (%) — 0 for inactive, 1.5–41% for active |
| `rt_family` | Evolutionary family (7 families) |
| `protein_length_aa` | Sequence length |

### Family Breakdown

| Family | n | Active | Inactive | Notes |
|--------|---|--------|----------|-------|
| Retroviral | 18 | 12 | 6 | Includes MMLV (41%), the gold standard |
| Retron | 12 | 5 | 7 | Bacterial RTs |
| LTR_Retrotransposon | 11 | 2 | 9 | Mobile genetic elements |
| CRISPR-associated | 5 | 0 | 5 | All inactive |
| Group_II_Intron | 5 | 2 | 3 | Self-splicing elements |
| Other | 5 | 0 | 5 | All inactive |
| Unclassified | 1 | 0 | 1 | Single sample |

### handcrafted_features.csv

98 biophysical features computed from predicted structures and sequences, grouped into 15 categories:

| Group | # Features | Description |
|-------|-----------|-------------|
| ESM-IF Perplexity | 2 | Inverse folding scores |
| ProtParam | 8 | Physicochemical properties: MW, aromaticity, instability, pI, secondary structure fractions, hydropathy |
| Net Charge | 5 | Charged residue counts and net charge at pH 7 |
| Thermostability | 11 | Predicted fraction retaining structure at 40–80°C |
| Asp Triad | 4 | Catalytic triad geometry |
| Contacts | 11 | Structural contacts: hydrophobic, salt bridges, H-bonds |
| Hydrophobicity | 5 | Hydrophobic residue distribution |
| Hairpin | 9 | Beta-hairpin detection near active site |
| Thumb Domain | 8 | Thumb subdomain surface charge |
| DGR Motif | 1 | Presence of diversity-generating retroelement motif |
| Procheck | 6 | Ramachandran quality and stereochemical G-factors |
| CamSol Solubility | 5 | Intrinsic solubility predictions |
| SASA | 5 | Solvent-accessible surface area of active-site residues |
| Motif Secondary Structure | 8 | Secondary structure at conserved catalytic motifs |
| FoldSeek Structural Alignment | 10 | TM-scores from structural alignment against reference RT crystal structures |

See `feature_dictionary.csv` for per-feature descriptions.

### structures/

ESMFold predicted 3D structures for all 57 RT enzymes in PDB format.

**Missing values:** Some structural features have NaN for RTs where the structural analysis pipeline could not resolve certain elements (e.g., no thumb domain detected, no YXDD motif found). Missing ≠ zero — handle accordingly.

**FoldSeek warning:** TM-scores to reference families (e.g., `foldseek_TM_MMLV`) encode structural similarity to known RT families. These are biophysically meaningful but are correlated with family membership. Use with care in leave-one-family-out evaluation.

### esm2_embeddings.npz

Pre-computed ESM-2 (650M parameter) embeddings, 1280 dimensions, mean-pooled across residue positions.

```python
data = np.load("data/esm2_embeddings.npz", allow_pickle=True)
names = data['names']       # (57,) RT names
embeddings = data['embeddings']  # (57, 1280) float32
```

**Warning:** ESM-2 embeddings encode evolutionary family membership extremely well. Models trained on raw embeddings will achieve high LOO-CV accuracy but fail on leave-one-family-out. They memorise family, not function.

---

## Evaluation

### Stage 1: Cross-Validation (March 2026)

All submissions evaluated on:

**Primary metric:** LOFO Macro-F1 across the 4 informative folds (Retroviral, Retron, LTR_Retrotransposon, Group_II_Intron). The three all-inactive families are excluded from the primary metric.

**Secondary:**
- Retroviral fold TP/12 — the single most informative number for cross-family generalisation
- LOO-CV F1 — within-distribution performance

**Bonus:**
- Spearman's ρ and Kendall's τ on ranking active RTs by PE efficiency

Top submissions selected based on quantitative performance and methodological novelty.

### Stage 2: Proprietary Validation (Q2 2026)

Finalist approaches will be run on Mandrake's internal experimental data — RT enzymes not in this dataset, tested in our lab. The final winner will be determined by generalisation to this unseen data.

### Leave-One-Family-Out (LOFO) Cross-Validation

```
For each of the 7 families:
    1. Hold out all RTs from that family as the test set
    2. Train on the remaining 6 families
    3. Predict active/inactive for the held-out family
    4. Record predictions

Aggregate all predictions across all 7 folds.
Report: F1 (positive class), AUC-ROC, TP, FP, FN, TN
```

LOFO tests whether your model can generalise to an entirely unseen evolutionary lineage. A model that performs well on LOFO has learned something beyond family distribution.

That said, we are not dogmatic about LOFO being the only metric. Learning family-specific biophysical patterns is real biology, not cheating. A model that learns "within Retroviral RTs, these structural features distinguish active from inactive" has learned something genuinely useful.

### Within-Family / Leave-One-Out CV

Standard LOO-CV across all 57 RTs. This measures within-distribution performance, including the ability to discriminate active from inactive within the same family.

**Caveat:** LOO-CV alone can be inflated by family memorisation, which is why we ask for both LOO and LOFO results.

### Ranking Quality

For the 21 active RTs, how well does your predicted score rank them by actual PE efficiency? Report Spearman's ρ and Kendall's τ.

---

## Baselines

Here are the approaches we've tested on leave-one-family-out:

| Approach | F1 | AUC | TP/21 | Retroviral TP/12 | Notes |
|----------|-----|-----|-------|------------------|-------|
| Predict all inactive | 0.000 | 0.500 | 0/21 | 0/12 | Trivial baseline |
| ESM-2 + Ridge (α=1000) | 0.000 | 0.548 | 0/21 | 0/12 | Embeddings memorise family |
| ESM-2 + RF (d=10) | 0.000 | 0.485 | 0/21 | 0/12 | Same problem |
| HandCrafted + RF (d=10) | 0.533 | 0.777 | 8/21 | 2/12 | Best overall |
| HandCrafted + LogReg (C=0.01) | 0.467 | 0.460 | 7/21 | 2/12 | |
| PLS experts + RF | 0.533 | 0.495 | 8/21 | 2/12 | PLS compression doesn't help |
| PLS + LightGBM LambdaRank | 0.524 | 0.759 | 11/21 | 6/12 | Best AUC, but 10 FPs |
| Within-family pairwise SVM | 0.426 | 0.563 | 10/21 | 2/12 | More TPs, many more FPs |

---

## Submission

Email to **challenges@mandrakebio.com** by **March 31, 2026** with:

1. **Code** — Full runnable pipeline (notebook or scripts) with a `requirements.txt`. We must be able to reproduce your results.
2. **Predictions CSV** — `rt_name`, `predicted_active`, `predicted_score`
3. **Writeup** — 1-2 pages. What you did, why, what worked, what didn't.

See `submissions/example_submission.csv` for the expected format:

```csv
rt_name,predicted_active,predicted_score
MMLV-RT,1,0.95
BLV-RT,0,0.12
...
```

- `predicted_active`: binary 0/1 prediction
- `predicted_score`: continuous score (higher = more likely active), used for AUC and ranking

---

## Rules

1. **Report both LOFO and LOO-CV** — we want to see performance across families and within them.
2. **Report the Retroviral fold separately** — TP/12 on this fold is the most informative single number.
3. **External data allowed** — protein databases, language model embeddings, structural predictions, etc. Describe everything you used.
4. **Reproducibility required** — the pipeline must be fully automated. We will run your code independently.
5. **No manual curation of predictions.**

---

## File Structure

```
Retroviral-Wall-Challenge/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── rt_sequences.csv
│   ├── handcrafted_features.csv
│   ├── esm2_embeddings.npz
│   ├── family_splits.csv
│   ├── feature_dictionary.csv
│   └── structures/
└── submissions/
    └── example_submission.csv
```

---

## Compute Credits

Have a compelling idea but need GPU resources to execute? Email us your proposed approach at challenges@mandrakebio.com. We provide compute credits on a case-by-case basis for ideas we find promising.

---

## License

**Code:** MIT License. See [LICENSE](LICENSE).

**Data:** Derived from Doman et al. (2023). Features and embeddings provided under CC-BY-4.0.

---

*A challenge by [Mandrake Bioworks](https://www.mandrakebio.com) — making life programmable.*
