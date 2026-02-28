# RT-Predict Benchmark

**Can you predict which reverse transcriptases will work for prime editing — without memorizing their family?**

## The Problem

Prime editing is a powerful genome-editing technology that uses a reverse transcriptase (RT) enzyme fused to a Cas9 nickase. The RT copies an RNA template to write precise edits into the genome. But here's the bottleneck: **only a handful of the thousands of known RT enzymes actually work for prime editing**, and discovering which ones work requires expensive, months-long experiments.

This benchmark asks: **given a reverse transcriptase protein sequence and its computed biophysical/structural properties, can you predict whether it will be active for prime editing, and how efficient it will be?**

The dataset comes from [Doman et al. (2023)](https://doi.org/10.1038/s41587-023-01905-6), who experimentally tested 57 diverse RT enzymes for prime editing activity.

## Why This Is Hard

This looks like a simple binary classification problem (57 samples, 98 features). It isn't. The dataset has pathologies that will eat naive approaches alive:

### Failure Mode 1: Family Memorization
The 57 RTs come from 7 evolutionary families. **The active/inactive ratio is heavily confounded with family membership.** Retroviral RTs are mostly active (12/18). Non-retroviral families are mostly inactive. A model that learns "Retroviral → active" gets ~75% accuracy but may have learned family identity rather than the underlying biophysics of what makes an RT work.

**The goal is to ensure your model's performance isn't purely explained by family distribution.** We use leave-one-family-out cross-validation as a sanity check alongside standard evaluation — not as the only metric, but as evidence that the model has learned something real.

### Failure Mode 2: Class Imbalance Masking
21 active, 36 inactive. A model that predicts "inactive" for everything gets 63% accuracy. **Do not report accuracy alone. F1 on the positive class is what matters.**

### Failure Mode 3: The Retroviral Wall
The Retroviral family contains 12 of 21 active RTs (57%). When held out during cross-validation, your model must predict Retroviral activity from patterns learned on non-retroviral families — which have only 9 active RTs across 3 families. **Every approach we've tested gets TP=2/12 on the Retroviral fold.** Breaking through this wall is the central challenge.

### Failure Mode 4: AUC on Single-Class Folds
Three families (CRISPR-associated, Other, Unclassified) have zero active members. AUC is undefined for these folds. A model predicting "inactive" for all gets Acc=1.0 on these folds. **Don't count these folds as successes.**

## Dataset

```
data/
├── rt_sequences.csv          # 57 RTs: name, sequence, active, PE efficiency, family
├── handcrafted_features.csv  # 98 biophysical features per RT
├── esm2_embeddings.npz       # ESM-2 1280-dim mean-pooled embeddings
├── family_splits.csv         # Family membership and class balance
├── feature_dictionary.csv    # What each of the 98 features means
└── structures/               # AlphaFold2/ESMFold predicted 3D structures (57 PDB files)
```

### rt_sequences.csv
| Column | Description |
|---|---|
| `rt_name` | Unique identifier |
| `sequence` | Full amino acid sequence |
| `active` | 1 = active for prime editing, 0 = inactive |
| `pe_efficiency_pct` | Prime editing efficiency (%) — 0 for inactive, 1.5–41% for active |
| `rt_family` | Evolutionary family (7 families) |
| `protein_length_aa` | Sequence length |

### Family Breakdown
| Family | n | Active | Inactive | Notes |
|---|---|---|---|---|
| Retroviral | 18 | 12 | 6 | Includes MMLV (41%), the gold standard |
| Retron | 12 | 5 | 7 | Bacterial RTs |
| LTR_Retrotransposon | 11 | 2 | 9 | Mobile genetic elements |
| CRISPR-associated | 5 | 0 | 5 | All inactive |
| Group_II_Intron | 5 | 2 | 3 | Self-splicing elements |
| Other | 5 | 0 | 5 | All inactive |
| Unclassified | 1 | 0 | 1 | Single sample |

### handcrafted_features.csv
98 handcrafted biophysical features computed from predicted structures (AlphaFold2/ESMFold) and sequences. These are the result of extensive feature engineering for feature-rich extraction. Grouped into 15 categories:

| Group | # Features | Description |
|---|---|---|
| ESM-IF Perplexity | 2 | Inverse folding scores — does the sequence "fit" the predicted structure? |
| ProtParam | 8 | Physicochemical properties: MW, aromaticity, instability, pI, secondary structure fractions, hydropathy |
| Net Charge | 5 | Charged residue counts and net charge at pH 7 |
| Thermostability | 11 | Predicted fraction retaining structure at 40–80°C, thermophilicity class |
| Asp Triad | 4 | Catalytic triad geometry — RMSD to known active RT catalytic sites |
| Contacts | 11 | Structural contacts: hydrophobic, salt bridges, H-bonds (whole protein + active-site pocket) |
| Hydrophobicity | 5 | Hydrophobic residue distribution, including near the YXDD catalytic motif |
| Hairpin | 9 | Beta-hairpin detection near active site, strand/helix content |
| Thumb Domain | 8 | Thumb subdomain surface charge (aligned to HIV-RT reference) |
| DGR Motif | 1 | Presence of diversity-generating retroelement motif |
| Procheck | 6 | Ramachandran quality and stereochemical G-factors |
| CamSol Solubility | 5 | Intrinsic solubility predictions: overall score, per-residue profile stats, fraction of poor/good residues |
| SASA | 5 | Solvent-accessible surface area of 20 active-site residues: total, average, exposure distribution |
| Motif Secondary Structure | 8 | DSSP secondary structure at conserved catalytic motifs (YXDD, QG, SP): strand/turn/helix assignments |
| FoldSeek Structural Alignment | 10 | TM-scores and lDDT from FoldSeek alignment against 25+ reference RT crystal structures, per reference family |

See `feature_dictionary.csv` for detailed per-feature descriptions.

### structures/
AlphaFold2/ESMFold predicted 3D structures for all 57 RT enzymes in PDB format. These structures were used to compute the structural biophysical features in `handcrafted_features.csv` (e.g., Asp triad geometry, contacts, SASA, hairpin detection, thumb domain analysis, FoldSeek alignments, Procheck validation, and motif secondary structure assignments).

**Missing values**: Some structural features have NaN for RTs where the structural analysis pipeline failed (e.g., no thumb domain detected, no YXDD motif found, no PDB structure available for Line1-RT). Missing ≠ zero — handle accordingly.

**FoldSeek warning**: TM-scores to reference families (e.g., `foldseek_TM_MMLV`) encode structural similarity to known RT families. These are biophysically meaningful but are correlated with family membership. Use with care in leave-one-family-out evaluation.

### esm2_embeddings.npz
Pre-computed ESM-2 (650M parameter protein language model) embeddings, 1280 dimensions, mean-pooled across residue positions. Load with:
```python
data = np.load("data/esm2_embeddings.npz", allow_pickle=True)
names = data['names']       # (57,) RT names
embeddings = data['embeddings']  # (57, 1280) float32
```

**Warning**: ESM-2 embeddings encode evolutionary family membership extremely well. Models trained on raw embeddings will achieve high LOO-CV accuracy but fail catastrophically on leave-one-family-out. They memorize family, not function.

## Evaluation Protocol

We care about two things: **(1) does the model actually learn biophysical determinants of RT activity, rather than just memorizing family identity?** and **(2) does the model perform well in practice?** These are complementary goals, not competing ones.

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

LOFO tests whether your model can generalize to an entirely unseen evolutionary lineage. We use this primarily as a **sanity check** — to make sure good results aren't simply explained by the model learning "Retroviral → active, everything else → inactive." A model that performs well on LOFO has clearly learned something beyond family distribution.

That said, **we are not dogmatic about LOFO being the only metric.** Learning family-specific biophysical patterns is real biology, not cheating. A model that learns "within Retroviral RTs, these structural features distinguish active from inactive" has learned something genuinely useful — even if that knowledge doesn't transfer perfectly to CRISPR-associated RTs.

### Within-Family / Leave-One-Out CV

Standard LOO-CV across all 57 RTs. This measures within-distribution performance, including the ability to discriminate active from inactive RTs within the same family. Strong within-family performance is valuable — it means the model captures family-specific determinants of activity, which is practically useful for screening novel RTs from known families.

**Caveat**: LOO-CV alone can be inflated by family memorization, which is why we ask for both LOO and LOFO results.

### Ranking Quality
For the 21 active RTs, how well does your predicted score rank them by actual PE efficiency? Report Spearman's ρ and Kendall's τ.

### Rules
1. **Report both LOFO and LOO-CV**: we want to see how the model performs across families *and* within them. Neither metric alone tells the full story.
2. **Report the Retroviral fold separately**: this is the hardest LOFO fold. TP/12 on this fold is the most informative single number for cross-family generalization.
3. **External data is allowed**: you may use additional protein databases, language model embeddings, structural predictions, etc. But you must describe what you used.
4. **No manual curation of predictions**: the pipeline must be fully automated and reproducible.

## Baselines

We've extensively tested approaches. Here are the results on leave-one-family-out:

| Approach | F1 | AUC | TP/21 | Retroviral TP/12 | Notes |
|---|---|---|---|---|---|
| Predict all inactive | 0.000 | 0.500 | 0/21 | 0/12 | Trivial baseline |
| ESM-2 + Ridge (α=1000) | 0.000 | 0.548 | 0/21 | 0/12 | Embeddings memorize family |
| ESM-2 + RF (d=10) | 0.000 | 0.485 | 0/21 | 0/12 | Same problem |
| **HandCrafted + RF (d=10)** | **0.533** | **0.777** | **8/21** | **2/12** | Best overall (98 features) |
| HandCrafted + LogReg (C=0.01) | 0.467 | 0.460 | 7/21 | 2/12 | |
| PLS experts + RF | 0.533 | 0.495 | 8/21 | 2/12 | PLS compression doesn't help |
| PLS + LightGBM LambdaRank | 0.524 | 0.759 | 11/21 | 6/12 | Best AUC, but 10 FPs |
| Within-family pairwise SVM | 0.426 | 0.563 | 10/21 | 2/12 | More TPs, many more FPs |

**The wall**: no approach gets more than 2 TPs on the Retroviral fold without also producing many false positives. Breaking TP>4/12 on Retroviral with FP<5 would be a significant advance.

### Baseline scripts
- `baselines/baseline_rf.py` — Random Forest on handcrafted features (F1=0.533)
- `evaluation/evaluate.py` — Standard evaluation script; pass it your predictions CSV

## Submission Format

Submit a CSV with columns: `rt_name`, `predicted_active`, `predicted_score`

```csv
rt_name,predicted_active,predicted_score
MMLV-RT,1,0.95
BLV-RT,0,0.12
...
```

`predicted_active`: binary 0/1 prediction
`predicted_score`: continuous score (higher = more likely active), used for AUC and ranking


## File Structure

```
rt-predict-benchmark/
├── README.md                      # This file
├── data/
│   ├── rt_sequences.csv           # Sequences + labels + families
│   ├── handcrafted_features.csv   # 98 handcrafted biophysical features
│   ├── esm2_embeddings.npz        # ESM-2 1280-dim embeddings
│   ├── family_splits.csv          # Family definitions
│   ├── feature_dictionary.csv     # Feature descriptions
│   └── structures/                # 57 predicted 3D structures (PDB format)
├── baselines/
│   └── baseline_rf.py             # Reference baseline implementation
├── evaluation/
│   └── evaluate.py                # Official evaluation script
└── export_data.py                 # Script used to generate data/
```
