#!/usr/bin/env python3
"""
Export clean competition datasets from raw project files.
Run once to populate data/ folder.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SRC = Path("/Users/aryanchandak/Documents/Mandrake work/similar experimental datasets")
DST = SRC / "rt-predict-benchmark" / "data"
DST.mkdir(exist_ok=True)

# ============================================================
# 1. Main dataset: sequences + labels + metadata
# ============================================================
base = pd.read_csv(SRC / "colab_notebooks" / "notebook1_data_prep" / "data" / "doman_57_merged_benchmark.csv")
enriched = pd.read_csv(SRC / "doman_57_enriched.csv")

main = base[['rt_name', 'sequence', 'active', 'pe_efficiency_pct', 'rt_family', 'protein_length_aa']].copy()
main = main.sort_values(['rt_family', 'rt_name']).reset_index(drop=True)
main.to_csv(DST / "rt_sequences.csv", index=False)
print(f"rt_sequences.csv: {main.shape} — sequences + labels + family")

# ============================================================
# 2. Hand-crafted biophysical features (70 features)
# ============================================================
meta_cols = ['rt_name', 'active', 'pe_efficiency_pct', 'rt_family', 'protein_length_aa']
feature_cols = [c for c in enriched.columns if c not in meta_cols]

features = enriched[['rt_name'] + feature_cols].copy()
features = features.sort_values('rt_name').reset_index(drop=True)
features.to_csv(DST / "handcrafted_features.csv", index=False)
print(f"handcrafted_features.csv: {features.shape} — 70 biophysical features")

# ============================================================
# 3. ESM-2 embeddings (1280-dim mean-pooled)
# ============================================================
emb_data = np.load(SRC / "khan_buff_doman_embeddings.npz", allow_pickle=True)
all_names = list(emb_data['names'])
all_embs = emb_data['embeddings']

emb_rows = []
for _, row in main.iterrows():
    idx = all_names.index(f"Doman_{row['rt_name']}")
    emb_rows.append(all_embs[idx])

emb_matrix = np.array(emb_rows)
np.savez_compressed(DST / "esm2_embeddings.npz",
                    names=main['rt_name'].values,
                    embeddings=emb_matrix)
print(f"esm2_embeddings.npz: {emb_matrix.shape} — ESM-2 1280-dim embeddings")

# ============================================================
# 4. Family split definitions (for reproducible evaluation)
# ============================================================
families = main['rt_family'].values
unique_fams = sorted(set(families))

splits = []
for fam in unique_fams:
    mask = families == fam
    n = mask.sum()
    n_act = main.loc[mask, 'active'].sum()
    n_inact = n - n_act
    names = main.loc[mask, 'rt_name'].tolist()
    splits.append({
        'family': fam,
        'n_samples': n,
        'n_active': int(n_act),
        'n_inactive': int(n_inact),
        'rt_names': '|'.join(names),
    })

pd.DataFrame(splits).to_csv(DST / "family_splits.csv", index=False)
print(f"family_splits.csv: {len(splits)} families")

# ============================================================
# 5. Feature dictionary (what each feature means)
# ============================================================
feature_dict = {
    # Perplexity (ESM-IF inverse folding)
    'perplexity': 'ESM-IF inverse folding perplexity — how "surprised" the structure-conditioned language model is by the sequence. Lower = sequence fits structure better.',
    'avg_log_likelihood': 'ESM-IF mean log-likelihood per residue. Higher (less negative) = better sequence-structure compatibility.',

    # ProtParam (physicochemical)
    'molecular_weight': 'Total molecular weight in Daltons.',
    'aromaticity': 'Fraction of aromatic residues (Phe, Trp, Tyr).',
    'instability_index': 'Guruprasad instability index. >40 = predicted unstable in vitro.',
    'isoelectric_point': 'Theoretical pI — pH at which net charge is zero.',
    'helix': 'Fraction of residues in alpha-helix (DSSP secondary structure).',
    'turn': 'Fraction of residues in turns.',
    'sheet': 'Fraction of residues in beta-sheet/strand.',
    'gravy': 'Grand Average of Hydropathy (Kyte-Doolittle). Positive = hydrophobic, negative = hydrophilic.',

    # Net charge
    'arginine_r_charge': 'Count of positively charged Arg residues.',
    'lysine_k_charge': 'Count of positively charged Lys residues.',
    'aspartate_d_charge': 'Count of negatively charged Asp residues.',
    'glutamate_e_charge': 'Count of negatively charged Glu residues.',
    'native_net_charge': 'Net charge at pH 7.0 (positive charges minus negative charges).',

    # Thermostability
    't40_raw': 'Predicted fraction of residues retaining structure at 40C.',
    't45_raw': 'Predicted fraction retaining structure at 45C.',
    't50_raw': 'Predicted fraction retaining structure at 50C.',
    't55_raw': 'Predicted fraction retaining structure at 55C.',
    't60_raw': 'Predicted fraction retaining structure at 60C.',
    't65_raw': 'Predicted fraction retaining structure at 65C.',
    't70_raw': 'Predicted fraction retaining structure at 70C.',
    't75_raw': 'Predicted fraction retaining structure at 75C.',
    't80_raw': 'Predicted fraction retaining structure at 80C.',
    'thermophilicity_num': 'Predicted thermophilicity class: 0=mesophilic, 1=undetermined, 2=thermophilic, 3=hyperthermophilic.',
    'clash_binary': 'Whether the predicted structure has steric clashes (1=yes, 0=no).',

    # Aspartate triad (catalytic residues)
    'triad_found_bin': 'Whether the conserved catalytic Asp triad (YXDD motif) was found (1=yes, 0=no).',
    'D1_D2_dist': 'Distance (Angstroms) between 1st and 2nd catalytic Asp residues.',
    'D2_D3_dist': 'Distance (Angstroms) between 2nd and 3rd catalytic Asp residues.',
    'triad_best_rmsd': 'RMSD (Angstroms) of the best-fit Asp triad geometry vs reference HIV-RT. Lower = more similar to known active RT.',

    # Contacts (structural interactions)
    'n_hydrophobic_contacts': 'Total hydrophobic contacts in structure.',
    'n_salt_bridges': 'Total salt bridges.',
    'n_hbonds': 'Total hydrogen bonds.',
    'hydrophobic_per_res': 'Hydrophobic contacts per residue.',
    'salt_per_res': 'Salt bridges per residue.',
    'hbonds_per_res': 'Hydrogen bonds per residue.',
    'pocket_hydrophobic': 'Hydrophobic contacts in the active-site pocket.',
    'pocket_salt_bridges': 'Salt bridges in the active-site pocket.',
    'pocket_hbonds': 'Hydrogen bonds in the active-site pocket.',
    'pocket_hydrophobic_per_res': 'Pocket hydrophobic contacts per pocket residue.',
    'pocket_hbonds_per_res': 'Pocket H-bonds per pocket residue.',

    # Hydrophobicity
    'whole_hydrophobic_fraction': 'Fraction of hydrophobic residues across entire protein.',
    'whole_mean_hydrophobicity': 'Mean Kyte-Doolittle hydrophobicity across all residues.',
    'whole_std_hydrophobicity': 'Standard deviation of hydrophobicity — measures hydrophobic patterning.',
    'yxdd_hydrophobic_fraction': 'Fraction of hydrophobic residues near the YXDD catalytic motif.',
    'yxdd_mean_hydrophobicity': 'Mean hydrophobicity near the YXDD motif.',

    # Hairpin / secondary structure geometry
    'hairpin_pass': 'Whether a beta-hairpin near the active site was found (1=yes, 0=no).',
    'best_confidence': 'Confidence score of the best beta-hairpin detection.',
    'best_s1_len': 'Length of strand 1 in the best beta-hairpin.',
    'best_turn_len': 'Length of the turn connecting the two hairpin strands.',
    'best_s2_len': 'Length of strand 2 in the best beta-hairpin.',
    'n_strands_total': 'Total number of beta-strands in the structure.',
    'n_hairpins_found': 'Number of beta-hairpins detected.',
    'pct_E': 'Percentage of residues in extended/beta-strand conformation (DSSP E).',
    'pct_H': 'Percentage of residues in helical conformation (DSSP H).',

    # Thumb domain
    'thumb_total_residues': 'Number of residues in the thumb subdomain (aligned to HIV-RT thumb).',
    'thumb_surface_residues': 'Surface-exposed residues in the thumb domain.',
    'thumb_surface_positive': 'Positively charged surface residues on thumb.',
    'thumb_surface_negative': 'Negatively charged surface residues on thumb.',
    'thumb_surface_net_charge': 'Net charge of the thumb domain surface.',
    'thumb_charge_ratio': 'Ratio of positive to negative surface charges on thumb.',
    'thumb_fident': 'Sequence identity (fraction) of thumb domain alignment to HIV-RT.',
    'thumb_charge_class_num': 'Thumb charge class: -1=negative, 0=neutral, 1=positive.',

    # DGR motif
    'has_dgr_motif': 'Whether a Diversity-Generating Retroelement (DGR) motif was found (1=yes, 0=no).',

    # Procheck (structural quality)
    'ramachandran_favoured': 'Percentage of residues in Ramachandran-favoured regions.',
    'ramachandran_allowed': 'Percentage in Ramachandran-allowed regions.',
    'ramachandran_outliers': 'Percentage Ramachandran outliers (higher = worse structure quality).',
    'g_factor_dihedrals': 'Procheck G-factor for dihedral angles (higher = better quality).',
    'g_factor_covalent': 'Procheck G-factor for covalent geometry.',
    'g_factor_overall': 'Procheck overall G-factor. Values < -0.5 indicate problematic structures.',
}

fd_df = pd.DataFrame([
    {'feature': k, 'description': v}
    for k, v in feature_dict.items()
])
fd_df.to_csv(DST / "feature_dictionary.csv", index=False)
print(f"feature_dictionary.csv: {len(fd_df)} feature descriptions")

print("\nDone! All data exported to", DST)
