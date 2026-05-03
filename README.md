# 🫁 Classification des Sons Respiratoires — ICBHI 2017

> Reproduction et amélioration de l'article **"Geometry-Aware Optimization for Respiratory Sound Classification: Enhancing Sensitivity with SAM-Optimized Audio Spectrogram Transformers"** ([arXiv:2512.22564](https://arxiv.org/abs/2512.22564))

---

##  Table des matières

- [Contexte](#-contexte)
- [Dataset](#-dataset-icbhi-2017)
- [Architecture](#-architecture)
- [Résultats — Table d'ablation](#-résultats--table-dablation)
- [Recall par classe](#-recall-par-classe)
- [Utilisation](#-utilisation)
- [Améliorations implémentées](#-améliorations-implémentées)
- [Analyse des erreurs](#-analyse-des-erreurs)

---

##  Contexte

La classification automatique des sons respiratoires est un enjeu majeur pour le diagnostic précoce des maladies pulmonaires (asthme, BPCO, pneumonie). Ce projet vise à **dépasser le recall du papier de référence (68.31%)**, métrique prioritaire en contexte médical car les faux négatifs — une pathologie non détectée — sont cliniquement dangereux.

**Objectif principal :** Recall macro > 68.31% sur le test set ICBHI 2017.

---

##  Dataset ICBHI 2017

| Classe | Cycles | % | Description |
|--------|--------|---|-------------|
| Normal | 3 642 | 52.8% | Sons respiratoires normaux |
| Crackle | 1 864 | 27.0% | Sons discontinus brefs (<15ms, ~650Hz) |
| Wheeze | 886 | 12.8% | Sons continus (>80ms, ~400Hz) |
| Both | 506 | 7.3% | Crackle + Wheeze simultanés |

**Défis principaux :**
- Déséquilibre sévère : Both est **6.8× moins représenté** que Normal
- Enregistrements en conditions cliniques réelles (bruit ambiant variable)
- 4 équipements hétérogènes : AKGC417L, LittC2SE, Litt3200, Meditron
- Les cycles anormaux contiennent souvent de longues plages de respiration normale → biais vers Normal

**Download :** [ICBHI 2017 Challenge Dataset](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge)

---

##  Architecture

```
Audio .wav
    ↓
ASTFeatureExtractor (MIT/ast-finetuned-audioset-10-10-0.4593)
    ↓  features shape: (128000,) → mel-spectrogram patches
CustomAST (Audio Spectrogram Transformer)
    ↓  Transformer encoder — patches 2D comme tokens
Classifieur linéaire 4 classes
    ↓
[Normal, Crackle, Wheeze, Both]
```

**Composants clés :**

| Composant | Détail |
|-----------|--------|
| Backbone | AST pré-entraîné AudioSet (MIT) |
| Optimiseur | SAM (Sharpness-Aware Minimization) + AdamW |
| Learning rate | 1e-5 (fine-tuning transformer) |
| Batch size | 8 |
| Early stopping | patience = 4–5 epochs |
| Critère de sauvegarde | Meilleur score ICBHI sur CV (pas la CV loss) |

**Pourquoi SAM ?** Effectue une double passe à chaque itération pour trouver des minima plats → meilleure généralisation sur les petits datasets médicaux.

---

##  Résultats — Table d'ablation

| Configuration | Recall macro | Spécificité | ICBHI Score | Δ Recall vs base |
|---------------|-------------|-------------|-------------|-----------------|
| NB0 — AST+SAM (reproduction) | 45.22% | 83.18% | 64.20% | — |
| NB1 — + WeightedSampler + Seuils | 48.30% | 83.43% | 65.87% | +3.08% |
| **NB2 — + Focal Loss γ=2.0** | **50.28%** | **82.91%** | **66.60%** | **+5.06%** |
| TTA N=5 (testé sur NB2) | 46.38% | 82.13% | 64.25% |  abandonné |
| NB4 — + Augmentation en ligne γ=1.5 | 48.45% | 82.64% | 65.55% | +3.23% |
| NB5 — + Patient CV + Aug offline | 47.10% | 83.16% | 65.13% | +1.88% |
| **Papier de référence (cible)** | **68.31%** | **67.89%** | **68.10%** | cible |

> **Meilleure configuration : NB2** — WeightedRandomSampler + Focal Loss γ=2.0 + Tuning de seuils CV

Notre modèle surpasse le papier en **spécificité (+15 pts)** et est proche en ICBHI (-1.5 pts).

---

##  Recall par classe

| Classe | NB0 | NB1 | NB2 ⭐ | NB4 | NB5 | Paper |
|--------|-----|-----|--------|-----|-----|-------|
| Normal | 81.6% | 73.7% | 67.3% | 65.4% | 70.4% | 68.31% (macro) |
| Crackle | 41.8% | 48.2% | 52.0% | 49.5% | 40.2% | 68.31% (macro) |
| Wheeze | 48.6% | 50.6% | 42.2% | 47.0% | 71.9% | 68.31% (macro) |
| Both | 8.9% | 20.7% | 39.6% | 32.0% | 5.9% | 68.31% (macro) |

**Observation clé :** la Focal Loss booste fortement Both (+30.7 pts vs NB0) mais crée un trade-off avec Wheeze. γ=1.5 corrige partiellement ce déséquilibre.

---


> **Note Kaggle :** tous les notebooks sont configurés pour Kaggle GPU (T4/P100). Les chemins `/kaggle/working/` et `/kaggle/input/` sont utilisés. Adapter `NPZ_PATH` et `CHECKPOINT_DIR` si exécution en local.

---

##  Utilisation

### Entraînement (configuration NB2 — meilleure)

```python
# Dans NB2_focal_loss.ipynb :
# - WeightedRandomSampler avec poids inverses de fréquence
# - Focal Loss γ=2.0 avec alpha normalisé par classe
# - Early stopping sur ICBHI score CV (pas la CV loss)
# - Sauvegarde du meilleur modèle par ICBHI, pas par CV loss

FOCAL_GAMMA = 2.0
PATIENCE    = 4
EPOCHS      = 20
LR          = 1e-5
```


---

##  Améliorations implémentées

###  WeightedRandomSampler
Surreprésentation des classes minoritaires à chaque batch. Poids = 1/class_count.
```
Normal: 0.00044  Crackle: 0.00079  Wheeze: 0.00157  Both: 0.00297
```

###  Focal Loss (γ=2.0)
Réduit la contribution des exemples faciles (Normal prédit avec 95% de confiance).
```
L_focal = (1 - pt)^γ × CE(logits, targets)
```
Combiné au WeightedSampler : le sampler contrôle *qui* le modèle voit, la Focal Loss contrôle *comment* chaque sample contribue au gradient.

###  Tuning de seuils par descente de coordonnées
Optimise des seuils par classe sur le CV set pour maximiser l'ICBHI.
```
prediction = argmax(probs / thresholds)
# thresholds optimaux NB2 : [0.59, 0.65, 0.45, 0.15]
# Both threshold = 0.15 → amplification ×3.3 du score Both
```

###  Patient-level CV split (NB5)
Corrige le patient leakage du split par cycle. Gap CV/test réduit de 18 pts → 1.8 pts.

###  Data Augmentation offline sur waveforms (NB5)
Copies augmentées pré-calculées avant entraînement (time shift, bruit gaussien, scaling).
```
AUG_PROB = {Normal: 0%, Crackle: 50%, Wheeze: 70%, Both: 80%}
```

###  TTA — Test-Time Augmentation (abandonné)
Contre-productif sur features 1D brutes (-3.91% recall, Wheeze -16.1%). Les opérations 1D détruisent les patterns diagnostiques fréquentiels. Valide uniquement sur spectrogrammes 2D.

---

##  Analyse des erreurs (NB2 — meilleure config)

**Confusions principales :**

| Classe réelle | Erreurs | Confusion principale | Cause |
|---------------|---------|---------------------|-------|
| Normal | 438/1340 (32.7%) | → Crackle  | Sons normaux avec segments brefs |
| Crackle | 288/600 (48.0%) | → Normal  | Crackles noyés dans du son normal |
| Wheeze | 144/249 (57.8%) | → Normal  | Wheezes doux |
| Both | 102/169 (60.4%) | → Wheeze  | Composante crackle ignorée |

**Réduction des erreurs à haute confiance :** 442 cas (NB0) → 48 cas (NB2) — **réduction de 89%**

**Taux d'erreur par device :**

| Device | Samples test | Taux d'erreur |
|--------|-------------|---------------|
| AKGC417L | 1234 (52%) | 46.5% |
| Meditron | 807 (34%) | 35.3% |
| Litt3200 | 174 (7%) | 47.1% |
| LittC2SE | 143 (6%) | 21.7% |

---
