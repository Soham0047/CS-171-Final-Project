# Sustainability Vision: Waste Sorting (CS171 Final Project)

## Authors
- Soham Bhowmick (GitHub: @Soham0047)
- Lwin Moe (GitHub: @lwinmoelovescoding)

## Description of Question and Research Topic
We study how well course-level image classifiers can sort common waste types from photos.
We’ll compare simple CNNs to classical baselines (KNN / Decision Tree / Logistic on color/texture features) and test a small ensemble via majority vote.
Our datasets combine **TrashNet** (studio images, 6 classes) and a small classification subset from **TACO** (litter “in the wild”) to explore domain shift.
We will use only techniques from our class (data splits, augmentation, cross-entropy training, BatchNorm/Dropout, learning-rate scheduling).
We evaluate with clear train/val/test splits, learning curves, confusion matrices, and short error analysis.
*(Sources: TrashNet; TACO.)*

## Project Outline / Plan
1. **Data setup:** download TrashNet + select TACO categories; resize to a fixed square; compute train mean/std; create stratified train/val/test splits.  
2. **Classical baselines:** extract RGB histograms + simple gradients; train KNN, Decision Tree / Logistic; record metrics.  
3. **Neural models:** build 2–3 block CNNs (Conv→ReLU→MaxPool, with BatchNorm and optional Dropout); train with cross-entropy & Adam.  
4. **Ablations:** augmentation strength (flip/rotation/RandomResizedCrop) and training-set size (e.g., 25/50/100%).  
5. **Ensemble:** majority vote over best classical + CNN; compare to best single model.  
6. **Analysis:** curves, per-class precision/recall, confusion matrices; highlight domain-shift errors (studio vs in-the-wild).

## Data Collection Plan
- **Soham Bhowmick (TrashNet lead):**  
  - Download **TrashNet** (6 classes), verify license/readme; organize images by class.  
  - Preprocess to 200×200, compute dataset mean/std, create stratified splits and a CSV with per-class counts.  
- **Lwin Moe(TACO lead):**  
  - Use **TACO** (waste “in the wild”); pick a small set of categories that roughly map to TrashNet labels.  
  - Download images/annotations per instructions; export a classification subset, preprocess to 200×200, compute stats and splits.  

## Model Plans
- **Soham Bhowmick (TrashNet models):**  
  - Classical: KNN (k∈{3,5,7}), Decision Tree/Logistic on color/texture features.  
  - CNN: 2–3 conv blocks (Conv→BN→ReLU→Pool), then GAP + Linear(…, 6); light augmentation (flip, ±10°).  
- **Lwin Moe (TACO models):**  
  - Classical: KNN/Logistic with same features; compare drop from clean (TrashNet) to in-the-wild (TACO).  
  - CNN: same backbone as A; compare augmentation levels and early-stopping patience.  

## Project Timeline
- **Week 1:** set up repo; download data; write data prep notebooks; push splits & stats.  
- **Week 2:** implement classical baselines + feature extraction; first CNN baseline; log curves.  
- **Week 3:** run ablations; add majority vote; finalize plots and tables.  
- **Week 4:** polish analysis notebook & slides; finalize README, LICENSE, and reproducibility steps.