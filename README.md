# Sustainability Vision: Waste Sorting (CS171 Final Project)

## Authors
- Soham Bhowmick (GitHub: @Soham0047)
- Lwin Moe (GitHub: @lwinmoelovescoding)

## Description of Question and Research Topic

We were fascinated by the image classification from our Homework 4 bird project and wanted to test a simlar approach for waste tracking and sorting.
Our goal is to study how well course-level image classifiers can identify common waste types from photos.
We believe that this can be a step toward future applications that could automatically classify trash to help people dispose of it correctly.
<br><br>
For this project, we will compare simple Convolutional Neural Networks (CNNs) with classical baselines such as KNN, Decision Tree, and Logistic Regression using color and texture features. 
We will also test a small ensemble model through majority voting. 
Our datasets combine TrashNet (studio images, 6 classes) and a small classification subset from TACO (litter “in the wild”) to examine how domain shift affects performance.

All techniques used will come from our class which includes proper train/validation/test splits, data augmentation, cross-entropy training, Batch Normalization, Dropout, and learning-rate scheduling. 
We will evaluate performance using learning curves, confusion matrices, and brief error analyses.
(Sources: TrashNet; TACO.)

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
  - Classical: KNN (k ∈ {3,5,7}), Decision Tree/Logistic on color/texture features.  
  - CNN: 2–3 conv blocks (Conv -> BN -> ReLU ->Pool), then GAP + Linear(…, 6); light augmentation (flip, ±10 degrees).  
- **Lwin Moe (TACO models):**  
  - Classical: KNN/Logistic with same features; compare drop from clean (TrashNet) to in-the-wild (TACO).  
  - CNN: same backbone as A; compare augmentation levels and early-stopping patience.  

## Project Timeline
- **Week 1:** Set up repo; download data; write data prep notebooks; push splits & stats. Initial data collections and prep must be done by Nov 23, 2025 (By Nov 23, 2025) 
- **Week 2:** implement classical baselines + feature extraction; first CNN baseline; log curves. (By Nov 28, 2025) 
- **Week 3:** run ablations; add majority vote; finalize plots and tables.  (Dec 3, 2025)
- **Week 4:** polish analysis notebook & slides; finalize README, LICENSE, and reproducibility steps. (Dec 8, 2025)