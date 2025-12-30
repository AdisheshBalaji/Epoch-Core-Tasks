# Emotion Classification with SVMs

This project applies **semi-supervised learning** using **Principal Component Analysis (PCA)** and **Support Vector Classification (SVC)** to classify facial images into different emotions. The workflow leverages both labeled and unlabeled datasets to improve model performance.

## Dataset
- **Labeled Dataset**: Emotion-tagged images in CSV format.
- **Unlabeled Dataset**: Raw image files used for semi-supervised learning.
- **Image Dimensions**: 48×48 pixels (2304-dimensional vectors).

## Data Preparation
1. **Unlabeled Data**:
   - Resized to 48×48 pixels and converted to grayscale.
   - Flattened into 2304-dimensional vectors.
   - Stored in a matrix `X_unlabelled`.

2. **Labeled Data**:
   - Dropped underrepresented emotions to address class imbalance.
   - Balanced the dataset using **SMOTEENN** (oversampling + undersampling).
   - Converted pixel strings to NumPy arrays for processing.
   - Attempted Data Augmentation which turned out to be aggressive and reduced model accuracy

## Dimensionality Reduction
- Applied **PCA** to reduce dimensionality while retaining 95% of the variance.
- Separate PCA models were trained for:
  - Labeled data (`pca_labelled`).
  - Combined labeled and pseudo-labeled data (`pca_combined`).

## Model Training
1. **Initial SVM Training**:
   - Trained on PCA-transformed labeled data.
   - Hyperparameters tuned using **GridSearchCV** with 5-fold cross-validation.
   - Addressed class imbalance using class weights.

2. **Iterative Pseudo-Labeling**:
   - Predicted labels for batches of 100 unlabeled images.
   - Added high-confidence pseudo-labeled data to the training set.
   - Repeated until 1000 pseudo-labeled samples were added.

3. **Final SVM Training**:
   - Trained on PCA-transformed combined dataset (labeled + pseudo-labeled).
   - Evaluated on the original test set.

## Results
- **Final Training Accuracy**: Achieved on the expanded training set.
- **Final Test Accuracy**: Evaluated on the original test set.
- Confusion matrices and explained variance plots were generated for analysis.

## Key Observations
- PCA significantly reduced dimensionality, improving computational efficiency.
- Data Augmentation on small labelled datasets proved to be counter productive
- Pseudo-labeling effectively leveraged the unlabeled dataset to enhance performance.
- Training PCA on labelled+unlabelled data gave a better set of feature vectors and hence improved final SVM's accuracy

## How to Run 
- pip install -r requirements.txt
