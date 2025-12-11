# Model Tuning & Optimization Log

## 2025-12-10: Phase 5 Optimization Implementation

### 1. Training Loop Enhancements (`03_train_injury_ner.py`)

**Changes:**

- **Early Stopping**: Integrated `EarlyStoppingCallback` with `patience=3`.
  - *Reason*: To prevent overfitting on the small dataset. The model was previously training for fixed epochs, potentially degrading validation performance after a certain point.
- **Best Model Checkpointing**: Enabled `load_best_model_at_end=True` and set `metric_for_best_model="f1"`.
  - *Reason*: Ensures that the final saved model is the one that performed best on the validation set, not just the one from the last epoch.
- **Extended Training**: Increased `num_train_epochs` from 3 to 10.
  - *Reason*: With early stopping in place, we can safely allow for more epochs to find the optimal convergence point without fear of overfitting.

### 2. Active Learning Pipeline (`04_active_learning.py`)

**Changes:**

- **Uncertainty Sampling**: Implemented "Least Confidence" scoring.
  - *Reason*: To prioritize labeling examples where the model is most unsure (lowest max probability).
- **Conflict Detection**: Added logic to flag examples where keyword extraction (heuristic) disagrees with model prediction.
  - *Reason*: These "disagreement" cases are often the most valuable for correcting model misconceptions or fixing bad heuristics.
- **Heuristic Filtering**: Added reporter blacklist filtering.
  - *Reason*: To automatically exclude noise (reporter bylines) from the training pool.

### 3. Error Analysis (`05_error_analysis.py`)

**Changes:**

- **Test Set Evaluation**: Created a dedicated script for evaluating on the held-out test set.
- **Error Export**: Dumps specific false positives/negatives to CSV.
  - *Reason*: To allow for manual inspection of failure modes (e.g., are we missing specific injury types? confusing players with teams?).

## Next Steps for Tuning

- **Hyperparameter Search**: Once the active learning loop provides more data, run a grid search on learning rate (`1e-5`, `2e-5`, `5e-5`) and dropout (`0.1`, `0.2`).
- **Regularization**: If overfitting persists despite early stopping, increase weight decay.

## 2025-12-10: Active Learning Cycle 1

### 1. Data Augmentation

- **Candidates Generated**: 50 examples selected via `04_active_learning.py`.
- **Selection Strategy**: Combined Least Confidence + Keyword Conflict.
- **Manual Validation**: User reviewed and corrected candidates using `02_validate_ner_data.py`.
- **Result**: Added **35** high-value examples to `gold_standard.jsonl`.
- **Total Gold Standard Size**: 56 examples.

### 2. Retraining Plan (V2)

- **Objective**: Incorporate the 35 hard examples to improve decision boundary.
- **Configuration**:
  - Use existing `03_train_injury_ner.py` (with Early Stopping enabled).
  - Compare metrics against Baseline (F1: 69.1%).

### 3. V2 Results (Post-Active Learning)

- **Metrics**:
  - **Precision**: 0.9541
  - **Recall**: 0.9455
  - **F1 Score**: 0.9498 (+25.9% improvement)
  - **Accuracy**: 0.9706
- **Analysis**:
  - The addition of 35 "hard" examples (conflicts/uncertainty) significantly corrected the model's decision boundary.
  - Early stopping prevented overfitting, allowing the model to generalize much better to the test set.
  - **Conclusion**: The Active Learning loop is highly effective.

## Next Steps

- **Demo**: Build Streamlit app to showcase the high-performance model.
- **Documentation**: Finalize README.
