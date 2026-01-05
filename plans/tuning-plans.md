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

### 4. Phase 6 Results: Baseline Comparison

#### **Experiment A: Domain Adaptation Value**

| Model | Precision | Recall | F1 Score | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| `bert-base-uncased` | 96.15% | 97.83% | 96.98% | 98.69% |
| `microsoft/SportsBERT` | **97.37%** | 96.94% | **97.16%** | **98.77%** |

**Analysis**:

- Both models perform exceptionally well, likely due to the high quality of the "Gold" data added in Phase 5.
- `SportsBERT` shows a slight edge in Precision (+1.2%) and F1 (+0.18%), confirming that domain pre-training offers a marginal but positive benefit for this specific task.
- The high baseline performance (>96% F1) suggests that further hyperparameter tuning might yield diminishing returns, and focus should shift to **data expansion** (covering more diverse injury types) rather than model architecture.

#### **Experiment B: Hyperparameter Sweep**

| Run | LR | Batch | Precision | Recall | F1 Score | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Run 3 | 2e-5 | 16 | 96.92% | 96.07% | 96.49% | 98.04% |
| Run 4 | 3e-5 | 16 | 95.67% | 96.51% | 96.09% | 98.04% |
| Run 5 | 5e-5 | 32 | 95.69% | 96.94% | 96.31% | 98.28% |
| **Baseline** | **5e-5** | **16** | **97.37%** | **96.94%** | **97.16%** | **98.77%** |

**Analysis**:

- **Baseline Remains Superior**: None of the new hyperparameter combinations outperformed the baseline configuration (LR=5e-5, Batch=16).
- **Lower LR Degradation**: Reducing the learning rate to `2e-5` or `3e-5` slightly degraded performance, suggesting the model benefits from the more aggressive updates of `5e-5` given the small dataset size.
- **Batch Size Impact**: Increasing batch size to 32 (Run 5) did not improve stability or performance compared to Batch 16.

**Conclusion**:

- We will stick with the **Baseline Configuration** (LR=5e-5, Batch=16) for production.
- Further tuning of these specific parameters is unlikely to yield significant gains.
- Future efforts should focus on **Data Expansion** (Active Learning Cycle 2) or **Synthetic Data Generation**.

## Next Steps

- **Documentation**: Finalize README.
