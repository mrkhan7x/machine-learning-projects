# PyTorch Deep Learning Projects

Production neural network models built with PyTorch from scratch — no high-level wrappers, raw `nn.Module` implementation with full training loops.

## Projects

### 01 · Diabetes Binary Classifier
- **Type:** Binary Classification
- **Dataset:** PIMA Indians Diabetes (768 samples, 8 features)
- **Architecture:** `8 → 32 → 32 → 1` | `BCEWithLogitsLoss` | Adam
- **Result:** ~77% test accuracy
- **Key Concepts:** StandardScaler, GPU device setup, `torch.inference_mode()`

### 02 · Sonar Rock vs Mine Classifier
- **Type:** Binary Classification
- **Dataset:** UCI Sonar (208 samples, 60 features)
- **Architecture:** `60 → 128 → 64 → 1` | `BCEWithLogitsLoss` | Adam
- **Result:** **95.24% test accuracy**
- **Key Concepts:** Wider architecture for higher-dimensional input, `F.relu()`

### 03 · Gold Price Predictor
- **Type:** Regression
- **Dataset:** Historical Gold Prices (SPX, GLD, USO, SLV, EUR/USD features)
- **Architecture:** `MSELoss` | Adam | Saved model weights (`.pth`)
- **Result:** **R² Score: 0.9313**
- **Key Concepts:** Regression output (no activation), `r2_score`, `torch.save()`

## Tech Stack

- PyTorch (`nn.Module`, `BCEWithLogitsLoss`, `MSELoss`, `Adam`)
- scikit-learn (`StandardScaler`, `train_test_split`, `r2_score`)
- Pandas, NumPy, Matplotlib

## Skeleton Used

Every project follows the same 6-step PyTorch workflow:

```
1. Data     → Load → Split → Scale → Tensors
2. Model    → Subclass nn.Module
3. Loss & Optimizer
4. Training Loop
5. Evaluation (torch.inference_mode)
6. Save model weights
```
