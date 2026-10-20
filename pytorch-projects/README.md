# PyTorch projects — notebook-first learning

Four self-contained, output-free walkthroughs build from data inspection to honest
baseline comparisons and a final holdout. Start with each folder's README, install
its requirements into an isolated environment, then run `learning_walkthrough.ipynb`
top to bottom. Paths work from the repository root or the notebook folder.

| Project | Learning task | Baselines vs PyTorch | Default CPU budget |
| --- | --- | --- | --- |
| [01 Diabetes](01_diabetes_classifier/README.md) | Binary recorded outcome; sentinel cleaning and train-only imputation | Majority + logistic regression vs 16-unit MLP | 100 full-batch epochs |
| [02 Sonar](02_sonar_classifier/README.md) | Rock/mine classification; 60 energy bands and small-sample uncertainty | Majority + logistic regression vs 16-unit MLP | 100 full-batch epochs |
| [03 Gold](03_gold_price_predictor/README.md) | Chronological **same-day estimation**, not future forecasting | Training mean + Ridge + rolling persistence vs 16-unit MLP | 150 full-batch epochs |
| [04 MNIST](04_mnist_digit_classifier/README.md) | Ten-class digits; official training-only validation split | Nearest centroid vs 128-unit MLP | 3 mini-batch epochs, 12,000 training images |

## The learning path

1. Read incremental markdown before each code cell: provenance, schema and local
   loading come before any modeling.
2. Validate input and declare cleaning rules. Reserve validation/test before EDA
   decisions; plot training data, not the held-out answer sheet.
3. Fit imputers/scalers and all learned preprocessing on training only. Use the
   same partitions for the baselines and neural model.
4. Train with explicit losses, correct tensor shapes, seeded NumPy/torch, and CPU
   defaults. Plot post-update train/validation learning curves.
5. Select the in-memory checkpoint by validation loss; freeze decisions, evaluate
   final test once, and reuse cached predictions for metrics/error plots.
6. Discuss uncertainty, leakage risks, population limits and what a new experiment
   would require. Record real results from your run: there are no promised scores.

## Data and environment

Each project has a README with schema, public source links, setup commands and
failure instructions, plus requirements matching its imports. Use Python 3.10+
and that environment's Jupyter kernel. Dependency ranges are not exact lockfiles;
record versions and source hashes/dates for reproducibility. Seeds help on the
same stack but do not guarantee identical results across hardware or versions.

The three tabular projects require manually obtained local public CSV files and
do not download automatically. MNIST uses official torchvision data with an
explicit `DOWNLOAD_DATA` flag, default **False**. Data is not bundled. No private
course files are copied into these projects.

The original `*_pytorch.ipynb` notebooks and any historical scripts/artifacts are
preserved as reference experiments. The new walkthroughs do not load their saved
models, untrusted weights or pickles, or overwrite old artifacts. Only MNIST's
opt-in official raw-data cache can be written. Historical notebook scores are not
claims about these new validation-separated experiments.
