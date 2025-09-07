![NeonCortex demo](project_video_hero.png)

# NeonCortex — Motor-Imagery EEG Classifier
Turn brain waves into intent: predict Left/Right/Foot/Tongue from EEG.

## Run locally
python -m streamlit run main.py

NeonCortex predicts Left/Right/Foot/Tongue intent from motor-imagery EEG. Using Kaggle’s BCI IV 2a, I ran EDA, engineered epoch-level stats (mean/std/ptp), compared LogReg/Linear SVM/Random Forest/Gradient Boosting with train/test + 5-fold CV, and shipped the best model as joblib (model + feature/label metadata). For visibility, a Streamlit demo lets users pick EEG feature ranges and view results on a single-color neon-blue brain-scan screen. (Research/demo only; not for clinical use.)
