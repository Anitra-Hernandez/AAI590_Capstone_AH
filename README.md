# Architecture Review Board (ARB) Model Evaluation and Decision Support System

This project is a part of the AAI-590 Capstone Project course in the Applied Artificial Intelligence Program at the University of San Diego.

## AAI590 Capstone Project

This project builds a machine learning system that evaluates Enterprise Arhitecture models submitted to an Archtecture Review Board and predicts outcomes based on structural and metadata features and provides explainability insights.

---

## Project Overview

Architecture Review Boards evaluate achitecture diagrams, documentation, and model details to make decisions regarding the feasibility and adherence to standards of proposed enterprise architecture models and initiatives, but this can be a time-consuming process. This project develops and evaluates machine learning models to predict ARB outcomes (Approve, Needs Work, Reject) for enterprise architecture models identifying patterns in high-quality versus low-quality enterprise architecture models. The system analyzes model characteristics such as view counts, element counts, relationships, and metadata to provide predictions and insights into key factors influencing review decisions as part of a preliminary review process and adds explainability. It provides a UI metadata can be input and predictions generated in real-time. 

### Key Features

This project creates a machine learning pipelines that consists of the following key features:

- **Multi-Model Comparison**: Logistic Regression, Random Forest, XGBoost, MLP Neural Network, and Hybrid ensembles
- **Hyperparameter Optimization**: Optuna-based tuning for optimal model performance
- **Explainability**: SHAP analysis and feature importance visualization
- **Interactive Web Application**: Streamlit-based app for real-time predictions
- **Comprehensive Analysis**: EDA, preprocessing, model evaluation, and results visualization

---

## Project Structure

```
AAI590_Capstone_AH/
├── Data/
│   └── ea_modelset/                   
│       └── eamodelset/
│           └── dataset/
│               ├── dataset.json
│               ├── preprocessed_models.csv
│               └── processed-models/
├── Models/
│   ├── baseline_models/    
│       ├── logistic_regression.pkl
│       ├── xgboost.pkl
│       ├── mlp_baseline.pkl
│       ├── random_forest.pkl
│   └── tuned_and_hybrid_models/       
│       ├── rf_tuned.pkl
│       ├── xgboost_tuned.pkl
│       ├── mlp_tuned.pkl
│       ├── hybrid_stacking.pkl
│       ├── hybrid_softvoting.pkl
│       └── model_metrics.csv
├── Notebooks/
│   ├── 01_eda_ea_modelset.ipynb              
│   ├── 02_preprocessing.ipynb                
│   ├── 03_baseline_model.ipynb              
│   ├── 04_tuning_and_hybrid.ipynb          
│   ├── 05_neural_network.ipynb               
│   ├── 06_model_comparion_and_explainability.ipynb  
│   ├── 07_results_and_images.ipynb         
│   ├── 08_arb_evaluator_app.ipynb           
│   └── arb_evaluator_app.py                  
├── LICENSE
├── README.md
├── requirements.txt
└── arb_evaluator_app.py  
```

---

## Project Notebooks Overview

This project follows a structured workflow across eight Jupyter notebooks, each focusing on a specific phase of the machine learning pipeline:

### 1. Exploratory Data Analysis (`01_eda_ea_modelset.ipynb`)
- Initial dataset exploration and statistics
- Distribution analysis of features and target variable
- Visualization of relationships between features
- Identification of patterns and outliers

### 2. Data Preprocessing (`02_preprocessing.ipynb`)
- Analysis of missing values and outliers
- Feature engineering (ratio calculations, num_formats)
- One-hot encoding for categorical variables (source, language)
- Engineering of arb_outcome as target feature

### 3. Baseline Models (`03_baseline_model.ipynb`)
- Implementation of three baseline models:
  - Logistic Regression
  - Random Forest Classifier
  - XGBoost Classifier
- Initial model evaluation and comparison
- Establishing performance benchmarks

### 4. Hyperparameter Tuning & Hybrid Models (`04_tuning_and_hybrid.ipynb`)
- Optuna-based hyperparameter optimization for:
  - Random Forest
  - XGBoost
- Development of hybrid ensemble models:
  - Soft Voting Classifier
  - Stacking Classifier
- Performance comparison with baseline models

### 5. Neural Network (`05_neural_network.ipynb`)
- Multi-Layer Perceptron (MLP) implementation
- Architecture design and layer configuration
- Hyperparameter tuning with Optuna
- Training with early stopping
- Performance evaluation and comparison

### 6. Model Comparison & Explainability (`06_model_comparion_and_explainability.ipynb`)
- Comprehensive evaluation of all models
- SHAP (SHapley Additive exPlanations) analysis
- Feature importance visualization
- Model interpretability insights
- Confusion matrix analysis
- Final model selection

### 7. Results & Visualizations (`07_results_and_images.ipynb`)
- Model performance comparisons (accuracy, F1-score)
- Confusion matrices for top models
- Feature importance charts
- Outcome distribution analysis
- SHAP beeswarm plots for each class

### 8. ARB Evaluator Application (`08_arb_evaluator_app.ipynb`)
- Streamlit web application development
- Interactive user interface design
- Real-time prediction functionality
- Feature importance visualization
- Model comparison dashboard

---

## Dataset Description

### EA Model Set

This project uses the **EA Model Set** dataset, a comprehensive collection of enterprise architecture models in the Archimate modeling language from various sources including GitHub, GenMyModel, and other repositories. The EA Model Set is publicly available and designed for machine learning research in enterprise architecture quality assessment.

#### Dataset Characteristics
- **Total Models**: 978 enterprise architecture models with 12 features
- **Engineered Target Variable**: ARB Outcome (3 classes)
  - **0 = Approve**: High-quality models meeting standards
  - **1 = Needs Work**: Models requiring improvements
  - **2 = Reject**: Low-quality models not meeting requirements

#### Key Features
The dataset includes both structural and metadata features:

**Structural Features:**
- `viewCount`: Number of architectural views in the model (e.g., logical, physical, deployment)
- `elementCount`: Total number of model elements (components, services, databases, etc.)
- `relationshipCount`: Number of relationships/connections between elements

**Quality Indicators:**
- `hasWarnings`: Boolean flag indicating presence of model warnings
- `hasDuplicates`: Boolean flag for duplicate elements
- `duplicateCount`: Count of duplicate elements (quality indicator)

**Engineered Features:**
- `rel_elem_ratio`: Relationship-to-element ratio (measures interconnectivity)
- `view_elem_ratio`: View-to-element ratio (measures model organization)
- `num_formats`: Number of different formats used in the model (measures complexity)

**Metadata Features:**
- `source`: Model repository source (GitHub, GenMyModel, Other, Unknown) - One-hot encoded
- `language`: Model language (en, es, pt, de, fr, Other) - One-hot encoded
- `name`: Model identifier (not used)
- `id`: Unique model ID (not used)

---

## Getting Started

### Prerequisites

- Python 3.11+
- Conda or pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Anitra-Hernandez/AAI590_Capstone_AH.git
   cd AAI590_Capstone_AH
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Using conda
   conda create -n arb_predictor python=3.9
   conda activate arb_predictor

3. **Install required Python libraries**
   ```bash
   pip install -r Code/requirements.txt
   ```

### Running the Notebooks

Navigate to the `Notebooks/` directory and run Jupyter:

```bash
cd Notebooks
jupyter notebook
```

Run notebooks in sequence:
1. `01_eda_ea_modelset.ipynb` - Explore the dataset
2. `02_preprocessing.ipynb` - Preprocess and prepare data
3. `03_baseline_model.ipynb` - Train baseline models
4. `04_tuning_and_hybrid.ipynb` - Optimize models
5. `05_neural_network.ipynb` - Train neural network
6. `06_model_comparion_and_explainability.ipynb` - Evaluate and explain models
7. `07_results_and_images.ipynb` - Generate visualizations

### Running the Streamlit App

#### Option 1: Access the Deployed App

The application is deployed and accessible online at:

**[ARB Evaluator App (Live)](https://aai590capstoneahgit-arbappevaluatorapp.streamlit.app/)**

---

## Model Performance

### Performance Summary

All models were evaluated on a held-out test set (20% of data) using accuracy and macro F1-score as primary metrics. The results demonstrate exceptional performance across all tuned models, with near-perfect classification capabilities.

| Model | Accuracy | F1-Score | Training Time | Model Type |
|-------|----------|----------|---------------|------------|
| **Tuned Random Forest** | 1.00 | 1.00 | Fast | Tree Ensemble |
| **Hybrid Stacking** | 1.00 | 1.00 | Medium | Ensemble |
| **Tuned XGBoost** | ~0.995 | ~0.995 | Medium | Gradient Boosting |
| **Tuned MLP** | ~0.995 | ~0.995 | Slow | Neural Network |
| Logistic Regression (Baseline) | ~0.836 | ~0.817 | Fast | Linear |

### Performance Highlights

**Best Overall Performance**: Random Forest (Tuned) and Hybrid Stacking both achieved perfect scores (100% accuracy and F1-score)
**Best Speed-Accuracy Trade-off**: Tuned Random Forest offers instant predictions with perfect accuracy
**Most Interpretable**: Random Forest provides clear feature importance without requiring SHAP
**Production Model**: Random Forest (Tuned) selected for the Streamlit application due to:
- Perfect accuracy and F1-score
- Fast inference time
- Built-in feature importance
- Robust performance across all classes

### Confusion Matrix Analysis

All top-performing models (Random Forest, XGBoost, MLP, Hybrid Stacking) demonstrated:
- **Zero false positives** for rejecting good models
- **Zero false negatives** for approving poor models
- **Perfect class separation** across all three outcome categories
- **Consistent performance** across different model architectures

### Cross-Validation Results

5-fold cross-validation confirmed model robustness:
- Random Forest: 99.8% ± 0.2% accuracy
- XGBoost: 99.5% ± 0.3% accuracy
- MLP: 99.3% ± 0.4% accuracy
- No evidence of overfitting across all models

---

## Explainability Summary

This project employs multiple explainability techniques to ensure model transparency and trust.

### SHAP Analysis (XGBoost)
- **SHAP TreeExplainer** computes feature importance for multi-class classification
- **Beeswarm plots** visualize how features impact predictions across all three outcome classes
- **Key Finding**: viewCount, duplpicateCount, and hasWarning are the strongest predictors

### Random Forest Feature Importance
- **Top 5 Features**:
  1. **view_count**: Most critical predictor
  2. **hasDuplicate**: Second most impactful
  3. **duplicateCount**: Data quality indicator
  4. **hasWarning**: Data quality flag
  5. **elementCount**: Structural quality indicator
- Displayed in real-time in the Streamlit app

### MLP Neural Network Analysis
- **Architecture**: 2-layer network (128, 112 neurons) with tanh activation
- **Strengths**: Captures non-linear feature interactions and complex patterns
- **Feature Sensitivity**: Neural networks implicitly weight features through learned connections
- **Performance**: 99.5% accuracy with consistent predictions across validation sets
- **Trade-off**: High accuracy but less interpretable than tree-based models (black-box nature)

### Business Interpretability
**5+ views & 50+ elements** → Likely approved  
**Warnings or duplicates** → Needs work/rejection  
**<20 elements or 0-1 views** → High rejection probability  
**rel_elem_ratio (0.5-2.0)** → Well-structured architecture

---

## Key Findings

### Important Features
- **View Count**: Number of architectural views in the model
- **Element Count**: Total number of model elements
- **Relationship Count**: Number of relationships between elements
- **rel_elem_ratio**: Ratio of relationships to elements
- **Source**: Model repository source (GitHub, GenMyModel, etc.)
- **Language**: Model language
- **Warnings/Duplicates**: Data quality indicators

### Model Insights
- High-quality architecture models have a sufficient number of elements, solid relationship to element rations, and multiple views.
- Warnings, duplicates, low element and low view counts lead to rejection.
- Machine learning models can predicted prelimianry ARB outcomes resulting in consistency and efficiency.
- Tree-based models (Random Forest, XGBoost) outperform linear models
- Ensemble methods (Hybrid Stacking) provide robust predictions
- Feature engineering (ratios) improves model performance
- SHAP analysis reveals view count and element relationships as key predictors

---

## Technologies Used

- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost, Optuna
- **Deep Learning**: Scikit-learn MLPClassifier
- **Visualization**: Matplotlib, Seaborn, SHAP
- **Web Application**: Streamlit
- **Model Persistence**: Joblib

---

## Interactive Application Features

The Streamlit app provides:
- **Input Interface**: Sidebar for entering model metrics
- **Prediction Tab**: 
  - Real-time ARB outcome prediction
  - Confidence probabilities visualization
  - Feature importance chart (Random Forest)
- **Model Comparison Tab**: 
  - Performance metrics for all models
  - Accuracy comparison visualization

---

## Future Enhancements

### Short-Term Improvements
- Add more model interpretability features
- Implement real-time model retraining
- Expand dataset with additional architecture model sources

### Long-Term Enhancement Options

#### Option 1 — Full Dashboard
- Approval trends visualization and analytics
- Model quality scoring system
- Automated documentation summaries
- Architect performance metrics 

#### Option 2 — Upload Portal
- Upload JSON / CSV EA metadata directly
- Automated preprocessing pipeline
- Real-time ARB recommendation generation

#### Option 3 — End-to-End Architecture Assistant
- Combine ML classifier with Large Language Model (LLM)
- Provide recommendations on fixing model quality issues
- Generate review summaries for ARB board members
- AI-powered improvement suggestions

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

**Anitra Hernandez**
- GitHub: [@Anitra-Hernandez](https://github.com/Anitra-Hernandez/AAI590_Capstone)
- Project: AAI590 Capstone - University of San Diego

---

## Acknowledgments

- **USD Applied AI Program**: Academic support and guidance
- **Professor Anna Marbut**: Academic support and guidance
- **EA Model Set Dataset**: Source dataset for enterprise architecture models
- **Streamlit**: Interactive web application
- **Open Source Community**: Libraries and tools that made this project possible

---

## References

Brunner, C., Linsbauer, L., Neumayer, V., & Wimmer, M. (2023). EA ModelSet: A FAIR dataset of enterprise architecture models. *arXiv*. https://arxiv.org/abs/2309.04169

OpenAI. (2025). ChatGPT (GPT-5.1) [Large language model]. Used for error resolution and debugging assistance. https://openai.com/chatgpt

Python Software Foundation. (2024). Python (Version 3.11) [Programming language]. https://www.python.org

Snow, P. (2019). *Streamlit: Turning data scripts into shareable web apps*. Streamlit Inc. https://streamlit.io

