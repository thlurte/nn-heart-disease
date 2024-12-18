# Heart Disease Classification using Transformers

This project implements a deep learning pipeline for heart disease classification using a Transformer-based architecture. The model is trained on the UCI Heart Disease dataset to predict the presence of heart disease in patients.

## Dataset

The [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) contains 303 instances with 13 features:

1. **age**: age in years
2. **sex**: sex (1 = male; 0 = female)
3. **cp**: chest pain type
    - Value 1: typical angina
    - Value 2: atypical angina
    - Value 3: non-anginal pain
    - Value 4: asymptomatic
4. **trestbps**: resting blood pressure (in mm Hg)
5. **chol**: serum cholesterol in mg/dl
6. **fbs**: fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
7. **restecg**: resting electrocardiographic results
8. **thalach**: maximum heart rate achieved
9. **exang**: exercise induced angina (1 = yes; 0 = no)
10. **oldpeak**: ST depression induced by exercise relative to rest
11. **slope**: the slope of the peak exercise ST segment
12. **ca**: number of major vessels colored by fluoroscopy
13. **thal**: thalassemia type

**Target Variable**: Presence of heart disease (0 = no, 1 = yes)

## Project Structure
```
.
├── data
│   ├── data_loader.py
├── LICENSE
├── main.py
├── models
│   ├── dataset.py
│   └── transformer_model.py
├── README.md
├── training
│   └── trainer.py
└── utils
    ├── preprocessing.py

```

## Model Architecture

The project uses a Tabular Transformer architecture with the following specifications:

- Input dimension: 13 features
- Embedding dimension: 32
- Number of transformer layers: 3
- Number of attention heads: 4
- MLP dimension: 64
- Output classes: 2 (binary classification)

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies:
- torch
- transformers
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

## Usage

1. Clone the repository:

```bash
git clone https://github.com/thlurte/nn-heart-disease.git
```

2. Install dependencies:

```bash
python -m venv venv

source venv/bin/activate 
```

3. Run the training:

```bash
python main.py
```

## Results

The model generates the following outputs:

1. **Training History Plot** (`training_history.png`):
   - Training and validation loss curves
   - Training and validation accuracy curves

2. **Confusion Matrix** (`confusion_matrix.png`):
   - Visual representation of model predictions

3. **Detailed Results** (`training_results_TIMESTAMP.json`):
   - Training metrics per epoch
   - Final model performance
   - Classification report
   - Model hyperparameters

## Model Performance

The model achieves the following metrics (example metrics, update with your actual results):
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 | 0.85 | 0.88 | 0.86 | 50 |
| 1 | 0.87 | 0.84 | 0.85 | 48 |
| Accuracy | - | - | 0.86 | 98 |
| Macro Avg | 0.86 | 0.86 | 0.86 | 98 |
| Weighted Avg | 0.86 | 0.86 | 0.86 | 98 |


## Hyperparameters

- Batch size: 32
- Learning rate: 1e-4
- Number of epochs: 50
- Validation split: 0.2
- Optimizer: AdamW

## Contributing

Feel free to open issues and pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UCI Machine Learning Repository for the heart disease dataset
- PyTorch team for the deep learning framework