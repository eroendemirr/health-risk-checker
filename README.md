# Health Risk Checker

This project aims to provide a preliminary health risk evaluation using personal medical data. The main goal is to allow individuals to assess whether they are at potential risk without the immediate need to visit a hospital.

## Features

- Trains a machine learning model (Random Forest) on labeled medical data.
- Splits data into training and testing sets for model evaluation.
- Outputs a classification report with precision, recall, and F1 scores.
- Displays a confusion matrix for visual performance assessment.

## Dataset

The dataset (`veriler.csv`) contains structured health data with a `label` column indicating risk classification (e.g., "at risk" or "not at risk").

## Requirements

- Python 3.x
- pandas
- scikit-learn
- matplotlib

You can install dependencies using:

```bash
pip install pandas scikit-learn matplotlib
