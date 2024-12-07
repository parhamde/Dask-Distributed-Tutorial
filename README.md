# Dask-Distributed-Tutorial

# Distributed Machine Learning with Dask and Random Forest Classifier

This project demonstrates how to use **Dask Distributed** for parallel data processing and train a machine learning model (Random Forest Classifier) using **Scikit-learn**. The code is optimized for handling large datasets that may not fit into memory, leveraging Dask's capabilities for distributed computation.

---

## Prerequisites

Ensure you have the following installed:
- Python 3.7 or higher
- Required libraries:
  - `dask`
  - `distributed`
  - `scikit-learn`

To install the dependencies, run:
```bash
pip install dask distributed scikit-learn
```

---

## How It Works

1. **Dask for Data Processing**:
   - The dataset is loaded using `Dask`'s `read_csv` function, which supports parallel file reading for large datasets.
   - Lazy computation in Dask helps avoid loading the entire dataset into memory.

2. **Scikit-learn for Machine Learning**:
   - The `RandomForestClassifier` is used for supervised classification.
   - The dataset is split into training and testing sets.
   - Accuracy and a classification report (precision, recall, f1-score) are computed.

3. **Distributed Computing**:
   - A `Dask Client` is initialized for distributed task scheduling and monitoring.

---

## Code Overview

### Key Steps

1. **Set Up Dask Client**:
   ```python
   from dask.distributed import Client
   client = Client()
   ```

2. **Load Data with Dask**:
   ```python
   import dask.dataframe as dd
   data_dask = dd.read_csv('path_to_your_data.csv')
   ```

3. **Split Data**:
   ```python
   from sklearn.model_selection import train_test_split
   X = data_dask.drop(columns=['target_column'])
   y = data_dask['target_column']
   X_train, X_test, y_train, y_test = train_test_split(X.compute(), y.compute(), test_size=0.3, random_state=42)
   ```

4. **Train and Evaluate Model**:
   ```python
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import accuracy_score, classification_report
   
   model = RandomForestClassifier(random_state=42)
   model.fit(X_train, y_train)
   
   y_test_pred = model.predict(X_test)
   print(classification_report(y_test, y_test_pred))
   ```

5. **Close Dask Client**:
   ```python
   client.close()
   ```

---

## Running the Code

1. Clone or download this repository.
2. Replace `path_to_your_data.csv` in the code with the path to your dataset.
3. Run the script:
   ```bash
   python your_script_name.py
   ```

---

## Output

The script will display:
- **Training Accuracy**
- **Test Accuracy**
- **Classification Report** (Precision, Recall, F1-Score for each class)

Example output:
```
Train Accuracy: 0.95
Test Accuracy: 0.89
Classification Report:
              precision    recall  f1-score   support
           0       0.90      0.92      0.91       100
           1       0.88      0.85      0.86       120
    ...
```

---

## Notes

- Ensure your dataset is properly preprocessed and formatted before using this script.
- Adjust the model parameters (`RandomForestClassifier`) for better performance based on your dataset.

---

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute it as needed.
