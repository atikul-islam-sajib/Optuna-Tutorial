# Hyperparameter Optimization with Optuna

This REPO demonstrates how to perform hyperparameter optimization using **Optuna**, a flexible and efficient library for hyperparameter tuning. Optuna allows you to automatically search for the best hyperparameters, improving the performance of machine learning models in an easy-to-use and scalable way.

## Features

- Hyperparameter tuning with Optuna's `Trial` and `Study` objects.
- Visualization of optimization results, such as parameter importance and optimization history.
- Efficient parallelization of hyperparameter searches.

## Installation

To get started, install the required packages:

```bash
pip install optuna
```

## Usage

1. **Define Objective Function**: The objective function evaluates the model's performance based on given hyperparameters. Optuna optimizes this function.
   
   ```python
   import optuna
   from sklearn.datasets import load_iris
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import accuracy_score

   # Define the objective function for Optuna to minimize/maximize
   def objective(trial):
       # Suggest hyperparameters
       n_estimators = trial.suggest_int('n_estimators', 50, 200)
       max_depth = trial.suggest_int('max_depth', 3, 20)

       # Load data and split
       data = load_iris()
       X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

       # Train model with suggested hyperparameters
       model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
       model.fit(X_train, y_train)
       
       # Calculate accuracy
       y_pred = model.predict(X_test)
       accuracy = accuracy_score(y_test, y_pred)
       
       return accuracy  # Objective to maximize
   ```

2. **Run the Optimization**:
   
   ```python
   # Create a study object and specify the direction (maximize for accuracy)
   study = optuna.create_study(direction="maximize")
   study.optimize(objective, n_trials=100)  # Adjust number of trials as needed

   print("Best hyperparameters:", study.best_params)
   print("Best accuracy:", study.best_value)
   ```

3. **Visualize Results** (Optional):

   ```python
   # Plot the parameter importance
   optuna.visualization.plot_param_importances(study).show()

   # Plot the optimization history
   optuna.visualization.plot_optimization_history(study).show()
   ```

## Results

The best hyperparameters found by Optuna are:

- `n_estimators`: `...`
- `max_depth`: `...`

These hyperparameters yielded an accuracy of `...`.

## References

- [Optuna Documentation](https://optuna.readthedocs.io/en/stable/)
- [Optuna GitHub Repository](https://github.com/optuna/optuna)