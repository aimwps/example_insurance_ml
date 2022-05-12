from celery import shared_task
from insurance_ml.constants import CAT_FIELDS, NUM_FIELDS, PARAM_GRID, MODEL_DICT
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics, preprocessing, impute, compose, pipeline
import pandas as pd
import requests
import json
import time
from django.apps import apps
import pickle

def _ModelTrainStatus():
    """
    This allows updates to records avoiding circular imports.
    As the task is imported to views, the model cannot be imported directly
    into tasks. This function works around the circular import.
    """
    return apps.get_model('trained_ml', "ModelTrainStatus")

def train_model(data_df, prepro, training_settings, model_details):
    # Extract name and model from arguments
    ml_model_name = model_details[0]
    ml_model = model_details[1]

    #combine the model with preproccessing into a pipeline
    regression_pipe = pipeline.Pipeline(steps=[("prepro", prepro), (ml_model_name, ml_model)])

    # Split the data into traning and target
    X = data_df.drop("premium", axis=1)
    Y = data_df["premium"]

    # Split into a training and test set to judget model performance.
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=894)

    # Create a dataframe for storing the results
    results = pd.DataFrame({"Model": [], "MSE": [], "RMSE": [], "Time": []})

    start_time = time.time()

    # Find the best parameters for the model
    model_params = {}
    for key, value in PARAM_GRID[0].items():
        if key.startswith(f"{ml_model_name}__"):
            model_params[key] = value

    grid_search = GridSearchCV(regression_pipe, param_grid=model_params, cv=2)
    grid_search.fit(x_train, y_train)

    # Use the best parameters found in grid search for creating a final training of the model
    regression_pipe_best = pipeline.Pipeline(steps=[("prepro", prepro), (ml_model_name, ml_model)]).set_params(**grid_search.best_params_)
    regression_pipe_best.fit(x_train, y_train)

    # Use the model to predict unseen data
    preds = regression_pipe_best.predict(x_test)

    # Save the mode
    id = training_settings['id']
    filename = f"model_data/{ml_model_name}_{id}"
    pickle.dump(regression_pipe_best, open(filename, 'wb'))

    # Calculate time it took to find best parameters and train model
    total_time = time.time() - start_time

    print(f"completed in {total_time}")
    # Store results of the best para
    results = results.append({"Model": ml_model_name,
                              "MSE":   metrics.mean_squared_error(y_test, preds, squared=True),
                              "RMSE":  round(metrics.mean_squared_error(y_test, preds, squared=False),2),
                              "Time":  total_time},
                              ignore_index=True)

    return results

def split_cat_num_vars(training_settings):
    """
    Returns 2 lists of variables split into categorical and numerical features.
    The lists will only contain fields selected by the user in training settings.
    """
    # Get a list of categorical variables from global constants:
    all_cat_vars = [cat_var[0] for cat_var in CAT_FIELDS]
    all_num_vars = [num_var[0] for num_var in NUM_FIELDS]

    # Determine if the variable is active, and what type of variable it is, store those variables
    cat_vars = []
    num_vars = []

    for key in training_settings:
        if training_settings[key] == True:
            if key in all_cat_vars:
                cat_vars.append(key)
            if key in all_num_vars:
                num_vars.append(key)
    return cat_vars, num_vars

def get_data_from_api(training_settings):
    """
    gets the necessary data from the Insurance API.
    Returns a dataframe of only the necessary fields
    """
    r = requests.get("http://0.0.0.0:8000/api/medical_premiums/")
    data_json = json.dumps(r.json())
    df = pd.DataFrame(r.json())

    cat_vars, num_vars = split_cat_num_vars(training_settings)
    vars = cat_vars + num_vars
    vars.append("premium")

    df = df[vars]

    print(df.head())
    return df

def build_tree_training(training_settings):
    """
    Applies preprocessing to tree and ensemble machine learning models.
    Categorical encoding is required for tree models.
    Numerical scaling or normalisation is irrelevant.
    Returns a pipeline for preprocesing to be used with tree ML models.
    """
    cat_vars, num_vars = split_cat_num_vars(training_settings)

    tree_prepro = compose.ColumnTransformer([
        ("cat_var", preprocessing.OrdinalEncoder(handle_unknown = "error"), cat_vars),
        ], remainder="drop")

    return tree_prepro

def build_mult_training(training_settings):
    """
    Applies preprocessing to multiplactive machine learning models.
    Categorical and numerical encoding is required for multiplactive.
    Returns a pipeline for preprocesing to be used with multiplactive ML models.
    """
    cat_vars, num_vars = split_cat_num_vars(training_settings)

    mult_prepro = compose.ColumnTransformer(transformers=[
        ("num_var", preprocessing.StandardScaler(), num_vars),
        ("cat_var", preprocessing.OrdinalEncoder(handle_unknown = "error"), cat_vars),
        ], remainder="drop")

    return mult_prepro

@shared_task
def train_model_from_db(training_settings):
    """
    A task that can be delayed with celery for long training times.
    Creates the training settings, preproceses data and stores the results
    and the model to be later called for inference.
    """

    # Find if selected model is tree/ensemble based or multiplative
    ml_model_name = training_settings["ml_model"]


    if ml_model_name in MODEL_DICT["tree"]:
        ml_model = MODEL_DICT["tree"][ml_model_name]
        prepro = build_tree_training(training_settings)

    elif ml_model_name in MODEL_DICT["mult"]:
        ml_model = MODEL_DICT["mult"][ml_model_name]
        prepro = build_mult_training(training_settings)
    else:
        print(f"We didn't recognise the model type: {ml_model_name}")

    # Collect dataset from insurance_api filter it on settings, and add it to a dataframe
    data = get_data_from_api(training_settings)

    # Run training
    results = train_model(data, prepro, training_settings, (ml_model_name, ml_model))

    # Select an orderby for the models
    training_status = _ModelTrainStatus().objects.get(id=training_settings['id'])

    training_status.accuracy = results['RMSE'][0]
    training_status.status = "complete"
    training_status.save()

    print(type(training_status))


    return "success"
