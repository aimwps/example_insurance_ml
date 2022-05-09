from celery                         import shared_task
from insurance_ml.global_constants  import CAT_FIELDS, NUM_FIELDS
from sklearn.tree                   import DecisionTreeRegressor
from sklearn.ensemble               import ExtraTreesRegressor, AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neural_network         import MLPRegressor
from sklearn.svm                    import SVR
from sklearn.linear_model           import Ridge, Lasso, SGDRegressor, BayesianRidge
from sklearn.neighbors              import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.model_selection        import train_test_split, GridSearchCV
from sklearn                        import metrics, preprocessing, impute, compose, pipeline
from catboost                       import CatBoostRegressor
from lightgbm                       import LGBMRegressor
from xgboost                        import XGBRegressor
import pandas as pd
import requests, json, time
random_state = 894
model_dict = {
        "tree": {
            "decisiontree": DecisionTreeRegressor(random_state=random_state),
            "extratrees": ExtraTreesRegressor(random_state=random_state),
            "randomforest": RandomForestRegressor(random_state=random_state),
            "hgb": HistGradientBoostingRegressor(random_state=random_state),
            "catboost": CatBoostRegressor(verbose=0, random_state=random_state),
            "gbm": GradientBoostingRegressor(random_state=random_state),
            "lightgbm": LGBMRegressor(random_state=random_state),
            "xgboost": XGBRegressor(random_state=random_state)
        },
        "mult": {
            "adaboost":AdaBoostRegressor(random_state=random_state),
            "svm": SVR(),
            "sgd": SGDRegressor(random_state=random_state),
            "multilayerperceptron": MLPRegressor(random_state=random_state),
            #"KNN": KNeighborsRegressor(),
        },
    }

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
    param_grid = [{
                "decisiontree__criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"] ,
                "extratrees__criterion":["squared_error","absolute_error",] ,
                "randomforest__criterion": ["squared_error", "absolute_error", "poisson"] ,
                "randomforest__n_estimators": [25, 50, 100, 150, 200, 250, 500] ,
                "hgb__loss":["squared_error", "absolute_error",  "poisson"] ,
                "hgb__learning_rate":[0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001],
                "hgb__min_samples_leaf":[12, 24, 36, 48],
                "catboost__depth": [6,8,10],
                "catboost__learning_rate": [0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001],
                "catboost__iterations": [25, 50, 100, 150, 200, 250, 500],
                "gbm__criterion": ["friedman_mse", "squared_error", "mse", "mae"] ,
                "gbm__n_estimators":[25, 50, 100, 150, 200, 250, 500],
                "lightgbm__learning_rate": [0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001],
                "lightgbm__boosting": ["gbdt","rf", "dart", "goss"],
                "adaboost__n_estimators": [25, 50, 100, 150, 200, 250, 500] ,
                "adaboost__loss": ["linear", "square", "exponential"],
                "adaboost__learning_rate": [0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001],
                "svm__C": [0.1,1, 10, 100],
                "svm__kernel":['rbf', 'poly', 'sigmoid'],
                "svm__gamma": [1,0.1,0.01,0.001]
                # "sgd": [],
                # "multilayerperceptron" :[] ,
                }]

    model_params = {}
    for key, value in param_grid[0].items():
        if f"{ml_model_name}__" in key:
            model_params[key] = value

    #param_grid = { key.split("__")[1] if f"{model_name}__" in key : value for key,value in param_grid }
    grid_search = GridSearchCV(regression_pipe, param_grid=model_params, cv=2)
    grid_search.fit(x_train, y_train)


    # best_model = ml_model
    regression_pipe_best = pipeline.Pipeline(steps=[("prepro", prepro), (ml_model_name, ml_model)]).set_params(**grid_search.best_params_)
    regression_pipe_best.fit(x_train, y_train)

    # Use the model to predict unseen data
    preds = regression_pipe_best.predict(x_test)

    # Calculate time it took to find best parameters and train model
    total_time = time.time() - start_time

    # Store results of the best para
    results = results.append({"Model": ml_model_name,
                              "MSE":   metrics.mean_squared_error(y_test, preds, squared=True),
                              "RMSE":  metrics.mean_squared_error(y_test, preds, squared=False),
                              "Time":  total_time},
                              ignore_index=True)

    return results

def get_data_from_api(training_settings):
    """
    gets the necessary data from the Insurance API.
    Returns a dataframe containing that data
    """
    r = requests.get("http://0.0.0.0:8000/api/medical_premiums/")
    data_json = json.dumps(r.json())
    df = pd.DataFrame(r.json())
    return df

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
        # handle_unknown="use_encoded_value"
    return tree_prepro

def build_mult_training(training_settings):
    """
    Applies preprocessing to multiplactive machine learning models.
    Categorical and numerical encoding is required for multiplactive.
    Returns a pipeline for preprocesing to be used with multiplactive ML models.
    """
    cat_vars, num_vars = split_cat_num_vars(training_settings)
    # Combine pipeline
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

    if ml_model_name in model_dict["tree"]:
        print("ITS A TREE MODEL")
        ml_model = model_dict["tree"][ml_model_name]
        prepro = build_tree_training(training_settings)

    elif ml_model_name in model_dict["mult"]:
        print("ITS A mult MODEL")
        ml_model = model_dict["mult"][ml_model_name]
        prepro = build_mult_training(training_settings)
    else:
        print(f"We didn't recognise the model type: {ml_model_name}")

    data = get_data_from_api(training_settings)

    results = train_model(data, prepro, training_settings, (ml_model_name, ml_model))

    print(results.head())
    return "success"
