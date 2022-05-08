from celery                         import shared_task
from insurance_ml.global_constants  import CAT_FIELDS, NUM_FIELDS
from sklearn.tree                   import DecisionTreeRegressor
from sklearn.ensemble               import ExtraTreesRegressor, AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network         import MLPRegressor
from sklearn.svm                    import SVR
from sklearn.linear_model           import Ridge, Lasso, SGDRegressor, BayesianRidge
from sklearn.neighbors              import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.experimental           import enable_hist_gradient_boosting
from sklearn.ensemble               import HistGradientBoostingRegressor
from sklearn.model_selection        import train_test_split
from sklearn                        import metrics, preprocessing, impute, compose, pipeline
import pandas as pd
import requests, json, time


model_dict = {
        'tree': {
            "Decision Tree": DecisionTreeRegressor(),
            "Extra trees": ExtraTreesRegressor(),
            "Random Forest": RandomForestRegressor(),
            "HGB": HistGradientBoostingRegressor(),
            #"Cat Boost": CatBoostRegressor(verbose=0),
            "GBM": GradientBoostingRegressor(),
            #"Light GBM": LGBMRegressor(),
        },
        'mult': {
            "Ada Boost":AdaBoostRegressor(),
            "Super Vector Machine": SVR(),
            "SGD": SGDRegressor(),
            "KNN": KNeighborsRegressor(),
        },
    }


def train_model(data_df, prepro, training_settings, ml_model):
    #combine the model with preproccessing into a pipeline
    training_pipeline = pipeline.make_pipeline(prepro, ml_model)

    # Split the data into traning and target
    X = data_df.drop("premium", axis=1)
    Y = data_df['premium']

    # Split into a training and test set to judget model performance.
    x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, random_state=894)

    # Create a dataframe for storing the results
    results = pd.DataFrame({'Model': [], 'MSE': [], 'RMSE': [], 'Time': []})

    # Being training
    start_time = time.time()
    training_pipeline.fit(x_train, y_train)
    preds = training_pipeline.predict(x_valid)
    total_time = time.time() - start_time

    results = results.append({"Model": training_settings["ml_model"],
                              "MSE":   metrics.mean_squared_error(y_valid, preds, squared=True),
                              "RMSE":  metrics.mean_squared_error(y_valid, preds, squared=False),
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
    print(cat_vars)
    print("----------")
    print(num_vars)
    # Apply any preprocessing required for tree models to number variables
    tree_cat_var = pipeline.Pipeline(steps=[
        ('ONEHOTENCODER', preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ])

    # Apply any preprocessing required for tree models to number variables
    tree_prepro = compose.ColumnTransformer(transformers=[
        ("TREE_CAT_VAR", tree_cat_var, cat_vars),
        ], remainder='passthrough')

    return tree_prepro

def build_mult_training(training_settings):
    """
    Applies preprocessing to multiplactive machine learning models.
    Categorical and numerical encoding is required for multiplactive.
    Returns a pipeline for preprocesing to be used with multiplactive ML models.
    """
    cat_vars, num_vars = split_cat_num_vars(training_settings)


    # Create pipeline for preprocessing numerical variables
    mult_num_var = pipeline.Pipeline(steps=[
        ('STANDARDSCALER',  preprocessing.StandardScaler()),
        ])

    # Create pipeline for preprocessing categorical variables
    mult_cat_var = pipeline.Pipeline(steps=[
        ('ONEHOTENCODER', preprocessing.OneHotEncoder(handle_unknown='ignore'))
        ])

    # Combine pipeline
    mult_prepro = compose.ColumnTransformer(transformers=[
        ("MULT_NUM_VAR", mult_num_var, num_vars),
        ("MULT_CAT_VAR", mult_cat_var, cat_vars),
        ], remainder='passthrough')

    return mult_prepro


@shared_task
def train_model_from_db(training_settings):
    """
    A task that can be delayed with celery for long training times.
    Creates the training settings, preproceses data and stores the results
    and the model to be later called for inference.
    """
    # Find if selected model is tree/ensemble based or multiplative

    if training_settings['ml_model'] in model_dict['tree']:
        print("ITS A TREE MODEL")
        ml_model = model_dict['tree'][training_settings['ml_model']]
        prepro = build_tree_training(training_settings)
    elif training_settings['ml_model'] in model_dict['mult']:
        print("ITS A mult MODEL")
        ml_model = model_dict['mult'][training_settings['ml_model']]
        prepro = build_mult_training(training_settings)
    else:
        print("We didn't recognise the model type")

    data = get_data_from_api(training_settings)

    results = train_model(data, prepro, training_settings, ml_model)

    print(results.head())
    return "success"
