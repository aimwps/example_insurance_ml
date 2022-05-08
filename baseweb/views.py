from django.shortcuts import render, redirect
from django.views.generic import View
from insurance_ml.global_constants import ML_MODEL_OPTIONS, NUM_FIELDS, CAT_FIELDS
from trained_ml.models import ModelTrainStatus
from trained_ml.tasks import train_model_from_db
# Create your views here.



## User can select a machine learning model
## User can select data to train model on from API
## User can run training, recieve accuracy results
## User can save model information
## User can use model for inference

class TrainAModel(View):
    template_name = "train_a_model.html"


    def get(self, request):
        context = {}
        context['ml_model_select'] = ML_MODEL_OPTIONS
        context['cat_field_select'] = CAT_FIELDS
        context['num_field_select'] = NUM_FIELDS
        return render(request, self.template_name, context)

    def post(self, request):

        # gather data required from post request
        numerical_fields =  request.POST.getlist("numerical")
        categorical_fields = request.POST.getlist("categorical")
        all_data_fields_set = numerical_fields + categorical_fields
        ml_model = request.POST.get("modeltype")

        # generate fields names we are training data on (All contain 'rf_')
        all_model_field_names = [field.name for field in ModelTrainStatus._meta.get_fields()]
        risk_factors = [rf for rf in all_model_field_names if "rf_" in rf]

        # Determine required settings from post
        training_data_settings = {}
        for rf in risk_factors:
            if rf in all_data_fields_set:
                training_data_settings[rf] = True
            else:
                training_data_settings[rf] = False

        # Create the new training record based on settings
        new_training = ModelTrainStatus(
                            ml_model = ml_model,
                            rf_age = training_data_settings['rf_age'],
                            rf_gender = training_data_settings['rf_gender'],
                            rf_bmi = training_data_settings['rf_bmi'],
                            rf_children = training_data_settings['rf_children'],
                            rf_is_smoker = training_data_settings['rf_is_smoker'],
                            rf_region = training_data_settings['rf_region'],
        )
        new_training.save()

        train_model_from_db.delay(new_training)



        return redirect("trained_models")
