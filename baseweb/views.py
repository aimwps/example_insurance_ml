from django.shortcuts import render
from django.views.generic import View
from insurance_ml.global_constants import ML_MODEL_OPTIONS, NUM_FIELDS, CAT_FIELDS
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
        print(request.POST)

        return HttpResponseRedirect
