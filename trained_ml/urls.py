from django.contrib import admin
from django.urls import path, include
from .views import TrainedModels, GetModelStatusAjax, DeleteModelStatusAjax, InferenceView

urlpatterns = [
    path("trained_models/", TrainedModels.as_view(), name="trained_models"),
    path("ajax_get_model_status/", GetModelStatusAjax),
    path("ajax_delete_model_training/", DeleteModelStatusAjax),
    path("inference/<int:training_id>", InferenceView.as_view(), name="inference")

]
