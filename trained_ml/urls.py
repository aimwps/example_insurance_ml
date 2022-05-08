from django.contrib import admin
from django.urls import path, include
from .views import TrainedModels, GetModelStatusAjax

urlpatterns = [
    path("trained_models/", TrainedModels.as_view(), name="trained_models"),
    path("ajax_get_model_status", GetModelStatusAjax), 

]
