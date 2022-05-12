from django.urls import path
from .views import TrainAModel
urlpatterns = [
    path("", TrainAModel.as_view(), name="train_model")
    

]
