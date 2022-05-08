from django.db import models
from insurance_ml.global_constants import TRAINING_STATUS, ML_MODEL_OPTIONS
# Create your models here.

class ModelTrainStatus(models.Model):
    create_date = models.DateField(auto_now=True)
    create_time = models.TimeField(auto_now=True)
    status = models.CharField(max_length=255, choices=TRAINING_STATUS, default="training")
    ml_model = models.CharField(max_length=255, choices=ML_MODEL_OPTIONS)
    rf_age = models.BooleanField(default=False)
    rf_gender = models.BooleanField(default=False)
    rf_bmi = models.BooleanField(default=False)
    rf_children = models.BooleanField(default=False)
    rf_is_smoker = models.BooleanField(default=False)
    rf_region = models.BooleanField(default=False)
    accuracy = models.FloatField(null=True, blank=True)
