from django.db import models
from insurance_ml.constants import TRAINING_STATUS, ML_MODEL_OPTIONS
from django.core.validators import MaxValueValidator, MinValueValidator
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


class ModelInference(models.Model):
    gender_choices = (("female", "Female"),("male", "Male"),)
    region_choices = (("northwest", "North West"), ("northeast", "North East"), ("southeast", "South East"), ("southwest", "South West"))
    create_date = models.DateField(auto_now=True)
    create_time = models.TimeField(auto_now=True)
    on_model = models.ForeignKey(ModelTrainStatus, on_delete=models.CASCADE)
    rf_age = models.PositiveIntegerField()
    rf_gender = models.CharField(max_length=255, choices=gender_choices)
    rf_bmi = models.DecimalField(max_digits=4, decimal_places=1, validators=[MinValueValidator(0.0), ], )
    rf_children = models.PositiveIntegerField()
    rf_is_smoker = models.BooleanField()
    rf_region = models.CharField(max_length=255, choices=region_choices)
    premium = models.DecimalField(max_digits=13, decimal_places=2)
