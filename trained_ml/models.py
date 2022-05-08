from django.db import models
from insurance_ml.global_constants import TRAINING_STATUS
# Create your models here.


class TrainingStatus(models.Model):
    status = models.CharField(max_length=255, choices=TRAINING_STATUS)
