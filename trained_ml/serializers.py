from rest_framework import serializers
from .models import ModelTrainStatus


class ModelTrainStatusSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelTrainStatus
        fields = (
            "id",
            "create_date",
            "create_time",
            "status",
            "ml_model",
            "rf_age",
            "rf_age",
            "rf_gender",
            "rf_bmi",
            "rf_children",
            "rf_is_smoker",
            "rf_region",
            "accuracy",
            )
