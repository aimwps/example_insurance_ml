from django import forms
from .models import ModelInference

class ModelInferenceForm(forms.ModelForm):
    class Meta:
        model = ModelInference
        fields = (
                "rf_age",
                "rf_gender",
                "rf_bmi",
                "rf_children",
                "rf_is_smoker",
                "rf_region",
                )
        widgets = {
                "rf_age": forms.TextInput(attrs={'class': 'form-control'}),
                "rf_gender": forms.Select(attrs={'class': 'form-control'}),
                "rf_bmi" : forms.TextInput(attrs={'class': 'form-control'}),
                "rf_children":forms.TextInput(attrs={'class': 'form-control'}),
                "rf_is_smoker": forms.CheckboxInput(attrs={'class': 'form-check-input'}),
                "rf_region": forms.Select(attrs={'class': 'form-control'}),
        }
