from django import forms


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
                "rf_age": forms.TextInput('class': 'form-control'),
                "rf_gender": forms.Select('class': 'form-control'),
                "rf_bmi", : forms.TextInput('class': 'form-control'),
                "rf_children":forms.TextInput('class': 'form-control'),
                "rf_is_smoker": forms.CheckboxInput('class': 'form-check-input'),
                "rf_region": forms.Select('class': 'form-control'),
        }
