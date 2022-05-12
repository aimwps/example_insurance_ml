from django.shortcuts import render
from django.views.generic import View
from .models import ModelTrainStatus
from django.http import JsonResponse
from .serializers import ModelTrainStatusSerializer
from .forms import ModelInferenceForm

class TrainedModels(View):
    template_name = "trained_models.html"
    def get(self, request):
        context = {}
        return render(request, self.template_name, context)

def GetModelStatusAjax(request):
    queryset = ModelTrainStatus.objects.all().order_by("accuracy")
    prepared_data = ModelTrainStatusSerializer(queryset, many=True)

    return JsonResponse(prepared_data.data, safe=False)

def DeleteModelStatusAjax(request):
    if request.method == "POST":
        object = ModelTrainStatus.objects.get(id=request.POST.get("training_id"))
        object.delete()
        return JsonResponse({"success":"success"})

class InferenceView(View):
    template_name = "inference.html"

    def get(self, request, training_id):
        context = {}
        object = ModelTrainStatus.objects.get(id=training_id)
        context['model'] = object
        context['form'] = ModelInferenceForm()
        return render(request, self.template_name, context)
