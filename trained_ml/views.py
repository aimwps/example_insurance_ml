from django.shortcuts import render
from django.views.generic import View
from .models import ModelTrainStatus
from django.http import JsonResponse
from .serializers import ModelTrainStatusSerializer


class TrainedModels(View):
    template_name = "trained_models.html"
    def get(self, request):
        context = {}
        return render(request, self.template_name, context)

def GetModelStatusAjax(request):
    queryset = ModelTrainStatus.objects.all()
    return JsonResponse(safe=False)
