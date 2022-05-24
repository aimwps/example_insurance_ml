from django.shortcuts import render
from django.views.generic import View, CreateView
from .models import ModelTrainStatus, ModelInference
from django.http import JsonResponse
from .serializers import ModelTrainStatusSerializer, ModelInferenceSerializer
from .forms import ModelInferenceForm
from .tasks import run_inference

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

class InferenceView(CreateView):
    template_name = "inference.html"

    def get(self, request, training_id):
        context = {}
        object = ModelTrainStatus.objects.get(id=training_id)
        context['model'] = object
        context['form'] = ModelInferenceForm()
        return render(request, self.template_name, context)

    def post(self, request, training_id):
        form =  ModelInferenceForm(request.POST)
        form.instance.on_model = ModelTrainStatus.objects.get(id=training_id)
        print(request.POST)
        x = request.POST.get("rf_is_smoker")
        print(f"HERE-->{x}")
        if request.POST.get("rf_is_smoker") == None:
            form.instance.rf_is_smoker = None

        if form.is_valid():
            object = form.save()
            print(object.rf_is_smoker)
            object_data = ModelInferenceSerializer(object).data
            print(object_data)
            run_inference.delay(object_data)

        return self.get(request, training_id)

def get_inference_results_ajax(request):
    model_id = request.GET.get("model_id")
    results = ModelInference.objects.filter(on_model=model_id)
    data = ModelInferenceSerializer(results, many=True).data

    return JsonResponse(data, safe=False)
