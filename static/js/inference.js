$(document).ready(function(){
  var questionTimeCounter = setInterval(getInferenceResultsAjax, 5000);
  function getInferenceResultsAjax(){
    let modelId = $("#modelId").val()
    $.ajax({url: "/get_inference_results_ajax/",
            type: "GET",
            data: {model_id: modelId},
            datatype: "json",
            success: function(json){
              console.log(json)
              $("#inferenceResults").empty()
              $.each(json, function(i, result){
                console.log(result)
                let sentenceBuilder = ""
                if (result.rf_age != null){
                  sentenceBuilder += ` Age: ${result.rf_age}`
                }
                if (result.rf_gender != null){
                  sentenceBuilder += ` Gender: ${result.rf_gender}`
                }
                if (result.rf_bmi != null){
                  sentenceBuilder += ` BMI: ${result.rf_bmi}`
                }
                if (result.rf_children != null){
                  sentenceBuilder += ` Children: ${result.rf_children}`
                }
                if (result.rf_is_smoker != null){
                  sentenceBuilder += ` Is Smoker: ${result.rf_is_smoker}`
                }
                if (result.rf_region != null){
                  sentenceBuilder += ` Region: ${result.rf_region}`
                }
                $("#inferenceResults").append(`
                  <li class="list-group-item">
                  <strong>Inputs: </strong>${sentenceBuilder}<br><strong>Premimum: </strong>$${result.premium}
                  </li>`)
              })
            }})
  }
  getInferenceResultsAjax()
})
