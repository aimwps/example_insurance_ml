$(document).ready(function(){
  function getModelStatus(){
    $.ajax({type:"GET",
            url: "/ajax_get_model_status/",
            datatype: "json",
            success: function(json){
              console.log(json);
              $("#displayTrainedModels").empty()
              if (json.length == 0){
                $("#displayTrainedModels").append(`
                  <li class="list-group-item">
                    <p class="lead">No models have been set to train..
                  </li>
                  `)
            } else {
              $(json).each(function(i, model){
                $("#displayTrainedModels").append(`
                  <li class="list-group-item px-0">
                    <button class="btn btn-link w-100" data-bs-toggle="collapse" href="#collapseModelInfo_${model.id}" role="button" aria-expanded="false" aria-controls="collapseModelInfo_${model.id}">
                      Created a <strong>${model.ml_model}</strong> on ${model.create_date} @ ${model.create_time.slice(0, -10)} [STATUS: <strong>${model.status}</strong>]
                    </button>
                  <div class="collapse my-2" id="collapseModelInfo_${model.id}">
                    <div class="card card-body">
                      Model uses data:
                      <ul class="list-group-flush">
                        <li class="list-group-item">
                          Age: ${model.rf_age}
                        </li>
                        <li class="list-group-item">
                          BMI: ${model.rf_bmi}
                        </li>
                        <li class="list-group-item">
                          Children: ${model.rf_children}
                        </li>
                        <li class="list-group-item">
                          Is Smoker: ${model.rf_is_smoker}
                        </li>
                        <li class="list-group-item">
                          Region: ${model.rf_region}
                        </li>
                        <li class="list-group-item">
                          Gender: ${model.rf_gender}
                        </li>
                      </ul>
                    </div>
                  </div>
                </li>
                  `)

              })
            }

  }})
}
  getModelStatus()
})
