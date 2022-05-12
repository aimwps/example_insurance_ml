$(document).ready(function(){
    var questionTimeCounter = setInterval(getModelStatus, 5000);
  function getBoolIcon(bool){
    if(bool === true){
      return `<i class="fa-solid fa-check"></i>`
    } else {
      return `<i class="fa-solid fa-xmark"></i>`
    }
  }
  function nullCheck(item){
    if(item){
      return item;
    } else {
      return " --.--";
    }
  }
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
                  <li class="list-group-item px-2 my-2">
                      <h4 class="p-2"><strong>${model.ml_model}</strong> on ${model.create_date} @ ${model.create_time.slice(0, -10)} </h4>
                      <div class="container">
                        <div class="row">
                          <div class="col-xs-12 col-md-4">
                          <h5>Training features</h5>
                            Age: ${getBoolIcon(model.rf_age)} BMI: ${getBoolIcon(model.rf_bmi)} Children: ${getBoolIcon(model.rf_children)}<br>
                            Is Smoker: ${getBoolIcon(model.rf_is_smoker)} Region: ${getBoolIcon(model.rf_region)}Gender: ${getBoolIcon(model.rf_gender)}
                          </div>
                          <div class="col-xs-12 col-md-3">
                          <h5>Status<h5>
                          <strong>${model.status}</strong>
                          </div>
                          <div class="col-xs-12 col-md-3">
                          <h5>Result (RMSE)<h5>
                          <strong>$${nullCheck(model.accuracy)}</strong>
                          </div>
                          <div class="col-xs-12 col-md-2">

                          <a class="btn btn-sm w-100 mb-1" href="/inference/${model.id}" name="infermodel" id="infermodel_${model.id}">
                            infer
                          </a>
                          <br>
                          <button class="btn btn-sm w-100 mt-1" value="${model.id}" name="deletemodel" id="deletemodel_${model.id}">
                            delete
                          </button>
                          </div>
                          </div>
                          </div>
                </li>
                  `)

              })
            }

  }})
}

  function deleteModelEntry(id){
    $.ajax({
      type: "post",
      url: "/ajax_delete_model_training/",
      datatype: "json",
      data:{training_id : id,
            csrfmiddlewaretoken:$('input[name=csrfmiddlewaretoken]').val(), },
      success:function(){
        getModelStatus()
      }
    })
  }
  getModelStatus()
  $(document).on("click", "button[name='deletemodel']", function(e){
    e.preventDefault()
    let id = $(this).val();
    deleteModelEntry(id)
  })
})
