$(document).ready(function(){
  function checkSubmit(){
    if($("#model_type_input").val() != "" & $("#numerical_inputs").val() != "" & $("#categorical_inputs").val() != ""){
      $("#runTraining").removeClass("disabled")
    }else {
      $("#runTraining").addClass("disabled")
    }
  }
  $(document).on("click", "#submitSelectModel", function(e){
    e.preventDefault()
    let modelName = $("#modelSelectBox").val()
    $("#mlModelList").empty()
    $("#mlModelList").append(`
      <li class="list-group-item">
        <p class="lead">${modelName}</p>
      </li>
      `)
    $("#model_type_input").val(modelName).trigger("change");
    $("#modelModal").modal("hide");

  })
  $(document).on("click", "#submitSelectNumerical", function(e){
    e.preventDefault()
    let modelName = $("#numericalSelectBox").val()
    $("#numFieldsList").empty()
    $("#numFieldsList").append(`
      <li class="list-group-item">
        <p class="lead">${modelName}</p>
      </li>
      `)
    $("#numerical_inputs").val(modelName).trigger("change");
    $("#numericalModal").modal("hide");

  })
  $(document).on("click", "#submitSelectCategorical", function(e){
    e.preventDefault()
    let modelName = $("#categoricalSelectBox").val()
    $("#catFieldsList").empty()
    $("#catFieldsList").append(`
      <li class="list-group-item">
        <p class="lead">${modelName}</p>
      </li>
      `)
    $("#categorical_inputs").val(modelName).trigger("change");
    $("#categoricalModal").modal("hide");
  })
  $(document).on("change", ".MLInputs", function(e){
    checkSubmit()
  })
})
