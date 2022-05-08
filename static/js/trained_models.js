$(document).ready(function(){
  function getModelStatus(){
    $.ajax({type:"GET",
            url: "/ajax_get_model_status/",
            datatype: "json",
            success: function(json){
              $(json).each(function(i, model){
                console.log(model)
              })

            }

    })
  }
  getModelStatus()
})
