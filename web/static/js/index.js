console.log("Webpage started");

let morphResult = document.getElementById("morph-result");
let morphAnswer = document.getElementById("answer-div");

$(function() {
    $('#new-morph-btn').click(function() {
        morphAnswer.innerHTML = "";
        morphResult.src = "/images/loading.gif";
        $.ajax({
            type: 'POST',
            url: '/new_morph',
            contentType: false,
            cache: false,
            processData: false,
            success: function(data) {
                morphResult.style.height = null;
                morphResult.src = `data:png;base64,` + data;
            },
        });
    });
});

$(function() {
    $('#answer-btn').click(function() {
        $.ajax({
            type: 'POST',
            url: '/answer',
            contentType: false,
            cache: false,
            processData: false,
            success: function(data) {
                if(data.name1 != null){ 
                    let answer = `This is the child of ${data.name1} and ${data.name2}.`;
                    morphAnswer.innerHTML = answer;
                }
            },
        });
    });
});

