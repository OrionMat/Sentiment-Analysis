$(function(){

    // displays the statement that is being searched
    // gets statement value from storage to display when popup opened
    chrome.storage.sync.get('statement', function(result){
        $('#statementTitle').text("\""+result.statement+"\"");
    });

    // gets results from storage to display when popup opened
    chrome.storage.sync.get(['results'], function(result){
        result_array = JSON.parse(result.results)

        var i;
        for(i = 0; i<2; i++){
            $('#result' + String(i)).text(result_array[i]);
        }
    });

});