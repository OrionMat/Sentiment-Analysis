var contextMenuItem = {
  "id" : "check_statement",
  "title" : "Analyse Text",
  "contexts" : ["selection"]
};
chrome.contextMenus.create(contextMenuItem)

function is_valid_statement(text){
  // checks text is valid
  return true
}

chrome.contextMenus.onClicked.addListener(function(clickedData){
  if (clickedData.menuItemId == "check_statement" && clickedData.selectionText){
    if (is_valid_statement(clickedData.selectionText)){

      // clear previouse results
      // chrome.storage.sync.set({'results' : '0'})

      chrome.storage.sync.set({'statement': clickedData.selectionText});
      var port = chrome.runtime.connectNative('host_manifest'); // runs python script

      port.onMessage.addListener(function(msg) {
        if(msg.name == "articleResults"){
          console.log(msg.results)
          chrome.storage.sync.set({"results" : msg.results})

          if (JSON.parse(msg.results)[0] > 0) {
          // create a notification
            var notifOptions = {
              type: "basic",
              iconUrl: "images/sentiment_pic_happy.png",
              title: "Done!",
              message: "The results are in!"
            };
          } else if (JSON.parse(msg.results)[0] < 0) { 
            // create a notification
            var notifOptions = {
              type: "basic",
              iconUrl: "images/sentiment_pic_sad.png",
              title: "Done!",
              message: "The results are in!"
            };
          } else {
            // create a notification
            var notifOptions = {
              type: "basic",
              iconUrl: "images/thumbs_up_down.png",
              title: "Done!",
              message: "The results are in!"
            };
          }
          chrome.notifications.create('doneNotif', notifOptions);
        }
      });

      port.onDisconnect.addListener(function() {
        console.log("Disconnected");
      });
      port.postMessage({ text: clickedData.selectionText });
      console.log("Attempted to send to host.")
    }
  }
});