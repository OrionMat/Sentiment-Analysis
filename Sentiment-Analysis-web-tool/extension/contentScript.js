// chrome.contextMenus.onClicked.addListener(function(clickedData){
//   if (clickedData.menuItemId == "check_statement" && clickedData.selectionText){
//     sel = window.getSelection();
//     if (sel.rangeCount && sel.getRangeAt) {
//       range = sel.getRangeAt(0);
//     }
//     // Set design mode to on
//     document.designMode = "on";
//     if (range) {
//       sel.removeAllRanges();
//       sel.addRange(range);
//     }
//     // Colorize text
//     document.execCommand("ForeColor", false, "red");
//     // Set design mode to off
//     document.designMode = "off";
//   }
// });


// "use strict";

// var selection = window.getSelection();
// var selectionString = selection.toString();

// if (selectionString) { // If there is text selected

//     var container = selection.getRangeAt(0).commonAncestorContainer;

//     // Sometimes the element will only be text. Get the parent in that case
//     // TODO: Is this really necessary?
//     while (!container.innerHTML) {
//         container = container.parentNode;
//     }

//     chrome.storage.sync.get('color', (values) => {
//         var color = values.color;
//         store(selection, container, window.location.pathname, color, () => {
//             highlight(selectionString, container, selection, color);
//         });
//     });
// }