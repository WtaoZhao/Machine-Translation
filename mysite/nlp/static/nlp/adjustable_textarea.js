// Targets all textareas with class "txta"
let textareas = document.querySelectorAll('.txta'),
    hiddenDiv = document.createElement('div'),
    content = null;


// Adds a class to all textareas
for (let j of textareas) {
    j.classList.add('txtstuff');
}

// Build the hidden div's attributes

// The line below is needed if you move the style lines to CSS
// hiddenDiv.classList.add('hiddendiv');

// Add the "txta" styles, which are common to both textarea and hiddendiv
// If you want, you can remove those from CSS and add them via JS
hiddenDiv.classList.add('txta');

// Add the styles for the hidden div
// These can be in the CSS, just remove these three lines and uncomment the CSS
hiddenDiv.style.display = 'none';
hiddenDiv.style.whiteSpace = 'pre-wrap';
hiddenDiv.style.wordWrap = 'break-word';

function adjust(i){
    // Append hiddendiv to parent of textarea, so the size is correct
    parentNode=i.parentNode;
    parentNode.appendChild(hiddenDiv);

    // Remove this if you want the user to be able to resize it in modern browsers
    i.style.resize = 'none';

    // This removes scrollbars
    i.style.overflow = 'hidden';

    // Every input/change, grab the content
    content = i.value;

    // Add the same content to the hidden div

    // This is for old IE
    content = content.replace(/\n/g, '<br>');

    // The <br ..> part is for old IE
    const txtStyle=getComputedStyle(i)
    hiddenDiv.style.width = txtStyle.width;
    hiddenDiv.style.fontFeatureSettings = txtStyle.fontFeatureSettings;
    hiddenDiv.style.lineHeight = txtStyle.lineHeight;

    hiddenDiv.innerHTML = content + '<br style="line-height: 3px;">';

    // Briefly make the hidden div block but invisible
    // This is in order to read the height
    hiddenDiv.style.visibility = 'hidden';
    hiddenDiv.style.display = 'block';
    i.style.height = hiddenDiv.offsetHeight + 'px';
    var hBound=txtStyle.maxHeight;
    var thresh=parseInt(hBound)-10;
    thresh=thresh.toString()+"px";
    console.log(thresh);
    if(txtStyle.height>thresh){
        console.log("over");
        i.style.height=hBound;
        i.style.overflowY="scroll";
    }

    // Make the hidden div display:none again
    hiddenDiv.style.visibility = 'visible';
    hiddenDiv.style.display = 'none';
}

if( document.readyState == 'loading' )  {
    document.addEventListener('DOMContentLoaded', function () {
        console.log( 'document was not ready, place code here' );
        var txtList=document.querySelectorAll(".txta");
        for(let i of txtList){
            console.log('load');
            adjust(i);
        }
    });
}



// Loop through all the textareas and add the event listener
for (let i of textareas) {
    i.addEventListener('input',function () {
        console.log('input');
        adjust(i);
    });
}