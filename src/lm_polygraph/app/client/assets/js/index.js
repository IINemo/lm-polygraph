const API_URL = '/';
const converter = new showdown.Converter();
let promptToRetry = null;
let uniqueIdToRetry = null;

const submitButton = document.getElementById('submit-button');
const promptInput = document.getElementById('prompt-input');
const modelSelect = document.getElementById('model');
const seqUeSelect = document.getElementById('seque');
const responseList = document.getElementById('response-list');
const modal = document.getElementById("modal");
const settingsButton = document.getElementById("settings-button");
const span = document.getElementsByClassName("close")[0];
const repetitionPenaltyInput = document.getElementById("repetition-penalty-input");
const repetitionPenaltyStyle = repetitionPenaltyInput.style.display;
const presencePenaltyInput = document.getElementById("presence-penalty-input");
const presencePenaltyStyle = presencePenaltyInput.style.display;
const topkInput = document.getElementById("topk-slider");
const topkStyle = topkInput.style.display;

promptInput.addEventListener('paste', function(e) {
    // To keep same text style + disable pasting non-text stuff like images
    e.preventDefault();
    var pastedText = (e.originalEvent || e).clipboardData.getData('text/plain');
    document.execCommand('insertHTML', false, pastedText);
});

const placeholder = `<div style="color: gray">` + promptInput.getAttribute('placeholder') + `</div>`;

// Set the placeholder as initial content if it's empty
promptInput.innerHTML === '' && (promptInput.innerHTML = placeholder);

promptInput.addEventListener('focus', function (e) {
    const value = e.target.innerHTML;
    value === placeholder && (e.target.innerHTML = '');
});

promptInput.addEventListener('blur', function (e) {
    const value = e.target.innerHTML;
    value === '' && (e.target.innerHTML = placeholder);
});

var temperature = 1.0;
var topk = 1;
var topp = 1.0;
var num_beams = 1;
var presence_penalty = 0;
var repetition_penalty = 1;

settingsButton.onclick = function() {
  modal.style.display = "block";
  const model = modelSelect.__vue__.modelSelected;
  if ('GPT-4' === model || 'GPT-3.5-turbo' === model) {
    topkInput.style.display = "none";
    repetitionPenaltyInput.style.display = "none";
    presencePenaltyInput.style.display = presencePenaltyStyle;
  } else {
    topkInput.style.display = topkStyle;
    repetitionPenaltyInput.style.display = repetitionPenaltyStyle;
    presencePenaltyInput.style.display = "none";
  }
}

span.onclick = function() {
  modal.style.display = "none";
}

window.onclick = function(event) {
  if (event.target == modal) {
    modal.style.display = "none";
  }
}

let isGeneratingResponse = false;
let loadInterval = null;

promptInput.addEventListener('keydown', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault();
        if (event.ctrlKey || event.shiftKey) {
            document.execCommand('insertHTML', false, '<br/><br/>');
        } else {
            getGPTResult();
        }
    }
});

function generateUniqueId() {
    const timestamp = Date.now();
    const randomNumber = Math.random();
    const hexadecimalString = randomNumber.toString(16);

    return `id-${timestamp}-${hexadecimalString}`;
}

function updateTemp() {
    var value = document.getElementById("temp-slider-value").value;
    var logValue = document.getElementById("temp-value");
    var logScaleValue = Math.min(Math.max(Math.round(100 * (Math.pow(2, value / 25) - 1)) / 100, 0.01), 2);
    logValue.innerHTML = logScaleValue;
    temperature = logScaleValue;
}

function updatePresencePenalty() {
    var value = document.getElementById("presence-penalty-slider-value").value;
    document.getElementById("presence-penalty-value").innerHTML = value;
    presence_penalty = value;
}

function updateRepetitionPenalty() {
    var value = document.getElementById("repetition-penalty-slider-value").value;
    var logValue = document.getElementById("repetition-penalty-value");
    var logScaleValue = Math.round(Math.pow(10, value) * 100) / 100;
    logValue.innerHTML = logScaleValue;
    repetition_penalty = logScaleValue;
}


function updateTopk() {
    var value = document.getElementById("topk-slider-value").value;
    topk = value;
}

function updateTopp() {
    var value = document.getElementById("topp-slider-value").value;
    var logValue = document.getElementById("topp-value");
    logValue.innerHTML = value;
    topp = value;
}

function updateNumBeams() {
    num_beams = value;
}

var escape = document.createElement('textarea');

function rawHtml(text) {
    escape.textContent = text;
    return escape.innerHTML;
}

function addResponse(selfFlag, desc, prompt) {
    const uniqueId = generateUniqueId();
    html = `
            <div class="response-container ${selfFlag ? 'my-question' : 'chatgpt-response'}">
                    <div class="desc-container">
                        ${desc}
                    </div>
                    <img class="avatar-image" src="assets/img/${selfFlag ? 'me.png' : 'ai.svg'}" alt="avatar"/>
                    <div class="prompt-content" id="${uniqueId}">
                       ${prompt}
                    </div>
            </div>
        `
    if (selfFlag) {
        html = `<div class="total-response-my-question">` + html + `</div>`
    }
    else {
        html = `<div class="total-response-chatgpt-response">` + html  + `</div>`
    }
    responseList.insertAdjacentHTML('beforeend', html);
    responseList.scrollTop = responseList.scrollHeight;
    return uniqueId;
}

addResponse(false, '', rawHtml("This is LM-Polygraph demo: it augments LLM responses with confidence scores, " +
    "helping to determine the reliability of LLM's answers. Choose a model and an uncertainty estimation method first."));

function dropDown(event) {
    event.target.parentElement.children[1].classList.remove("d-none");
    document.getElementById("overlay").classList.remove("d-none");
}

function hide(event) {
    var items = document.getElementsByClassName('menu');
    for (let i = 0; i < items.length; i++) {
        items[i].classList.add("d-none");
    }
    document.getElementById("overlay").classList.add("d-none");
}

function loader(element) {
    element.textContent = ''

    loadInterval = setInterval(() => {
        // Update the text content of the loading indicator
        element.textContent += '.';

        // If the loading indicator has reached three dots, reset it
        if (element.textContent === '....') {
            element.textContent = '';
        }
    }, 300);
}

function setErrorForResponse(element, message) {
    element.innerHTML = message;
    element.style.color = 'rgb(200, 0, 0)';
}

function setRetryResponse(prompt, uniqueId) {
    promptToRetry = prompt;
    uniqueIdToRetry = uniqueId;
}

function formatResponse(data){
    let generation = data.generation;
    let tok_confidence = data.token_confidence;
    let tok_norm = data.token_normalization;
    let seq_confidence = data.sequence_confidence;
    let seq_norm = data.sequence_normalization;

    console.log('Model output:')
    console.log('  generation: ' + generation)
    console.log('  token confidence: ' + tok_confidence);
    console.log('  token normalization: ' + tok_norm);
    console.log('  sequence confidence: ' + seq_confidence);
    console.log('  sequence normalization: ' + seq_norm);

    let response = '<p style="display:inline">';
    let started_border = false;
    let confidence_thrs = 0.2;
    for (let i = 0; i < generation.length; i++) {
        if (tok_norm == 'none' || tok_confidence[i] >= confidence_thrs || tok_confidence.length == 0) {
            response += rawHtml(generation[i]);
            continue;
        }
        const white = 100;
        const green = white + Math.floor((255 - white) * tok_confidence[i]);
        const red = white + Math.floor((255 - white) * (1 - tok_confidence[i]));
        const color = 'rgb(' + red + ', ' + green + ', ' + white + ')';
        for (let j = 0; j < generation[i].length - generation[i].trimLeft().length; j++) {
            response += ' ';
        }
        if (generation[i].trim().length > 0) {
            let left_border = false;
            if (!started_border) {
                left_border = true;
                started_border = true;
            }
            let right_border = false;
            if (generation[i].slice(-1) == ' ' || i + 1 == generation.length || (
                    i + 1 < generation.length &&
                    (generation[i + 1][0] == ' ' || tok_confidence[i + 1] >= confidence_thrs))) {
                right_border = true;
                started_border = false;
            }
            let border_str = '';
            if (left_border) {
                border_str += 'border-top-left-radius:10px;border-bottom-left-radius:10px;padding-left:3px;';
            }
            if (right_border) {
                border_str += 'border-top-right-radius:10px;border-bottom-right-radius:10px;padding-right:3px;';
            }
            response += '<span style="background-color:' + color + ';' + border_str + 'padding-top:3px;padding-bottom:3px">';
            response += rawHtml(generation[i].trim());
            response += '</span>';
        }
        for (let j = 0; j < generation[i].length - generation[i].trimRight().length; j++) {
            response += ' ';
        }
    }

    if (seq_confidence.length != 0) {
        const white = 100;
        const green = white + Math.floor((255 - white) * seq_confidence[0]);
        const red = white + Math.floor((255 - white) * (1 - seq_confidence[0]));
        var colorStyle = '';
        var ueStr = seq_confidence[0].toPrecision(3);
        if (seq_norm != 'none') {
            colorStyle = 'background-color:rgb(' + red + ', ' + green + ', ' + white + ');';
            ueStr = Math.round(seq_confidence[0] * 100) + '%';
        }
        response += '<div style="line-height:30%;"><br></div>Confidence: </span>';
        response += '<span style="' + colorStyle + 'border-radius:10px;padding: 3px;">' + ueStr + '</span>';
    }
    response += '</p>';
    return response
}



// Function to get GPT result
async function getGPTResult(_promptToRetry, _uniqueIdToRetry) {
    // Get the prompt input
    const prompt = _promptToRetry ?? promptInput.textContent;

    const model = modelSelect.__vue__.modelSelected;
    if (!model) {
        return;
    }
    const seq_ue = seqUeSelect.__vue__.sequeSelected;

    let tok_str = "None";
    let seq_str = "None";
    if (seq_ue)
        seq_str = seq_ue;

    // Left for token-level
//    const modeldesc = model;
//    const tokdesc = 'token-level: ' + tok_str;
//    const seqdesc = 'sequence-level: ' + seq_str;
//    const desc = `${modeldesc}<br>${tokdesc}<br>${seqdesc}`;
    const desc = `${model}<br>${seq_str}`;

    // If a response is already being generated or the prompt is empty, return
    if (isGeneratingResponse || !prompt) {
        return;
    }

    // Add loading class to the submit button
    submitButton.classList.add("loading");

    // Clear the prompt input
    promptInput.textContent = '';

    if (!_uniqueIdToRetry) {
        // Add the prompt to the response list
        addResponse(true, ``, `<div><p style="display:inline">${rawHtml(prompt)}</p></div>`, );
    }

    // Get a unique ID for the response element
    const uniqueId = _uniqueIdToRetry ?? addResponse(false, desc);

    // Get the response element
    const responseElement = document.getElementById(uniqueId);

    // Show the loader
    loader(responseElement);

    // Set isGeneratingResponse to true
    isGeneratingResponse = true;

    var ensemblePathInput = document.getElementById('ensemble-path-input');
    var ensembles
    if (ensemblePathInput) {
        var ensembles = ensemblePathInput.value
    }
    var openaiKeyInput = document.getElementById('openai-secret-key-input');
    var openai_key
    if (openaiKeyInput) {
        var openai_key = openaiKeyInput.value
    }

    // TODO(rediska): надо пошифровать openai_key

    try {
        // Send a POST request to the API with the prompt in the request body
        const response = await fetch(API_URL + 'get-prompt-result', {
        // const response = await fetch('localhost:5239/get-prompt-result', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt,
                model,
                openai_key,
                ensembles,
                seq_ue,
                temperature,
                topp,
                topk,
                num_beams,
                repetition_penalty,
                presence_penalty
            })
        });
        console.log(response)
        if (!response.ok) {
            setRetryResponse(prompt, uniqueId);
            var err_text = (await response.text()).split('<p>');
            err_text = err_text[err_text.length - 1];
            err_text = err_text.split('</p>')[0];
            setErrorForResponse(responseElement, `Error: ${err_text}`);
            return;
        }
        responseRaw = await response.json()
        responseText = formatResponse(responseRaw)
        responseElement.innerHTML = responseText;
        // responseText = await response.text();
        // console.log(responseRaw)
        // console.error('response html: ' + responseElement.innerHTML);

        promptToRetry = null;
        uniqueIdToRetry = null;
        setTimeout(() => {
            // Scroll to the bottom of the response list
            responseList.scrollTop = responseList.scrollHeight;
            hljs.highlightAll();
        }, 10);
    } catch (err) {
        setRetryResponse(prompt, uniqueId);
        // If there's an error, show it in the response element
        setErrorForResponse(responseElement, `Error: ${err.message}`);
    } finally {
        // Set isGeneratingResponse to false
        isGeneratingResponse = false;

        // Remove the loading class from the submit button
        submitButton.classList.remove("loading");

        // Clear the loader interval
        clearInterval(loadInterval);
    }
}

submitButton.addEventListener("click", () => {
    getGPTResult();
});


async function checkMethods() {
    try {
        const response = await fetch(API_URL + 'methods', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
        })
        const methods_reponse = await response.json()
        if (methods_reponse.allow_all){
            seqUeSelect.__vue__.allMethods=true;
            modelSelect.__vue__.allModels=true;
        }

    } catch (error) {
        console.log(error)
    }
}

document.addEventListener("DOMContentLoaded", function(){
    promptInput.focus();
    checkMethods()
});
