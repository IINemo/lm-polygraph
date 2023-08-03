const API_URL = '/';
const converter = new showdown.Converter();
let promptToRetry = null;
let uniqueIdToRetry = null;

const submitButton = document.getElementById('submit-button');
const promptInput = document.getElementById('prompt-input');
const modelSelect = document.getElementById('model');
const tokUeSelect = document.getElementById('tokue');
const seqUeSelect = document.getElementById('seque');
const responseList = document.getElementById('response-list');
const selectContainer = document.getElementById('select-container');
const modal = document.getElementById("modal");
const settingsButton = document.getElementById("settings-button");
const span = document.getElementsByClassName("close")[0];

var temperature = 1.0;
var topk = 1;
var topp = 1.0;
var num_beams = 1;
var repetition_penalty = 1;

settingsButton.onclick = function() {
  modal.style.display = "block";
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
    var logScaleValue = Math.max(Math.round(100 * (Math.pow(2, value / 25) - 1)) / 100, 0.01);
    logValue.innerHTML = logScaleValue;
    temperature = logScaleValue;
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

// Function to get GPT result
async function getGPTResult(_promptToRetry, _uniqueIdToRetry) {
    // Get the prompt input
    const prompt = _promptToRetry ?? promptInput.textContent;

    const model = modelSelect.__vue__.modelSelected;
    const tok_ue = tokUeSelect.__vue__.tokueSelected;
    const seq_ue = seqUeSelect.__vue__.sequeSelected;

    let tok_str = "None";
    if (tok_ue.length > 0) {
        tok_str = tok_ue.join(', ');
    }
    let seq_str = "None";
    if (seq_ue.length > 0) {
        seq_str = seq_ue.join(', ');
    }
    const modeldesc = model;
    const tokdesc = 'token-level: ' + tok_str;
    const seqdesc = 'sequence-level: ' + seq_str;
    const desc = `${modeldesc}<br>${tokdesc}<br>${seqdesc}`;

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
        addResponse(true, ``, `<div>${prompt}</div>`, );
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
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt,
                model,
                openai_key,
                ensembles,
                tok_ue,
                seq_ue,
                temperature,
                topp,
                topk,
                num_beams,
                repetition_penalty
            })
        });
        if (!response.ok) {
            setRetryResponse(prompt, uniqueId);
            setErrorForResponse(responseElement, `HTTP Error: ${await response.text()}`);
            return;
        }
        responseText = await response.text();

        responseElement.innerHTML = converter.makeHtml(responseText);
        console.error('response html: ' + responseElement.innerHTML);

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

document.addEventListener("DOMContentLoaded", function(){
    promptInput.focus();
});
