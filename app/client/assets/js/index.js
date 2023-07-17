const API_URL = '/';
const converter = new showdown.Converter();
let promptToRetry = null;
let uniqueIdToRetry = null;

const submitButton = document.getElementById('submit-button');
const regenerateResponseButton = document.getElementById('regenerate-response-button');
const promptInput = document.getElementById('prompt-input');
const modelSelect = document.getElementById('model-select');
const ueSelect = document.getElementById('ue-select');
const levelSelect = document.getElementById('level-select');
const responseList = document.getElementById('response-list');
const typeSelect = document.getElementById('ue-type-select');
const ensPathInput = document.getElementById('ensemble-path-input');

const modelSelectCont = document.getElementById('model-select-container');
const ensPathInputCont = document.getElementById('ensemble-path-input-container');

var single_lookup = {
   'token-level': [
        'Maximum Probability',
        'Normalized Maximum Probability',
        'Entropy',
        'Mutual Information',
        'Conditional Mutual Information',
        'Attention Entropy',
        'Attention Recursive Entropy',
        'Exponential Attention Entropy',
        'Exponential Attention Recursive Entropy',
        'Semantic Entropy'
   ],
   'sequence-level': [
        'Maximum Probability',
        'Normalized Maximum Probability',
        'Entropy',
        'Mutual Information',
        'Conditional Mutual Information',
        'Attention Entropy',
        'Attention Recursive Entropy',
        'Exponential Attention Entropy',
        'Exponential Attention Recursive Entropy',
        'P(True)',
        'P(Uncertainty)',
        'Predictive Entropy Sampling',
        'Normalized Predictive Entropy Sampling',
        'Lexical Similarity Rouge-1',
        'Lexical Similarity Rouge-2',
        'Lexical Similarity Rouge-L',
        'Lexical Similarity Rouge-BLEU',
        'Semantic Entropy',
        'Adaptive Sampling Predictive Entropy',
        'Adaptive Sampling Semantic Entropy'
   ]
};

var ensemble_lookup = {
   'token-level': [
       'EP-T-total-uncertainty',
       'EP-T-data-uncertainty',
       'EP-T-mutual-information',
       'EP-T-rmi',
       'EP-T-epkl',
       'EP-T-epkl-tu',
       'EP-T-entropy-top5',
       'EP-T-entropy-top10',
       'EP-T-entropy-top15',
       'PE-T-total-uncertainty',
       'PE-T-data-uncertainty',
       'PE-T-mutual-information',
       'PE-T-rmi',
       'PE-T-epkl',
       'PE-T-epkl-tu',
       'PE-T-entropy-top5',
       'PE-T-entropy-top10',
       'PE-T-entropy-top15'
   ],
   'sequence-level': [
       'EP-S-total-uncertainty',
       'EP-S-rmi',
       'EP-S-rmi-abs',
       'PE-S-total-uncertainty',
       'PE-S-rmi',
       'PE-S-rmi-abs',
   ]
};

function refreshUEMethods() {
    var level = levelSelect.value;
    var ueType = typeSelect.value;
    var lookup

    if (ueType == 'single') {
        lookup = single_lookup
    } else {
        lookup = ensemble_lookup
    }
    
    if (typeof lookup[level] !== "undefined") {
        ueSelect.innerHTML = "";
        for (i = 0; i < lookup[level].length; i++) {
            let option = document.createElement("option");
            option.setAttribute('value', level + ', ' + lookup[level][i]);
            let optionText = document.createTextNode(lookup[level][i]);
            option.appendChild(optionText);
            ueSelect.appendChild(option);
        }
    }
}

levelSelect.addEventListener("change", refreshUEMethods)

typeSelect.addEventListener("change", function() {
    var type = typeSelect.value;

    if (type == 'single') {
        ensPathInputCont.style.display = "none"
        modelSelectCont.style.display = "block"
    } else {
        ensPathInputCont.style.display = "block"
        modelSelectCont.style.display = "none"
    }

    refreshUEMethods();
})

modelSelect.addEventListener("change", function() {
    promptInput.style.display = 'block';
});

ueSelect.addEventListener("change", function() {
    promptInput.style.display = 'block';
});

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


function addResponse(selfFlag, prompt) {
    const uniqueId = generateUniqueId();
    const html = `
            <div class="response-container ${selfFlag ? 'my-question' : 'chatgpt-response'}">
                <img class="avatar-image" src="assets/img/${selfFlag ? 'me' : 'model'}.png" alt="avatar"/>
                <div class="prompt-content" id="${uniqueId}">${prompt}</div>
            </div>
        `
    responseList.insertAdjacentHTML('beforeend', html);
    responseList.scrollTop = responseList.scrollHeight;
    return uniqueId;
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
    regenerateResponseButton.style.display = 'flex';
}

async function regenerateGPTResult() {
    try {
        await getGPTResult(promptToRetry, uniqueIdToRetry)
        regenerateResponseButton.classList.add("loading");
    } finally {
        regenerateResponseButton.classList.remove("loading");
    }
}

// Function to get GPT result
async function getGPTResult(_promptToRetry, _uniqueIdToRetry) {
    // Get the prompt input
    const prompt = _promptToRetry ?? promptInput.textContent;

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
        addResponse(true, `<div>${prompt}</div>`);
    }

    // Get a unique ID for the response element
    const uniqueId = _uniqueIdToRetry ?? addResponse(false);

    // Get the response element
    const responseElement = document.getElementById(uniqueId);

    // Show the loader
    loader(responseElement);

    // Set isGeneratingResponse to true
    isGeneratingResponse = true;

    try {
        const model = modelSelect.value;
        const level = levelSelect.value;
        const type = typeSelect.value;
        const ensPath = ensPathInput.value;

        var ue = Array.prototype.slice.call(document.querySelectorAll('#ue-select option:checked'),0).map(function(v,i,a) {
            return v.value;
        });

        // Send a POST request to the API with the prompt in the request body
        const response = await fetch(API_URL + 'get-prompt-result', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt,
                model,
                ue,
                level,
                type,
                ensPath
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
        regenerateResponseButton.style.display = 'none';
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
regenerateResponseButton.addEventListener("click", () => {
    regenerateGPTResult();
});

document.addEventListener("DOMContentLoaded", function(){
    promptInput.focus();
});
