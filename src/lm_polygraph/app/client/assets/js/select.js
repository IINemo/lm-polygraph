/*
This is implementation for method selector and sequence uncertainty selector
It works currently by you just importing the file and praying there are treebank elements with #model and #seque id
 */
function toOptions(list) {
    var listOfObjects = [];
    for (var i = 0; i < list.length; i++) {
        var obj = {
            id: list[i],
            label: list[i]
        };
        listOfObjects.push(obj);
    }
    return listOfObjects;
}



function modelType(newModel) {
    var newType;
    if (newModel === 'Ensemble') {
        newType = 'ensemble';
    } else if (['BART Large CNN', 'Flan T5 XL', 'T5 XL NQ'].includes(newModel)) {
        newType = 'T5';
    } else if (['GPT-4', 'GPT-3.5-turbo'].includes(newModel)) {
        newType = 'openai';
    } else {
        newType = 'seq2seq';
    }
    return newType
}


const typeMethods = {
    'ensemble': [
        'EP-S-Total-Uncertainty', 'EP-S-RMI', 'PE-S-Total-Uncertainty',
        'PE-S-RMI', 'EP-T-Total-Uncertainty', 'EP-T-Data-Uncertainty', 'EP-T-Mutual-Information',
        'EP-T-RMI', 'EP-T-EPKL', 'EP-T-Entropy-Top5', 'EP-T-Entropy-Top10',
        'EP-T-Entropy-Top15', 'PE-T-Total-Uncertainty', 'PE-T-Data-Uncertainty',
        'PE-T-Mutual-Information', 'PE-T-RMI', 'PE-T-EPKL',
        'PE-T-Entropy-Top5', 'PE-T-Entropy-Top10', 'PE-T-Entropy-Top15'
    ],
    'T5': [
        'Maximum Sequence Probability', 'Perplexity', 'Mean Token Entropy',
        'Mean Pointwise Mutual Information', 'Mean Conditional Pointwise Mutual Information',
        'P(True)', 'P(True) Sampling',
        'Monte Carlo Sequence Entropy', 'Monte Carlo Normalized Sequence Entropy',
        'Lexical Similarity',
        "Eigenvalue Laplacian", "Eccentricity", "Degree Matrix", "Number of Semantic Sets",
        'Semantic Entropy',
        'Mahalanobis Distance', 'Mahalanobis Distance - Encoder', 'RDE', 'RDE - Encoder',
        'HUQ - Decoder', 'HUQ - Encoder'
    ],
    'openai':
    [
        'Lexical Similarity',
        "Eigenvalue Laplacian", "Eccentricity", "Degree Matrix", "Number of Semantic Sets",
    ],
    'seq2seq': [
        'Maximum Sequence Probability', 'Perplexity', 'Mean Token Entropy',
        'Mean Pointwise Mutual Information', 'Mean Conditional Pointwise Mutual Information',
        'P(True)', 'P(True) Sampling',
        'Monte Carlo Sequence Entropy', 'Monte Carlo Normalized Sequence Entropy',
        'Lexical Similarity',
        "Eigenvalue Laplacian", "Eccentricity", "Degree Matrix", "Number of Semantic Sets",
        'Semantic Entropy', 'Mahalanobis Distance', 'RDE', 'HUQ - Decoder'
    ]
}

const allModels = [
    'GPT-4', 'GPT-3.5-turbo',
    'Dolly 3b', 'Dolly 7b', 'Dolly 12b',
    'BLOOMz 560M', 'BLOOMz 3b', 'BLOOMz 7b', 'Falcon 7b',
    'Llama 2 7b', 'Llama 2 13b', 'Vicuna 7b', 'Vicuna 13b',
    'Open Llama 3b', 'Open Llama 7b', 'Open Llama 13b',
    'Flan T5 XL', 'T5 XL NQ', 'BART Large CNN', 'Ensemble'
]


const curatedMethods = {
    'T5': ['Lexical Similarity'],
    'openai': ['Lexical Similarity'],
    'seq2seq': [
        'Maximum Sequence Probability', 'Perplexity', 'Mean Token Entropy',
        'Mean Pointwise Mutual Information', 'Mean Conditional Pointwise Mutual Information',
        'P(True)', 'P(True) Sampling',
        'Monte Carlo Sequence Entropy', 'Monte Carlo Normalized Sequence Entropy',
        'Lexical Similarity',
        "Eigenvalue Laplacian", "Degree Matrix", "Number of Semantic Sets",
        'Semantic Entropy'
    ]
}

const curatedModels = [
    'GPT-4', 'GPT-3.5-turbo', 'Llama 2 7b', 'Vicuna 7b'
]

var defaultPrompt = '';
var defaultModel = 'GPT-4';
var defaultMethod = 'Lexical Similarity';

fetch('./example_requests.json')
    .then((response) => response.json())
    .then((requests) => {
    if (requests.length !== 0) {
        let i = Math.floor(Math.random() * requests.length);
        const promptInput = document.getElementById('prompt-input');
        promptInput.innerHTML = requests[i].prompt;
        defaultModel = requests[i].model;
        defaultMethod = requests[i].method;
        document.getElementById('model').__vue__.modelSelected = defaultModel;
        document.getElementById('seque').__vue__.sequeSelected = defaultMethod;
    }
});

Vue.component('treeselect', VueTreeselect.Treeselect);
new Vue({
    el: '#model',
    data: {
        modelSelected: defaultModel,
        value: '',
        allModels: false,
    },
    computed: {
        computedOptions() {
            if (this.allModels){
                return toOptions(allModels)
            } else {
                return toOptions(curatedModels)
            }
        }
    },
});


Vue.component('treeselect', VueTreeselect.Treeselect);
new Vue({
    el: '#seque',
    data: {
        sequeSelected: defaultMethod,
        model: '',
        type: '',
        allMethods: false
    },
    computed: {
        computedOptions() {
            var newModel = document.getElementById('model').__vue__.modelSelected;
            newType = modelType(newModel)

            if (this.type !== '' && this.type !== newType) {
                if (defaultModel == newModel) {
                    this.sequeSelected = defaultMethod;
                } else {
                    this.sequeSelected = curatedMethods[this.type][0];
                }
            }
            this.model = newModel;
            this.type = newType;

            if (this.allMethods){
                return toOptions(typeMethods[this.type])
            } else {
                return toOptions(curatedMethods[this.type])
            }
        }
    }
});
