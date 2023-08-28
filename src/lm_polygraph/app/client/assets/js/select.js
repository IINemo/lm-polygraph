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
    'Llama 2 7b', 'Llama 2 13b',
    'Open Llama 3b', 'Open Llama 7b', 'Open Llama 13b',
    'Flan T5 XL', 'T5 XL NQ', 'BART Large CNN', 'Ensemble'
]


const curatedMethods = {
    'T5': [],
    'openai': [],
    'seq2seq': [
        "Maximum Token Probability", "Token Entropy",
        "Pointwise Mutual Information",
        "Conditional Pointwise Mutual Information",
        "Semantic Token Entropy",
    ]
}

const curatedModels = [
    'BLOOMz 560M', 'Llama 2 7b', 'Dolly 3b', 'BLOOMz 3b'
]

const defaultModel = curatedModels[0];

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


//Vue.component('treeselect', VueTreeselect.Treeselect);
//new Vue({
//    el: '#seque',
//    data: {
//        sequeSelected: curatedMethods[modelType(defaultModel)][0],
//        model: '',
//        type: '',
//        allMethods: false
//    },
//    computed: {
//        computedOptions() {
//            var newModel = document.getElementById('model').__vue__.modelSelected;
//            newType = modelType(newModel)
//
//            if (this.type !== '' && this.type !== newType) {
//                this.sequeSelected = curatedMethods[newType][0];
//            }
//            this.model = newModel;
//            this.type = newType;
//
//            if (this.allMethods){
//                return toOptions(typeMethods[this.type])
//            } else {
//                return toOptions(curatedMethods[this.type])
//            }
//        }
//    }
//});

Vue.component('treeselect', VueTreeselect.Treeselect);
new Vue({
    el: '#tokue',
    data: {
        tokueSelected: curatedMethods[modelType(defaultModel)][0],
        model: '',
        type: '',
        allMethods: false
    },
    computed: {
        computedOptions() {
            var newModel = document.getElementById('model').__vue__.modelSelected;
            newType = modelType(newModel)

            if (this.type !== '' && this.type !== newType) {
                this.tokueSelected = curatedMethods[newType][0];
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
