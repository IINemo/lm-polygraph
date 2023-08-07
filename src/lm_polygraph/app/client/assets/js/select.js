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

Vue.component('treeselect', VueTreeselect.Treeselect);
const modelVue = new Vue({
    el: '#model',
    data: {
        modelSelected: 'Dolly 7b',
        options: toOptions([
            'GPT-4',
            'GPT-3.5-turbo',
            'Dolly 7b',
            'BLOOMz 3b',
            'Llama 2 7b'
        ]),
        value: '',
    },
});


Vue.component('treeselect', VueTreeselect.Treeselect);
new Vue({
    el: '#seque',
    data: {
        sequeSelected: ['Mean Token Entropy'],
        model: '',
        type: ''
    },
    computed: {
        computedOptions() {
            var newModel = document.getElementById('model').__vue__.modelSelected;
            var newType = '';
            if (newModel == 'Ensemble') {
                newType = 'ensemble';
            } else if (['BART Large CNN', 'Flan T5 XL', 'T5 XL NQ'].includes(newModel)) {
                newType = 'T5';
            } else if (['GPT-4', 'GPT-3.5-turbo'].includes(newModel)) {
                newType = 'openai';
            } else {
                newType = 'seq2seq';
            }
            if (this.type != '' && this.type != newType) {
                this.sequeSelected = [];
            }
            this.model = newModel;
            this.type = newType;
            if (newType === 'ensemble') {
                return toOptions([
                    'EP-S-Total-Uncertainty', 'EP-S-RMI', 'PE-S-Total-Uncertainty',
                    'PE-S-RMI', 'EP-T-Total-Uncertainty', 'EP-T-Data-Uncertainty', 'EP-T-Mutual-Information',
                    'EP-T-RMI', 'EP-T-EPKL', 'EP-T-Entropy-Top5', 'EP-T-Entropy-Top10',
                    'EP-T-Entropy-Top15', 'PE-T-Total-Uncertainty', 'PE-T-Data-Uncertainty',
                    'PE-T-Mutual-Information', 'PE-T-RMI', 'PE-T-EPKL',
                    'PE-T-Entropy-Top5', 'PE-T-Entropy-Top10', 'PE-T-Entropy-Top15']);
            } else if (newType === 'T5') {
                return toOptions(['Maximum Sequence Probability', 'Perplexity', 'Mean Token Entropy',
                    'Mean Pointwise Mutual Information', 'Mean Conditional Pointwise Mutual Information',
                    'P(True)', 'P(True) Sampling',
                    'Monte Carlo Sequence Entropy', 'Monte Carlo Normalized Sequence Entropy',
                    'Lexical Similarity',
                    "Eigenvalue Laplacian", "Eccentricity", "Degree Matrix", "Number of Semantic Sets",
                    'Semantic Entropy',
                    'Mahalanobis Distance', 'Mahalanobis Distance - Encoder', 'RDE', 'RDE - Encoder',
                    'HUQ - Decoder', 'HUQ - Encoder']);
            } else if (newType === 'openai') {
                return toOptions([
                    'Lexical Similarity',
                    "Eigenvalue Laplacian", "Eccentricity", "Degree Matrix", "Number of Semantic Sets",]);
            } else {
                return toOptions(['Maximum Sequence Probability', 'Perplexity', 'Mean Token Entropy',
                    'Mean Pointwise Mutual Information', 'Mean Conditional Pointwise Mutual Information',
                    'P(True)', 'P(True) Sampling',
                    'Monte Carlo Sequence Entropy', 'Monte Carlo Normalized Sequence Entropy',
                    'Lexical Similarity',
                    "Eigenvalue Laplacian", "Eccentricity", "Degree Matrix", "Number of Semantic Sets",
                    'Semantic Entropy', 'Mahalanobis Distance', 'RDE', 'HUQ - Decoder']);
            }
        }
    }
});

Vue.component('treeselect', VueTreeselect.Treeselect);
new Vue({
    el: '#tokue',
    data: {
        tokueSelected: ['Token Entropy'],
        model: '',
        type: ''
    },
    computed: {
        computedOptions() {
            var newModel = document.getElementById('model').__vue__.modelSelected;
            var newType = '';
            if (newModel == 'Ensemble') {
                newType = 'ensemble';
            } else if (['BART Large CNN', 'Flan T5 XL', 'T5 XL NQ'].includes(newModel)) {
                newType = 'T5';
            } else if (['GPT-4', 'GPT-3.5-turbo'].includes(newModel)) {
                newType = 'openai';
            } else {
                newType = 'seq2seq';
            }
            if (this.type != '' && this.type != newType) {
                this.tokueSelected = [];
            }
            this.model = newModel;
            this.type = newType;
            if (newType === 'ensemble') {
                return toOptions([]);
            } else if (newType === 'openai') {
                return toOptions([]);
            } else {
                return toOptions(['Maximum Token Probability', 'Token Entropy',
                    'Pointwise Mutual Information', 'Conditional Pointwise Mutual Information',
                    'Semantic Token Entropy']);
            }
        }
    }
})
