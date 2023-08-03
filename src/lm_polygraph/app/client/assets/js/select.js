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
        modelSelected: 'Bloomz 560M',
        options: toOptions([
            'gpt-4', 'gpt-3.5-turbo',
            'Ensemble', 'Dolly 3b', 'Dolly 7b', 'Dolly 12b',
            'Bloomz 560M', 'Bloomz 3b', 'Bloomz 7b', 'Falcon 7b',
            'Llama 3b', 'Llama 7b', 'Llama 13b', "OPT 2.7b",
            'Flan T5 XL', 'T5 XL NQ', 'BART Large CNN']),
        value: '',
    },
});


Vue.component('treeselect', VueTreeselect.Treeselect);
new Vue({
    el: '#seque',
    data: {
        sequeSelected: ['Maximum Probability'],
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
            } else if (['gpt-4', 'gpt-3.5-turbo'].includes(newModel)) {
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
                    'EP-T-RMI', 'EP-T-EPKL', 'EP-T-EPKL-TU', 'EP-T-Entropy-Top5', 'EP-T-Entropy-Top10',
                    'EP-T-Entropy-Top15', 'PE-T-Total-Uncertainty', 'PE-T-Data-Uncertainty',
                    'PE-T-Mutual-Information', 'PE-T-RMI', 'PE-T-EPKL', 'PE-T-EPKL-TU',
                    'PE-T-Entropy-Top5', 'PE-T-Entropy-Top10', 'PE-T-Entropy-Top15']);
            } else if (newType === 'T5') {
                return toOptions(['Maximum Probability', 'Normalized Maximum Probability', 'Perplexity', 'Entropy',
                    'Mutual Information', 'Conditional Mutual Information', 'Attention Entropy',
                    'Attention Recursive Entropy', 'Exponential Attention Entropy',
                    'Exponential Attention Recursive Entropy', 'P(True)', 'P(True) Sampling',
                    'Predictive Entropy Sampling', 'Normalized Predictive Entropy Sampling',
                    'Lexical Similarity',
                    "Eigenvalue Laplacian", "Eccentricity", "Degree Matrix", "Number of Semantic Sets",
                    'Semantic Entropy', 'Adaptive Sampling Predictive Entropy',
                    'Adaptive Sampling Semantic Entropy', 
                    'Mahalanobis Distance', 'Mahalanobis Distance - Encoder', 'RDE', 'RDE - Encoder', 'PPL+MD', 'PPL+MD - Encoder']);
            } else if (newType === 'openai') {
                return toOptions([
                    'Lexical Similarity',
                    "Eigenvalue Laplacian", "Eccentricity", "Degree Matrix", "Number of Semantic Sets",]);
            } else {
                return toOptions(['Maximum Probability', 'Normalized Maximum Probability', 'Perplexity', 'Entropy',
                    'Mutual Information', 'Conditional Mutual Information', 'Attention Entropy',
                    'Attention Recursive Entropy', 'Exponential Attention Entropy',
                    'Exponential Attention Recursive Entropy', 'P(True)', 'P(True) Sampling',
                    'Predictive Entropy Sampling', 'Normalized Predictive Entropy Sampling',
                    'Lexical Similarity',
                    "Eigenvalue Laplacian", "Eccentricity", "Degree Matrix", "Number of Semantic Sets",
                    'Semantic Entropy', 'Adaptive Sampling Predictive Entropy',
                    'Adaptive Sampling Semantic Entropy', 'Mahalanobis Distance', 'RDE', 'PPL+MD']);
            }
        }
    }
});

Vue.component('treeselect', VueTreeselect.Treeselect);
new Vue({
    el: '#tokue',
    data: {
        tokueSelected: ['Maximum Probability'],
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
            } else if (['gpt-4', 'gpt-3.5-turbo'].includes(newModel)) {
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
                return toOptions(['Maximum Probability', 'Normalized Maximum Probability', 'Entropy',
                    'Mutual Information', 'Conditional Mutual Information', 'Attention Entropy',
                    'Attention Recursive Entropy', 'Exponential Attention Entropy',
                    'Exponential Attention Recursive Entropy', 'Semantic Entropy']);
            }
        }
    }
})
