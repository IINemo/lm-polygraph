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
        sequeSelected: [],
        model: ''
    },
    computed: {
        computedOptions() {
            var newModel = document.getElementById('model').__vue__.modelSelected;
            if ((this.model === 'Ensemble') != (newModel === 'Ensemble')) {
                this.sequeSelected = [];
            }
            this.model = newModel;
            if (newModel === 'Ensemble') {
                return toOptions([
                    'EP-S-total-uncertainty', 'EP-S-rmi', 'EP-S-rmi-abs', 'PE-S-total-uncertainty',
                    'PE-S-rmi', 'PE-S-rmi-abs', 'EP-T-total-uncertainty', 'EP-T-data-uncertainty', 'EP-T-mutual-information',
                    'EP-T-rmi', 'EP-T-epkl', 'EP-T-epkl-tu', 'EP-T-entropy-top5', 'EP-T-entropy-top10',
                    'EP-T-entropy-top15', 'PE-T-total-uncertainty', 'PE-T-data-uncertainty',
                    'PE-T-mutual-information', 'PE-T-rmi', 'PE-T-epkl', 'PE-T-epkl-tu',
                    'PE-T-entropy-top5', 'PE-T-entropy-top10', 'PE-T-entropy-top15']);
            } else if (['BART Large CNN', 'Flan T5 XL', 'T5 XL NQ'].includes(newModel)) {
                return toOptions(['Maximum Probability', 'Normalized Maximum Probability', 'Entropy',
                    'Mutual Information', 'Conditional Mutual Information', 'Attention Entropy',
                    'Attention Recursive Entropy', 'Exponential Attention Entropy',
                    'Exponential Attention Recursive Entropy', 'P(True)', 'P(Uncertainty)',
                    'Predictive Entropy Sampling', 'Normalized Predictive Entropy Sampling',
                    'Lexical Similarity Rouge-1', 'Lexical Similarity Rouge-2', 'Lexical Similarity Rouge-L',
                    'Lexical Similarity Rouge-BLEU', 'Semantic Entropy', 'Adaptive Sampling Predictive Entropy',
                    'Adaptive Sampling Semantic Entropy', 
                    'Mahalanobis Distance', 'Mahalanobis Distance - Encoder', 'RDE', 'RDE - Encoder']);
            } else {
                return toOptions(['Maximum Probability', 'Normalized Maximum Probability', 'Entropy',
                    'Mutual Information', 'Conditional Mutual Information', 'Attention Entropy',
                    'Attention Recursive Entropy', 'Exponential Attention Entropy',
                    'Exponential Attention Recursive Entropy', 'P(True)', 'P(Uncertainty)',
                    'Predictive Entropy Sampling', 'Normalized Predictive Entropy Sampling',
                    'Lexical Similarity Rouge-1', 'Lexical Similarity Rouge-2', 'Lexical Similarity Rouge-L',
                    'Lexical Similarity Rouge-BLEU', 'Semantic Entropy', 'Adaptive Sampling Predictive Entropy',
                    'Adaptive Sampling Semantic Entropy', 'Mahalanobis Distance', 'RDE']);
            }
        }
    }
});

Vue.component('treeselect', VueTreeselect.Treeselect);
new Vue({
    el: '#tokue',
    data: {
        tokueSelected: [],
        model: ''
    },
    computed: {
        computedOptions() {
            var newModel = document.getElementById('model').__vue__.modelSelected;
            if ((this.model === 'Ensemble') != (newModel === 'Ensemble')) {
                this.tokueSelected = [];
            }
            this.model = newModel;
            if (newModel === 'Ensemble') {
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
