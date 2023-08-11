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
            'BLOOMz 560M',
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
        sequeSelected: ['Lexical Similarity'],
        model: '',
        type: ''
    },
    computed: {
        computedOptions() {
            var newModel = document.getElementById('model').__vue__.modelSelected;
            var newType = 'openai';

            if (this.type != '' && this.type != newType) {
                this.sequeSelected = [];
            }
            this.model = newModel;
            this.type = newType;
            return toOptions(['Lexical Similarity']);
        }
    }
});
