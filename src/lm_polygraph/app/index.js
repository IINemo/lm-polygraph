const express = require('express');
const {Configuration, OpenAIApi} = require("openai");
const app = express();
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const multer  = require('multer');
const { v4: uuidv4 } = require('uuid');
require("dotenv").config();
const configuration = new Configuration({
    apiKey: process.env.OPENAI_API_KEY,
});
const openai = new OpenAIApi(configuration);

app.use(cors());
app.use(express.json());
app.use('/', express.static(__dirname + '/client')); // Serves resources from client folder

app.post('/get-prompt-result', async (req, res) => {
    // Get the prompt from the request body
    const {prompt, model, ensembles, tok_ue, seq_ue, temperature, topk, topp, do_sample, num_beams} = req.body;

    // Check if prompt is present in the request
    if (!prompt) {
        // Send a 400 status code and a message indicating that the prompt is missing
        return res.status(400).send({error: 'Prompt is missing in the request'});
    }

    try {
        // Use the OpenAI SDK to create a completion
        // with the given prompt, model and maximum tokens

        const result = await openai.createChatCompletion({
            model: model,
            ensembles: ensembles,
            tok_ue: tok_ue,
            seq_ue: seq_ue,
            parameters: {
                temperature: temperature,
                topk: topk,
                topp: topp,
                do_sample: do_sample,
                num_beams: num_beams,
            },
            messages: [
                { role: "user", content: prompt }
            ]
        })
        let generation = result.data.generation;
        let tok_confidence = result.data.token_confidence;
        let tok_norm = result.data.token_normalization;
        let seq_confidence = result.data.sequence_confidence;
        let seq_norm = result.data.sequence_normalization;

        console.log('Model output:')
        console.log('  generation: ' + generation)
        console.log('  token confidence: ' + tok_confidence);
        console.log('  token normalization: ' + tok_norm);
        console.log('  sequence confidence: ' + seq_confidence);
        console.log('  sequence normalization: ' + seq_norm);

        let response = '';
        let started_border = false;
        let confidence_thrs = 0.2;
        for (let i = 0; i < generation.length; i++) {
            if (tok_norm == 'none' || tok_confidence[i] >= confidence_thrs || tok_confidence.length == 0) {
                response += generation[i];
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
                response += generation[i].trim();
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
            response += '<div style="line-height:1%;"><br></div><span style="color: rgb(77, 65, 74)">Confidence: </span>';
            response += '<span style="' + colorStyle + 'border-radius:10px;padding: 3px;">' + ueStr + '</span>';
        }

        return res.send(response);
    } catch (error) {
        const errorMsg = error.response ? error.response.data.error : `${error}`;
        console.error(errorMsg);
        // Send a 500 status code and the error message as the response
        return res.status(500).send(errorMsg);
    }
});

const port = process.env.PORT || 3001;
app.listen(port, () => console.log(`Listening on port ${port}`));
