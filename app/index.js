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
    const {prompt, model, ue} = req.body;

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
            ue: ue,
            messages: [
                { role: "user", content: prompt }
            ]
        })
        let generation = result.data.generation;
        let uncertainty = result.data.uncertainty;

        console.log('Model output:')
        console.log('  generation: ' + generation)
        console.log('  uncertainty: ' + uncertainty);

        let response = '';
        let started_border = false;
        let uncertainty_thrs = 0.2;
        let token_level = ue.includes('token-level');
        for (let i = 0; i < generation.length; i++) {
            if (!token_level || uncertainty[i] >= uncertainty_thrs) {
                response += generation[i];
                continue;
            }
            const white = 100;
            const green = white + Math.floor((255 - white) * uncertainty[i]);
            const red = white + Math.floor((255 - white) * (1 - uncertainty[i]));
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
                        (generation[i + 1][0] == ' ' ||
                         uncertainty[i + 1] >= uncertainty_thrs))) {
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

        if (!token_level) {
            const white = 100;
            const green = white + Math.floor((255 - white) * uncertainty[0]);
            const red = white + Math.floor((255 - white) * (1 - uncertainty[0]));
            const color = 'rgb(' + red + ', ' + green + ', ' + white + ')';
            response += '\n<span style="color: rgb(178, 190, 181)">Uncertainty: </span>';
            response += '<span style="background-color:' + color + ';border-radius:10px;padding: 3px;">';
            response += Math.round(uncertainty[0] * 100) + '%';
            response += '</span>';
        }

        if (generation.length != 0) {
            response += '\n<span style="color: rgb(178, 190, 181); font-size: 12px">Uncertainty estimation: ' + ue + '</span>';
            response += '\n<span style="color: rgb(178, 190, 181); font-size: 12px">Model: ' + model + '</span>';
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
