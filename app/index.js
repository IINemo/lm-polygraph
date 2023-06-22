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
        let response = '';
        for (let i = 0; i < generation.length; i++) {
            const white = 100;
            const green = white + Math.floor((255 - white) * uncertainty[i]);
            const red = white + Math.floor((255 - white) * (1 - uncertainty[i]));
            const color = 'rgb(' + red + ', ' + green + ', ' + white + ')';
            if (uncertainty[i] >= 0.7) {
                response += generation[i];
            } else {
                response += '<span style="background-color:' + color + ';border-radius:10px;padding: 5px;">';
                response += generation[i].trim();
                response += '</span>';
            }
        }
        if (generation.length != 0) {
            response += '\n<span style="color: rgb(178, 190, 181); font-size: 12px">Uncertainty estimation: ' + ue + '</span>';
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
