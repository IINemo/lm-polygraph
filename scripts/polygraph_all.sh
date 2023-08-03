#!/bin/bash
python -m lm_polygraph.app.service &
cd /app/src/lm_polygraph/app
node index.js &
wait

