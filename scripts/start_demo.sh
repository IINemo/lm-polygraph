docker run --name lm-polygraph  -p 8881:3001 --memory=30g -v $HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub  --gpus all inemo/lm-polygraph-demo
