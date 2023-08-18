Demo web application
=====

.. demo_web_application:

Demo illustration
----------------

.. raw:: html
    <img width="900" alt="gui7" src="https://github.com/IINemo/lm-polygraph/assets/21058413/51aa12f7-f996-4257-b1bc-afbec6db4da7">


Start woth Docker
----------------

.. code-block::console
    docker run -p 3001:3001 -it -v $HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub --gpus all mephodybro/polygraph_demo:0.0.17 polygraph_server

The server should be available on `http://localhost:3001`

Original implementation
----------------
The chat GUI is based on the following project: https://github.com/ioanmo226/chatgpt-web-application