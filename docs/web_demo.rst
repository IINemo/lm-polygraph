Demo web application
====================

.. _demo_web_application:

Demo illustration
-----------------

.. _demo_illustration:

.. raw:: html

    <img width="900" alt="gui7" src="https://user-images.githubusercontent.com/21058413/261061587-51aa12f7-f996-4257-b1bc-afbec6db4da7.png">



Start with Docker
-----------------

run this command::

    docker run -p 3001:3001 -it -v $HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub --gpus all mephodybro/polygraph_demo:0.0.17 polygraph_server

The server should be available on `http://localhost:3001`

Demo running in the Jupyter notebook
------------------------------------

Example of running interface from notebook (careful: only `bloomz-560m`, `gpt-3.5-turbo` and `gpt-4` fits default memory limit, other models can be run only with Colab-pro subscription): https://colab.research.google.com/drive/1JS-NG0oqAVQhnpYY-DsoYWhz35reGRVJ?usp=sharing


Original implementation
-----------------------
The chat GUI is based on the following project: https://github.com/ioanmo226/chatgpt-web-application
