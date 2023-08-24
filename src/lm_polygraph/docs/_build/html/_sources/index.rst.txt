.. LM-Polygraph documentation master file, created by
   sphinx-quickstart on Thu Aug 24 05:45:45 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LM-Polygraph's documentation!
========================================

**LM-Polygraph** is a Python library that specialises on uncertainty estimation for LLMs.

Recent advancements in the capabilities of large language models (LLMs) have paved the way for 
a myriad of groundbreaking applications in various fields. However, a significant challenge arises
as these models often ''hallucinate'', i.e., fabricate facts without providing users an apparent 
means to discern the veracity of their statements.
 
Uncertainty estimation (UE) methods could be used
to detect unreliable generations unlocking the safer, more responsible, and more effective use of LLMs in practice.
However, as of now, UE methods for language models are a subject of bleeding-edge research with no 
easy-to-use implementations. In this work, we tackle this issue by introducing LM-Polygraph.
 
This program framework provides a battery of state-of-the-art UE methods for LLMs in text generation tasks 
with unified program interfaces in Python.

Check out the :doc:`usage` section for further information, including several subsections:
   
   How to conduct :ref:`installation` the project.
   Several jupyter notebooks for :ref:`quick_start`.
   To reproduce expiremental results we provide innstructions for benchmarks in :ref:`benchmarks` section.

Web demo illustration colud be found in :doc:`web_demo` section. It is presented in subsection :ref:`demo_illustration`.
Detailed information about modules is available in the :doc:`modules` section.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   web_demo
   modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
