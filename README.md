# Introduction
**XNLP** is a research that approaches the explainability of NLP models from a different perspective. We investigate the interplay between a target prediction task and its explainability by introducing explanation as auxiliary tasks.

The fundamental NLP task, natural language inference (NLI), is taken as our target task since it essentially evaluates the capability of a learning based model for general natural language understanding.

We extract the explainable information by using the attention weights of the prediction for the target task and also extract the meta-explainable information by using the attention weights of the prediction for explanation. We further compare the explainable information with the meta-explainable information, revealing how the NLP model works in depth.

In addition to the evaluation of model explainability, we introduce a novel experiment that demonstrates the preference of explanation style to human. We recruit annoatators from the crowdsourcing platform, Amazon Mechanical Turk, to participate in a trust evaluation.

The contributions of this work are threefold as follows:

1. We investigate the interplay between general natural language understanding task and its explanation, showing the performance of NLI models can be enhanced by co-training with explanation tasks.

2. Beyond explainability, we explore the meta-explainability of NLP models.

3. We study the human preferences of explanations provided by models, giving an insight for introducing explanation models in real-world applications.

For a detailed description of technical details and experimental results, please refer to our paper(Traditional Chinese Version):

[Modeling Meta-Explainability of Natural Language Inference](http://thesis.lib.nccu.edu.tw/cgi-bin/gs32/gsweb.cgi/ccd=6ubRHU/record?r1=3&h1=0)

# Results
