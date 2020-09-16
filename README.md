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

# Architecture
![Test!](figure/arch.png 'test')

Figure above illustrates the overview of our co-training setting. The input **_X_** is a pair of sentences, and the Transformer layer in the middle is a shared layer. We then proceed to multi-task training for specific tasks, in order to learn knowledge that is common to all three tasks and useful to other tasks. After extracting explanations through different methods and models, we take explanations into a human-involved trust evaluation.

# Results
Model|NLI Accuracy(%)|MAX@MRR|SUM@MRR|MAX@F1|SUM@F1|
---|---|---|---|---|---|
Pretrained|36.0|0.4835|0.5144|0.2202|0.2074|
RTE|57.4|0.5056|0.5203|0.2242|0.2057|
RTE+SP|61.5|0.5005|0.5272|0.2260|0.2125|
RTE+SP+SD|**66.1**|**0.5344**|**0.5685**|**0.2304**|**0.2330**|

>Table 1. Overall results of NLI and explanation extraction.

The results of NLI and explanation extraction are shown in Table 1.
**Pretrained** is the lowest of the four in terms of both NLI accuracy (36.0\%) and explainability metrics since it is not trained with NLI data. 
The performance of the remaining three models are, in descending order, the **single-task (RTE)** model (57.4\%), the **two-tasks (RTE+SP)** model (61.5\%), and the **three-tasks (RTE+SP+SD)** model (66.1\%). It shows that the performance of NLI task can be improved after learning more related knowledge. Whether the increase in performance comes from the model's ability to focus more on key information and improve explainability depends on the variation in the four metrics. 

Model|NLI Accracy(%)|Top3|Top5|Top7|
-|-|-|-|-|
Pretrained|36.0|0.3882|0.4970|0.5299|
RTE|57.4|0.3875|0.4950|0.5266|
RTE+SP|61.5|0.4021|0.5117|0.5403|
RTE+SP+SD|**66.1**|**0.4466**|**0.5388**|**0.5730**|
> Table 2. Overall results of NLI and keywords evaluation.

In keywords evaluation we can observe the same trend as the explanation evaluation, which means the model can attend the most informative words in the premise than those less informative.

