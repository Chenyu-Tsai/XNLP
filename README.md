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
![Arch!](figure/arch.png 'Arch')

Figure above illustrates the overview of our co-training setting. The input **_X_** is a pair of sentences, and the Transformer layer in the middle is a shared layer. We then proceed to multi-task training for specific tasks, in order to learn knowledge that is common to all three tasks and useful to other tasks. After extracting explanations through different methods and models, we take explanations into a human-involved trust evaluation.

# Results
Model|NLI Accuracy(%)|MAX@MRR|SUM@MRR|MAX@F1|SUM@F1|
-|:-:|:-:|:-:|:-:|:-:|
Pretrained|36.0|0.4835|0.5144|0.2202|0.2074|
RTE|57.4|0.5056|0.5203|0.2242|0.2057|
RTE+SP|61.5|0.5005|0.5272|0.2260|0.2125|
RTE+SP+SD|**66.1**|**0.5344**|**0.5685**|**0.2304**|**0.2330**|

>Table 1. Overall results of NLI and explanation extraction.

The results of NLI and explanation extraction are shown in Table 1.
**Pretrained** is the lowest of the four in terms of both NLI accuracy (36.0\%) and explainability metrics since it is not trained with NLI data. 
The performance of the remaining three models are, in descending order, the **single-task (RTE)** model (57.4\%), the **two-tasks (RTE+SP)** model (61.5\%), and the **three-tasks (RTE+SP+SD)** model (66.1\%). It shows that the performance of NLI task can be improved after learning more related knowledge. Whether the increase in performance comes from the model's ability to focus more on key information and improve explainability depends on the variation in the four metrics. 

Model|NLI Accracy(%)|Top3|Top5|Top7|
-|:-:|:-:|:-:|:-:|
Pretrained|36.0|0.3882|0.4970|0.5299|
RTE|57.4|0.3875|0.4950|0.5266|
RTE+SP|61.5|0.4021|0.5117|0.5403|
RTE+SP+SD|**66.1**|**0.4466**|**0.5388**|**0.5730**|
> Table 2. Overall results of NLI and keywords evaluation.

In keywords evaluation we can observe the same trend as the explanation evaluation, which means the model can attend the most informative words in the premise than those less informative.

Model|Entailment(SUM@F1/Corr)|Neutral(SUM@F1/Corr)|Contradiction(SUM@F1/Corr)|
-|:-:|:-:|:-:|
Pretrained|0.5909/-0.03|0.4049-0.29|0.4953/-0.12|
RTE|0.6002/0.11|0.4080/-0.22|0.5049/-0.12|
RTE+SP|0.6100/0.17|0.4124/-0.12|0.5188/-0.09|
RTE+SP+SD|**0.6562/0.30**|**0.4385/-0.07**|**0.5577/-0.05**|
> Table 3. Performances of the three types.

In Table 3., we list the SUM@F1 metric and the Pearson correlation coefficient between correct NLI task prediction and the variation of SUM@F1. In all three entailment relation, we can observe that the correlations coefficient increase with the addition of new tasks, implying that the model predict correctly is correlate with whether the model attend the key span.On pretrained model, we observe through the correlation that although the pretrained model is able to attend the key span, yet it cannot make effective prediction of NLI task from the attended span.

Model|0.1%|1%|10%|100%|
-|:-:|:-:|:-:|:-:|
#Training Data|549|5493|54,936|549,367|
XLNet|65.0|80.9|87.2|90.7|
Proposed Model|69.1|82.9|87.7|91.0|
> Table 4. Performances on the SNLI dataset, reported in Accuracy

In Table 4., we observe that co-training with SP classification task improves the performance and explainability of NLI task. 
We compare whether the SP classification task works on other NLI tasks as well, using following procedure:
    
- use XLNet model both.
- two models were trained, SNLI and SNLI with explainable information.
- truncate RTE-5 example to fit SNLI example sequence length.

Model|Mechanism|Supervised|Exact|Precision|Recall|F1|
-|:-:|:-:|:-:|:-:|:-:|:-:|
Pretrained|Attn@SUM||11.72|58.49|80.82|62.20|
Single RTE|Attn@SUM||12.67|72.98|73.58|67.74|
Single SD|Prediction|V|28.83|72.23|84.23|74.94|
RTE+SP|Attn@SUM||15.67|73.82|74.79|69.04|
RTE+SP+SD|Prediction|V|**30.17**|**77.22**|**84.92**|**77.74**|
RTE+SP+SD|Attn@SUM|V|18.00|76.62|74.85|71.86|
> Table 5. Results of explanation evaluation.

In Table 5., we observe that our co-training model with span detection prediction outperforms other methods, achieving an F-score of 77.74\%. This finding is similar to that of the attention evaluation in previous section. By observing precision and recall, we can find that the performance of the model after multi-task training comes from the improvement of precision, i.e., its prediction of the span is becoming more precise. The difference between 2 methods of 3 multi-task learning model mainly influenced by the recall, which means the SD classifier is tend to extract longer explanation. It is worth noting that pretrained model method has a high recall, implying that it prefers to extract long explanations.

# Case Study

Model|Correctness|F1|Extrated Explanation Span|
-|:-:|:-:|-|
Pretrained@AS|F|0.40|Harry Houdini performed at the Hippodrome at 1120 Avenue of the Americas near 44th Street. Many of the best and most famous performer|
RTE@AS|F|0.40|Harry Houdini performed at the Hippodrome at 1120 Avenue of the Americas near 44th Street. Many of the best and most famous performer|
SD@SD||0.74|Harry Houdini performed|
RTE+SP@AS|F|0.55|Harry Houdini performed at the Hippodrome at 1120 Avenue of the Americas near 44th Street.|
RTE+SP+SD@SD||1.0|Harry Houdini performed at the Hippodrome|
RTE+SP+SD@AS|T|0.74|Harry Houdini performed|
> Table 6. Sample output of the models under different settings.

In Table 6., we observe that the explanation from 3 multi-task learning model SD classifier (RTE+SP+SD@SD) is longer than the explanation extracting from attention score (RTE+SP+SD@AS), this is in line with our observation in the previous section that the recall is higher in the former. The long explanation of the pretrained model also shows the same features of low precision and high recall. 

# Trust Evaluation
![Trust!](figure/Trend.jpg 'Trust')

Figure above shows a trend of changing trust levels over the progress  of  four rounds of betting. In the first round, the most trusted method was RTE+SP+SD@SD(49), followed by Pretrained@AS(37), and the least trusted method was RTE+SP+SD@AS (12). RTE+SP+SD@AS does not change much from the first to the fourth round, but after four rounds of betting, we observe that although RTE+SP+SD@SD starts off with a high level of trust, the participants gradually lose trust in it and switch to Pretrained@AS. In the end, participants choose between the two models, giving them a higher level of trust in the latter.

Through the variation in the results and the analysis through the different explanation characteristics, the following insights can be drawn:
    
- The disadvantage of RTE+SP+SD@SD in terms of gaining trust is that it can initially gain the trust of the participants because it can give short and concise information, therefore, each word in the explanation is very informative, and any missing word will have a significant impact on the entire explanation. RTE+SP+SD@SD's inability to consistently give very accurate and unambiguous explanations will easily lead to a loss of trust in the participant.
- The advantage of Pretrained@AS is that it has a higher fault tolerance than the RTE+SP+SD@SD method. Benefiting from the power of the model it self and attention mechanisms. In downstream NLI task, although the pretrained model does not perform well, it is still able to capture key information, but the range is long and unfocused. Longer explanations, on the other hand, become an advantage because of their longer length. Containing relatively less information per word are relatively less affected by syntactic error in words. Furthermore, it contains key information, which humans can automatically extract from it, and excessively long information can provide some additional contextual information.
- RTE+SP+SD@AS falls somewhere in between, has weaker explanation characteristics than either, regardless of which explanation the participant prefers, so it receives less trust throughout.

# Conclusion
This work investigates the explainability of NLP models from a novel perspective. 
By introducing explanation tasks as auxiliary tasks co-trained with the main task, the model is significantly improved, especially for the scenario with only few training instances. 
We also reveal the human preferences for the explanation extracted by using attention weights. 
The results and the findings of this work provide insights to one of the most attractive topic in the AI community. 