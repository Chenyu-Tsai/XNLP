# Human Evaluation
This section contains the task template used to collect data, the collected raw data and the statistic results.

## Question Template
There are 2 tasks we used to collect data. Task 1 is the RTE span annotation, worker need to highlight the span in the premise that can provide sufficient information to determine the relationship between the premise and the hypothesis. Task 2 is trust evaluation, we design a competition that worker have to allocate their bet, the bet is a quantitative indicator of the human trust to the different models.

## Raw Data
The crowd_dataset_1, 2 and 3 are the task1 dataset. Trust evaluation result is stored in json file, which has 
- Response from 100 participants.
- Answers to the question of distinguishing the explanations.
- Bet on each robots of 4 rounds.
- Reason on each bet.
- Number of correct answer.