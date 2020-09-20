# Example
## Entailment

### Premise
When a mosquito drinks the blood an infected person the insect also swallows the malaria parasite which then must incubate and multiply before migrating into the insect's saliva. The process can take weeks. And because mosquitoes are small-bodied and cold-blooded outside temperatures make a big difference in how long it takes before they can transmit the disease. If it happens too slowly the insects die before they can infect anyone. In general the malaria parasite does better at warmer temperatures which is why the disease occurs most often in tropical environments. But with mathematical models Thomas has found that even when conditions are warm highly fluctuating temperatures over the course of a day slow down the rate of parasite incubation and larval development in the mosquito.
### Hypothesis
Warm weather facilitates the spreading of malaria.
### Relation
Enatailment
### Three models answers
#### 3MT with SD
"In general the malaria parasite does better at warmer temperatures"
#### 3MT Attention
"malaria parasite does better at warmer temperatures which is why the disease occurs most often in tropical environments. But with mathematical models Thomas has found that even when conditions are warm"
#### Pretrained Attention
"the disease occurs most often in tropical environments. But with mathematical models Thomas has found that even when conditions are warm highly fluctuating temperatures over the course of a day slow down the rate of parasite"
### Features
3 tasks' span is much short and have sufficient information, pretrained can also lead the hypothesis, but in more complicated context. 
## Unknown
### Premise
I was nearly charged with petty theft for pilfering coffee at the illustrious Hippodrome Building. But lest I be judged too quickly I must convey the sublimity of the fourth floor's coffee machine. Harry Houdini performed at the Hippodrome at 1120 Avenue of the Americas near 44th Street. Many of the best and most famous performers of the time appeared there. It was one of the biggest and most successful theaters of its time capable of accommodating 5200 people.
### Hypothesis
Harry Houdini was a magician.
### Relation
UNKNOWN
### Three models answers
#### 3MT with SD
"Harry Houdini performed at the Hippodrome"
#### 3MT Attention
"Harry Houdini performed"
#### Pretrained Attention
"Harry Houdini performed at the Hippodrome at 1120 Avenue of the Americas near 44th Street. Many of the best and most famous performers"
### Features
3 task SD and attention have short span with sufficient information, while pretrained has a long span with unecessary informations.
## Contradiction
### Premise
Leftist Mauricio Funes of El Salvador's former Marxist rebel FMLN party has won the country's presidential election. He defeated his conservative rival the Arena party's Rodrigo Avila who has admitted defeat. Arena had won every presidential election since the end of El Salvador's civil war 18 years ago. Addressing jubilant supporters Mr Funes said it was the happiest day of his life and the beginning of a new chapter of peace for the country. Branded by his opponents as a puppet of Venezuala's President Hugo Chavez Mr Funes vowed to respect all Salvadorian democratic institutions.
### Hypothesis
Rodrigo Avila has won El Salvador's presidential election.
### Relation
Contradiction
### Three models answers
#### 3MT with SD
"Leftist Mauricio Funes of El Salvador's former Marxist rebel FMLN party has won the country's presidential election."
#### 3MT Attention
"won the country's presidential election."
#### Pretrained Attention
"country's presidential election. He defeated his conservative rival the Arena party's Rodrigo Avila who has admitted defeat. Arena had won every presidential election since the end of El"
### Features
3mtl with SD provide much accurate span, 3mlt attention has no subject, pretrained has no subject and lots of redundant informations.
# Round 1
## Entailment

### Premise
Cancer survivors who were a mean age of 4.6 years when first diagnosed often had more physical visual and hearing disability impairments as complications of treatment but they did not have a higher rate of grade repetition. Indeed the rates between the cancer and non-cancer groups were similar at about 22 per cent. Another positive finding was that there were no significant differences in achievements on Grade 12 provincial exams between the study groups. Lead investigator Mary McBride and radiation oncologist Dr. Karen Goddard said in an interview that radiation treatment has helped improve childhood survival outcomes (to 80 per cent in the past decade) but it is also a leading culprit as far as educational difficulties go.
### Hypothesis
Cancer survivors often suffer from visual impairments after treatment.
### Relation
Entailment
### Three models answers
#### 3MT with SD (A)
"Cancer survivors who were a mean age of 4.6 years when first diagnosed often had more physical visual and hearing disability impairments as complications of treatment"
#### 3MT Attention (C)
"Cancer survivors who were a mean age of 4.6 years when first diagnosed often had more physical visual and hearing disability impairments as complications of treatment but they did not have a higher rate of grade repetition."
#### Pretrained Attention (B)
"survivors who were a mean age of 4.6 years when first diagnosed often had more physical visual"
### Features
This is a example that the span must have long enough length to provide suffcient informations, and the pretrained model's length is the shortest one and provides least informations. We try to evaluate here, if the user can correctly find these phonomenons, or they just choose the longest span.
## Unknown
### Premise
Speaking after he discovered that he would not face criminal charges Mr Green disclosed that the officers who arrested him last November warned him that he could be given the longest possible sentence. "They said 'Do you realise that this offence could lead to life imprisonment?'" Mr Green told BBC Newsnight. I just thought this was absurd. "I assume it's because it's a common law offence therefore because there is no law on the statute book which I was alleged to have broken then presumably there is no set sentence for it."
### Hypothesis
Mr. Green is the shadow immigration minister of the UK.
### Relation
Unknown
### Three models answers
#### 3MT with SD (B)
"Green disclosed that the officers who arrested him last November"
#### 3MT Attention (A)
"he discovered that he would not face criminal charges Mr Green disclosed that the officers who arrested him last November"
#### Pretrained Attention (C)
"Green disclosed that the officers who arrested him last November warned him that he could be given the longest possible sentence. \"They said 'Do you realise that this offence could lead to life imprisonment?'\""
### Features
There is not specific span that provide sufficient information, the common place of each models is that they all concerned about Mr Green with different length of span.
## Contradiction
### Premise
Nepal's 10-year civil war has come to a peaceful conclusion with the signing of a historic accord between Prime Minister Girija Prasad Koirala and Prachanda leader of the Maoist rebel faction that had been fighting for political change. The deal was signed in Kathmandu on Tuesday. The deal would allow the Maoists into the Nepalese government and place both Maoist and government weapons under UN scrutiny. The Maoists had been observing a ceasefire since its declaration more than six months previously. Prachanda said that the peace agreement would end the 238-year old feudal system. He added that his party would work with new responsibility and make new strong Nepal.
### Hypothesis
Girija Prasad Koirala is the leader of a Maoist rebel faction.
### Relation
Contradiction
### Three models answers
#### 3MT with SD (C)
" Prime Minister Girija Prasad Koirala and Prachanda leader of the Maoist rebel faction that had been fighting for political change "
#### 3MT Attention (A)
" Girija Prasad Koirala and Prachanda leader of the Maoist rebel faction "
#### Pretrained Attention (B)
" leader of the Maoist rebel faction "
# Round 2
## Entailment
### Premise
Stock markets around the world particularly those in the United Kingdom have increased dramatically today. This is due to the ongoing events in the financial world including the United States government's announcement that it would buy billions of dollars of bad loans from US banks. The primary UK index the FTSE 100 rose in value by 8.04% which is 392.50 points to above the 5000 mark at 5272.50. The Dow Jones was up 3.10% at 14:58 UTC an increase of 330.89. The Dow Jones currently has a value of 11350.58 points. The NASDAQ index has risen by 2.53% to 2248.63 while the Dax was 5.06% higher than the start of the day as of 14:58 UTC.
### Hypothesis
The NASDAQ index has risen to above the 2000 mark.
### Relation
Entailment
### Three models answers
#### 3MT with SD (A)
"NASDAQ index has risen by 2.53% to 2248.63"
#### 3MT Attention (B)
"NASDAQ index has risen by 2.53% to"
#### Pretrained Attention (C)
"NASDAQ index has risen by 2.53% to 2248.63 while the Dax was 5.06% higher than the start of the day as of 14:58 UTC."
### Features
two 3 tasks methods can get the key information, the pretrained provides wrong span and lots of redundant information. We try to evaluate that whether the user can tell which model perferm better, if not, then why?
## Unknown
### Premise
Mr. Lieberman is part of the new government led by Benjamin Netanyahu and his conservative Likud Party which was sworn in late Tuesday. Mr. Lieberman leads the hawkish party Yisrael Beiteinu, an important partner in the governing coalition and the third largest party in Parliament. Critics of Mr. Lieberman were outraged at the outcome of the recent coalition negotiations which put Yisrael Beiteinu in charge of the Ministry of Public Security which is responsible for the police.
### Hypothesis
Lieberman is the new Foreign Minister of Israel.
### Relation
Unknown
### Three models answers
#### 3MT with SD (B)
"Lieberman leads the hawkish party Yisrael Beiteinu"
#### 3MT Attention (A)
"Lieberman leads the hawkish party Yisrael Beiteinu, an important partner in the governing coalition"
#### Pretrained Attention (C)
"Mr. Lieberman is part of the new government led by Benjamin Netanyahu and his conservative Likud Party which was sworn in late Tuesday. Mr. Lieberman leads the hawkish party Yisrael Beiteinu, an important partner in the governing coalition"
### Features
3 spans all have the key information, while pretrained attention has more redundant informations.
## Contradiction
### Premise
The magazine states that she has been on the set of the soap opera at CBS Television City since March 11. For a while she didn't even tell her husband former Guiding Light actor Michael Tylo where she was really going. She told him that she was just going on "a lot of auditions" and that if she told sooner than she did the news might slip out accidentally. Her husband currently teaches a class at the University of Nevada Las Vegas and she couldn't afford any accidental hints that his students could have picked up on.
### Hypothesis
Michael Tylo is an employee of CBS Television
### Relation
Contradiction
### Three models answers
#### 3MT with SD (B)
" Guiding Light actor Michael Tylo "
#### 3MT Attention (A)
" even tell her husband former Guiding Light actor Michael Tylo "
#### Pretrained Attention (C)
" For a while she didn't even tell her husband former Guiding Light actor Michael Tylo where she was really going. "

# Round 3
## Entailment
### Premise
Friends have said Poplawski was concerned about his weapons being seized during Barack Obama's presidency and friends said he owned several handguns and an AK-47 assault rifle. Three officers killed. Autopsies show Sciullo 37 died of wounds to the head and torso. Mayhle 29 was shot in the head. A witness awakened by two gunshots told investigators of seeing the gunman standing in the home's front doorway and firing two to three shots into one officer who was already down. Sciullo was later found dead in the home's living room and Mayhle near the front stoop police said. A third officer Eric Kelly 41 was killed as he arrived to assist the first two officers. Kelly was in uniform but on his way home when he responded and was gunned down in the street. Kelly's radio call for help summoned other officers including a SWAT team.
### Hypothesis
Sciullo was killed by Poplawski.
### Relation
### Three models answers
#### 3MT with SD (A)
" Three officers killed. Autopsies show Sciullo 37"
#### 3MT Attention (B)
" killed. Autopsies show Sciullo "
#### Pretrained Attention (C)
" Poplawski was concerned about his weapons being seized during Barack Obama's presidency and friends said he owned several handguns and an AK-47 assault rifle. Three officers killed. Autopsies show Sciullo 37 died of wounds to the head and torso. "
## Unknown
### Premise
WEDDING bells are set to ring in Aras an Uachtarain after it emerged that the President's eldest daughter is to tie the knot. Emma McAleese the eldest child of Mary McAleese and her husband Martin has become engaged to her boyfriend of five years Michael O'Connell. A spokesperson for the President has confirmed the news after 27-year-old Emma was spotted wearing a diamond on her engagement finger.
### Hypothesis
Emma McAleese was born in Aras an Uachtarain.
### Relation
Unknown
### Three models answers
#### 3MT with SD (B)
"the eldest child of Mary McAleese"
#### 3MT Attention (C)
"Uachtarain after it emerged that the President's eldest daughter is to tie the knot. Emma McAleese the eldest child of Mary McAleese"
#### Pretrained Attention (A)
"Aras an Uachtarain after it emerged that the President's eldest daughter is to tie the knot. Emma McAleese"
### Features
Same schema as examples above
## Contradiction
### Premise
The International Criminal Court (ICC) in The Hague Netherlands has ordered the arrest of the president of the African country of Sudan Omar Hassan al-Bashir. The warrant was issued by the ICC for seven charges of crimes against humanity and war crimes in the Darfur region of the country.
### Hypothesis
Al-Bashir was arrested for crimes against Egypt.
### Relation
Contradiction
### Three models answers
#### 3MT with SD (C)
" ordered the arrest of the president of the African country of Sudan Omar Hassan al-Bashir. The warrant was issued by the ICC for seven charges of crimes against humanity and war crimes in the Darfur region of the country. "
#### 3MT Attention (A)
" al-Bashir. The warrant was issued by the ICC for seven charges of crimes against humanity"
#### Pretrained Attention (B)
" arrest of the president of the African country of Sudan Omar Hassan al-Bashir. "