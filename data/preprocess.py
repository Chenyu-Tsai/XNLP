import xml.etree.ElementTree as ET
import pandas as pd
import argparse

# replace the word in raw RTE dataset
replacement = {"hasn't": 'has not', 
               "couldn't": 'could not', 
               "wasn't": 'was not', 
               "weren't": 'were not', 
               "doesn't": 'does not',
               "don't": 'do not'
              }

# mapping the raw RTE dataset labels
label_mapping = {'ENTAILMENT': 0, 'UNKNOWN': 1, 'CONTRADICTION': 2}

def train_to_csv(file):
    """ extract the raw RTE training dataset to TSV file """
    root = ET.parse(file).getroot()
    text = []
    hypothesis = []
    entailment = []
    

    for type_tag in root.findall('pair'):
    
        e = type_tag.get('entailment')
    
        t = type_tag.find('t').text
        for word, rep in replacement.items():
            t = t.replace(word.lower(), rep)
        t = preprocess_text(t, remove_space=True)

        h = type_tag.find('h').text
        for word, rep in replacement.items():
            h = h.replace(word.lower(), rep)
        h = preprocess_text(h, remove_space=True)
        
        text.append(t)
        hypothesis.append(h)
        entailment.append(label_mapping[e])

    return text, hypothesis, entailment

def test_to_csv(file):
    """ extract the raw RTE testing dataset to TSV file """
    root = ET.parse(file).getroot()
    text = []
    hypothesis = []
    entailment = []
    # attention span
    attention = []


    for type_tag in root.findall('pair'):
    
        e = type_tag.get('entailment')

        t = type_tag.find('t').text
        for word, rep in replacement.items():
            t = t.replace(word.lower(), rep)
        t = preprocess_text(t, remove_space=True)

        h = type_tag.find('h').text
        for word, rep in replacement.items():
            h = h.replace(word.lower(), rep)
        h = preprocess_text(h, remove_space=True)
        
        a = type_tag.find('a').text
        for word, rep in replacement.items():
            a = a.replace(word.lower(), rep)
        a = preprocess_text(a, remove_space=True)

        text.append(t)
        hypothesis.append(h)
        attention.append(a)
        entailment.append(label_mapping[e])
    
    return text, hypothesis, attention, entailment

def SP_union(annotator1, annotator2):
    """ Get the unnion dataset of 2 annotator's labels """
    df1 = pd.read_csv(annotator1)
    df2 = pd.read_csv(annotator2) 

    # Since df1 is the subset of df2, we loop over the df1's id
    for index, row in df2.iterrows():
        idx = row['id']
        # Check if the df1's example is in df2
        if row['id'] in df1['id'].values:
            for col in df1.columns:
                # Either one annotator labeled 'YES', we record it True
                if (df1[df1['id'] == idx][col] == 'YES').bool():
                    df2.loc[index, col] = 'YES'
                else:
                    pass
    
    return df2


def extract_semantic_phenomenons(df_sp, df_train):
    """ Extracting 5 negative phenomenos from the semantic phenomenons file"""

    # 5 Negative phenomenons
    NSP = ['id',
           'neg_disconnect_rel',
           'neg_excl_arg',
           'neg_excl_rel',
           'neg_miss_arg',
           'neg_miss_rel',
    ]

    # Get the NSP columns of the SP's dataframe
    df_sp = df_sp[NSP]
    lst = []
    
    # SP dataset only have 218 subset of the orginal RTE dataset, we have to get these 218 examples
    for idx in df_sp['id'].values:
        # Get the corresponded example by id
        lst.append(df_train.loc[idx-1].values)
    lst = pd.DataFrame(lst, columns=['text_a', 'text_b', 'entail'])
    df = pd.concat([lst, df_sp], axis=1)
    df = df.drop(columns=['id'])

    # Get the labels of 5 categories
    df['neg_disconnect_rel'] = df.neg_disconnect_rel.apply({'YES':1, 'NO': 0}.get)
    df['neg_excl_arg'] = df.neg_excl_arg.apply({'YES':1, 'NO': 0}.get)
    df['neg_excl_rel'] = df.neg_excl_rel.apply({'YES':1, 'NO': 0}.get)
    df['neg_miss_arg'] = df.neg_miss_arg.apply({'YES':1, 'NO': 0}.get)
    df['neg_miss_rel'] = df.neg_miss_rel.apply({'YES':1, 'NO': 0}.get)

    # Combine 5 labels to 1 column
    NEP1 = df['neg_disconnect_rel'].values
    NEP2 = df['neg_excl_arg'].values
    NEP3 = df['neg_excl_rel'].values
    NEP4 = df['neg_miss_arg'].values
    NEP5 = df['neg_miss_rel'].values

    multi = []
    for i in range(len(NEP1)):
        tmp = []
        tmp.append(NEP1[i])
        tmp.append(NEP2[i])
        tmp.append(NEP3[i])
        tmp.append(NEP4[i])
        tmp.append(NEP5[i])
        multi.append(tmp)
    df['labels'] = multi

    return df
def preprocess_text(inputs, remove_space=True):
  if remove_space:
    outputs = ' '.join(inputs.strip().split())
  else:
    outputs = inputs
  outputs = outputs.replace("``", '"').replace("''", '"')

  return outputs

def main():

    text, hypothesis, entailment = train_to_csv('raw/RTE5_train.xml')
    df_train = pd.DataFrame((zip(text[:500], hypothesis[:500], entailment[:500])), columns=['text_a', 'text_b', 'label'])
    df_valid = pd.DataFrame((zip(text[500:], hypothesis[500:], entailment[500:])), columns=['text_a', 'text_b', 'label'])
    df_train.to_csv("RTE5_train.tsv", sep="\t", index=False, encoding="utf_8_sig")
    df_valid.to_csv("RTE5_valid.tsv", sep="\t", index=False, encoding="utf_8_sig")

    text, hypothesis, attention, entailment = test_to_csv('raw/RTE5_test.xml')
    df_test = pd.DataFrame((zip(text, hypothesis, attention, entailment)), columns=['text_a', 'text_b', 'eval_text','label'])
    df_test.to_csv("RTE5_test.tsv", sep="\t", index=False, encoding="utf_8_sig")

    df_sp = SP_union('raw/RTE5_SP1.csv', 'raw/RTE5_SP2.csv')
    df_multi_label = extract_semantic_phenomenons(df_sp, df_train)
    df_multi_label.to_csv("train_multi_label.tsv", sep='\t', index=False, encoding="utf_8_sig")

if __name__ == "__main__":
    main()