import torch
from transformers import ElectraTokenizer, ElectraForQuestionAnswering
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re

max_length = 256 # The maximum length of a feature (question and context)
model_name = "QA/model/" #path to our trained qa model

model=ElectraForQuestionAnswering.from_pretrained(model_name)
tokenizer=ElectraTokenizer.from_pretrained(model_name)


def get_passages(embedded_question):
    # Read the Faiss index from the file
    passages=pd.read_csv('QA/encodedPassages.csv')

    #Query the index
    query_vector = embedded_question
    vector_2d = np.reshape(query_vector, (1, -1))
    dataframe_reduced = passages.iloc[:, :512]
    similarity_scores = cosine_similarity(vector_2d, dataframe_reduced.values)
    similarity_series = pd.Series(similarity_scores[0], index=dataframe_reduced.index, name='Similarity')
    top5_similar_entries = similarity_series.nlargest(10, keep='all')
    
    return top5_similar_entries.index

def get_answers(question,embedded_question):
    '''
    extracts answers from relevent passages
    '''

    doc=pd.read_csv('QA/passages.csv')

    answers=[]
    rel_passages_text=[]
    passages=get_passages(embedded_question)
    for docid in passages:
        context=doc['passage'].iloc[docid]
        inputs=tokenizer.encode_plus(question,
                    context,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=True,
                    return_attention_mask=True)
    
        model.eval()
        with torch.no_grad():
                preds=model(**inputs)
                start=torch.argmax(torch.softmax(preds.start_logits,dim=1)).cpu().item()
                end=torch.argmax(torch.softmax(preds.end_logits,dim=1)).cpu().item()
        inputs.to('cpu')
        answer=tokenizer.decode(inputs['input_ids'][0][start:end+1])


        check_answer=re.search(answer,context)
        if '[CLS]' not in answer and answer!="" and answer!=question and '[SEP]' not in answer and '[PAD]' not in answer and check_answer:
            rel_passages_text.append(context)
            answers.append(answer)
            
        else:
            continue
    if len(answers)==0:
          return None
    return ({'relevant_passages' : rel_passages_text, 'answers' : answers})


def main():
    return None

if __name__=='__main__':
    main()