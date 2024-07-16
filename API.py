#!/usr/bin/env python
# coding: utf-8

# In[380]:


import pandas as pd


# In[381]:


from sentence_transformers import SentenceTransformer,util
import torch

#tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')




# In[382]:


import numpy as np



def match_and_merge(df):
    embeddings = model.encode(df['answer'].values, convert_to_tensor=True)
    cos_scores = util.cos_sim(embeddings, embeddings)
    cos_scores = cos_scores.cpu().numpy()
    
    np.fill_diagonal(cos_scores, -np.inf)
    

    similar_indices = np.where(cos_scores > 0.92)
    
    filtered_df = df.iloc[similar_indices[0]]
    
    if len(filtered_df) == 0:
        return pd.DataFrame(columns=['studentName', 'answer', 'matched_answer','plag'])
    
    # Find the index of the most similar answer for each answer
    matches = np.argmax(cos_scores, axis=1)
    
    merged_df = pd.DataFrame({
        'studentName1': df['studentName'].values,
        'studentName2': df.iloc[matches]['studentName'].values,
        'question': df['question'].values,
        'answer1': df['answer'].values,
        'answer2': df.iloc[matches]['answer'].values
    })
    
    
    # Select only the rows with cosine similarity above 0.85
    merged_df = merged_df.loc[df.index.isin(filtered_df.index)]
    
    return merged_df

    


# In[383]:


class StudentQuestion:
    def __init__(self, studentName, question, answer, plag):
        self.studentName = studentName
        self.question = question
        self.answer = answer
        self.plag = plag

# Define your array of StudentQuestion objects
student_questions = [
    StudentQuestion("Alice", "What is the capital of France?", "The capital of France is Paris, which is known for its rich history and cultural heritage.", False),
    StudentQuestion("Bob", "What is the capital of France?", "Paris is the capital of France, famous for its landmarks such as the Eiffel Tower and the Louvre.", False),
    StudentQuestion("Charlie", "What is the capital of France?", "Paris, the capital city of France, is renowned for its art, fashion, and gastronomy.", False),
    StudentQuestion("David", "What is the capital of France?", "The capital of France is Paris, a city celebrated for its museums and cafe culture.", False),
    StudentQuestion("Eve", "What is the capital of France?", "Paris is the capital of France, known for its beautiful architecture and vibrant cultural scene.", False),
    StudentQuestion("Frank", "What is the capital of France?", "Paris, the capital of France, is well-known for its historical monuments and world-famous institutions.", False),
    StudentQuestion("Grace", "What is the capital of France?", "The capital of France is Paris, famous for its art, historical sites, and significant influence on global culture.", False),
    StudentQuestion("Musab", "What is the capital of France?", "The capital of France is Paris, famous for its art, historical sites, and significant influence on global culture.", False)
]

# Convert the array of StudentQuestion objects into a DataFrame
df = pd.DataFrame([
    {"studentName": sq.studentName, "question": sq.question, "answer": sq.answer, "plag": sq.plag}
    for sq in student_questions
])


# In[384]:


#df.head()


# In[385]:


def get_sorted_names(row):
  return tuple(sorted([row['studentName1'], row['studentName2']]))


# In[386]:


#merged_df=match_and_merge(df)


# In[387]:


#merged_df['combined_names'] = merged_df.apply(get_sorted_names, axis=1)


# In[388]:


#merged_df.head()


# In[389]:


#merged_df=merged_df.drop_duplicates(subset=['combined_names'])


# In[390]:


#merged_df.head()


# In[391]:


#result=merged_df[['studentName1', 'studentName2']].values.tolist()
#print(result)


# In[392]:


#df.head()


# In[393]:


import os
import threading
import uvicorn
from pydantic import BaseModel
from typing import List

from fastapi import FastAPI
from fastapi import File,UploadFile,Body


app = FastAPI()

# Define the Pydantic model
class StudentQuestion(BaseModel):
    studentName: str
    question: str
    answer: str
    isplag:bool


@app.get('/')
async def read_root():
    return "Hello, FastAPI!"

@app.post('/upload/')
async def object_upload(student_questions: List[StudentQuestion]):
    global  received_df
    global merged_df2
    try:
        print("here1")
        data = [
            {"studentName": student_question.studentName, "question": student_question.question, "answer": student_question.answer,"plag":student_question.isplag}
            for student_question in student_questions
        ]
        received_df = pd.DataFrame(data)
    
    except Exception as e:
        return {"message": "Error Creating Dataframe", "error": str(e)}
    
    try:
        merged_df2=match_and_merge(received_df)
        print("here2")
    except Exception as e:
        return {"message": "Error in function Match and Merge", "error": str(e)}
    
    try:
        if (len(merged_df2)>0):
            merged_df2['combined_names'] = merged_df2.apply(get_sorted_names, axis=1)
            merged_df2=merged_df2.drop_duplicates(subset=['combined_names'])
            print("here3")
        else:
             return {"message": " No Plaigiarized Responses Found"}
            
    except Exception as e:
        return {"message": "Error in Processing merged_df2", "error": str(e)}
    
    try:
        unique_pairs = merged_df2[['studentName1', 'studentName2']].values.tolist()
        unique_pairs_json = [
            {
                "studentName1": pair[0],
                "studentName2": pair[1]
            }
            for pair in unique_pairs
        ]
        print("here4")
        return {"unique_pairs": unique_pairs_json}
        
    except Exception as e:
        return {"message": "Error in sending data back", "error": str(e)}

        

            

# uvicorn API:app --host 0.0.0.0 --port 8000


# In[ ]:




