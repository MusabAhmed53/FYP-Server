#!/usr/bin/env python
# coding: utf-8

# In[14]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout , LayerNormalization
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from sentence_transformers import SentenceTransformer,util
from transformers import AutoTokenizer, AutoModel
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras import regularizers
from tensorflow.keras.layers import SimpleRNN, LSTM
from keras.callbacks import CSVLogger
from tensorflow.keras.metrics import MeanSquaredError
import coral_ordinal as coral


# In[15]:


import pandas as pd


tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
tokenizer2 = AutoTokenizer.from_pretrained('llmrails/ember-v1')
model2 = AutoModel.from_pretrained('llmrails/ember-v1')
tokenizer3 = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model3 = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

# In[17]:


tokenizer_arr = []
tokenizer_arr.append(tokenizer)
tokenizer_arr.append(tokenizer2)
tokenizer_arr.append(tokenizer3)
embeddingsmodel_arr = []
embeddingsmodel_arr.append(model)
embeddingsmodel_arr.append(model2)
embeddingsmodel_arr.append(model3)


# In[18]:


def get_embeddings(texts, tokenizer, model):
    encoded_input = tokenizer(texts, padding='max_length', max_length=56, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input, output_hidden_states=True)

    fourth_layer_embeddings = model_output.hidden_states[3]
    output_layer_embeddings = model_output.last_hidden_state

    concatenated_embeddings = torch.cat((fourth_layer_embeddings, output_layer_embeddings), dim=-1)

    normalized_embeddings = F.normalize(concatenated_embeddings, p=2, dim=-1)

    return (normalized_embeddings.numpy(), encoded_input['attention_mask'])


# In[19]:


from torch.utils.data import Dataset, DataLoader


class CustomTextDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]['question'], self.df.iloc[idx]['answer']


def process_in_batches(df, tokenizer, model, batch_size=64):
    # Create a custom dataset
    dataset = CustomTextDataset(df)
    question_mask = []
    answer_mask = []
    # Create a data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    question_embeddings = []
    answer_embeddings = []
    # Iterate over the data loader
    i = 0
    for batch_questions, batch_answers in data_loader:
        print(i)
        i = i + 1
        retval = get_embeddings(batch_questions, tokenizer, model)
        question_embeddings_batch = retval[0]
        print(question_embeddings_batch.shape)
        print(retval[1])
        question_mask.append(retval[1])

        retval = get_embeddings(batch_answers, tokenizer, model)
        answer_embeddings_batch = retval[0]
        answer_mask.append(retval[1])

        question_embeddings.append(question_embeddings_batch)
        answer_embeddings.append(answer_embeddings_batch)
    return question_embeddings, answer_embeddings, question_mask, answer_mask


# In[20]:


model1 = tf.keras.models.load_model('Models1/model_0.keras')
model2 = tf.keras.models.load_model('Models1/model_1.keras')
model3 = tf.keras.models.load_model('Models1/model_2.keras')
model_arr = []
model_arr.append(model1)
model_arr.append(model2)
model_arr.append(model3)


# In[21]:


def data_generator(ques_embeddings, ans_embeddings, ques_mask, ans_mask, batch_size=64):
    while (True):
        for i in range(0, len(ques_embeddings)):
            q_embed_batch = ques_embeddings[i]
            a_embed_batch = ans_embeddings[i]
            q_mask_batch = ques_mask[i]
            a_mask_batch = ans_mask[i]

            yield ((q_embed_batch, a_embed_batch, q_mask_batch, a_mask_batch),)


# In[22]:


"""
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

"""

# In[25]:


import numpy as np
from scipy.stats import mode


def majority_vote_with_avg(preds):
    mode_result = mode(preds, axis=1)

    mode_labels = mode_result.mode.flatten()
    mode_counts = mode_result.count.flatten()

    final_preds = mode_labels.copy()

    for i in range(len(final_preds)):
        if mode_counts[i] == 1:
            final_preds[i] = int(round(np.mean(preds[i])))

    return final_preds


# In[ ]:


from pydantic import BaseModel
from typing import List

from fastapi import FastAPI

app = FastAPI()


# Define the Pydantic model
class StudentQuestion(BaseModel):
    studentName: str
    question: str
    answer: str
    isplag: bool


class PredictionResponse(BaseModel):
    data: list


@app.get('/')
async def read_root():
    return "Hello, FastAPI!"


@app.post('/gradeResponses/')
async def object_upload(student_questions: List[StudentQuestion]):
    global received_df
    global X_test_question_embeddings, X_test_answer_embeddings, X_test_question_mask, X_test_answer_mask
    global test_gen
    global y_final_preds
    try:
        print("here1")
        data = [
            {"studentName": student_question.studentName, "question": student_question.question,
             "answer": student_question.answer, "plag": student_question.isplag}
            for student_question in student_questions
        ]
        received_df = pd.DataFrame(data)

    except Exception as e:
        return {"message": "Error Creating Dataframe", "error": str(e)}
    y_pred_arr = []
    for i in range(0, 3):
        try:
            X_test_question_embeddings, X_test_answer_embeddings, X_test_question_mask, X_test_answer_mask = process_in_batches(
                received_df, tokenizer_arr[0], embeddingsmodel_arr[0])
            print("here2")
        except Exception as e:
            return {"message": "Error Creating Batches", "error": str(e)}
        try:
            test_gen = data_generator(X_test_question_embeddings, X_test_answer_embeddings, X_test_question_mask,
                                      X_test_answer_mask)
            print("here3")
        except Exception as e:
            return {"message": "Error Creating test_gen", "error": str(e)}
        try:
            print("here4")
            prediction_steps = len(received_df) // 64
            if (len(received_df)) % 64 != 0:
                prediction_steps += 1
            y_pred = np.argmax(coral.ordinal_softmax(model_arr[i].predict(test_gen, steps=prediction_steps)), axis=-1)
            y_pred_arr.append(y_pred)
        except Exception as e:
            return {"message": "Error Predicting values", "error": str(e)}
    try:
        y_preds_stack = np.vstack(y_pred_arr)
        print("here5")
        y_preds_stack = y_preds_stack.T
        y_final_preds = majority_vote_with_avg(y_preds_stack)

    except Exception as e:
        return {"message": "Error Ensemble Vote", "error": str(e)}
    try:
        y_final_preds_list = y_final_preds.tolist()
        unique_pairs = received_df[['studentName', 'answer']].values.tolist()
        unique_pairs_json = [
            {
                "studentName1": pair[0],
                "studentName2": pair[1],
                "prediction": pred
            }
            for pair, pred in zip(unique_pairs, y_final_preds_list)
        ]
        print(y_final_preds)
        print(y_final_preds_list)
        return {"unique_trios": unique_pairs_json}

    except Exception as e:
        return {"message": "Error Sending Response", "error": str(e)}

# %%
