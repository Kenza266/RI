import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


st.set_page_config(layout='wide')
models = ['Scalar', 'Cosine', 'Jaccard', 'BM25'] # 'Bool', 'DataMining'
rocs = {}

col1, col2 = st.columns(2)

for model in models:
    rocs[model] = np.load('Eval//'+model+'.npy')

eval = pd.read_csv('Eval//Eval.csv')
eval = eval.drop([eval.columns[0]], axis=1)
col1.title('Precisions, Recalls and F-scores')
col1.dataframe(eval)

col2.title('ROC curves')

to_display = col2.multiselect(
    'Choose the models to display',
    models, models)

fig, ax = plt.subplots()

for roc in to_display:
    plt.plot(rocs[roc][:, 1], rocs[roc][:, 0], label=roc)
plt.legend()

col2.pyplot(fig)