import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


st.set_page_config(layout='wide')
models = ['Scalar', 'Cosine', 'Jaccard', 'BM25']
rocs = {}
PRs = {}

col1, col2 = st.columns(2)

for model in models:
    rocs[model] = np.load('Eval/ROC_'+model+'.npy')
    PRs[model] = np.load('Eval/PR_'+model+'.npy')

eval = pd.read_csv('Eval//Eval_Vector.csv')
eval = eval.drop([eval.columns[0]], axis=1)
col1.title('Precisions, Recalls and F-scores')
col1.markdown('Vector based')
col1.dataframe(eval)

eval = pd.read_csv('Eval//Eval_Prob.csv')
eval = eval.drop([eval.columns[0]], axis=1)
col1.markdown('Probability based')
col1.dataframe(eval)

col2.title('Curves')

to_display = col2.multiselect(
    'Choose the models to display',
    models, models)

tabs = col2.tabs(['Precision Recall', 'ROC'])

with tabs[0]:
    fig, ax = plt.subplots()

    for roc in to_display:
        ax.plot(PRs[roc][:, 1], PRs[roc][:, 0], label=roc)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    plt.legend()
    st.pyplot(fig)

with tabs[1]:
    fig, ax = plt.subplots()

    for roc in to_display:
        ax.plot(rocs[roc][0], rocs[roc][1], label=roc)

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    plt.legend()

    st.pyplot(fig)