import pandas as pd
import streamlit as st
from Index import Index

st.set_page_config(layout='wide')

index = Index(('DS/index.json', 'DS/inverted.json', 'DS/queries.json', 'DS/ground_truth.csv', 'DS/raw_queries.json', 'DS/raw_docs.json'), preprocessed=True)

col1, col2 = st.columns(2)

with col1:
    st.title('Index')
    #st.write(index.index)
    doc = st.number_input('Choose a document', min_value=0, max_value=len(index.index)-1)
    out1 = pd.DataFrame(index.index[str(doc)])
    out1 = out1.transpose()
    out1.columns = ['Frequency', 'Weight'] # 'Token', 
    st.dataframe(out1)

with col2:
    st.title('Inverted')
    #st.write(index.inverted)
    token = st.text_input('Enter a token') 
    try:
        out2 = pd.DataFrame(index.get_docs(token, details=True))
        out2 = out2.transpose()
        out2.columns = ['Frequency', 'Weight'] # 'Document', 
        st.dataframe(out2)
    except:
        st.write('Token not found')

st.title('Query')
query = st.text_input('Enter a query')
try:
    details, all = index.get_docs_query(query)
    output1 = pd.DataFrame(details)
    output2 = pd.DataFrame(all)
    output2 = output2.transpose()
    output2.columns = ['Frequency', 'Weight']
    output = pd.concat([output1, output2], axis=1)
    st.dataframe(output)
except:
    pass

st.markdown("""---""")

st.title('Dataset\'s queries')
q = st.number_input('Choose a query', min_value=1, max_value=len(list(index.queries.values())))
query = index.queries[str(q-1)] 

col1, col2 = st.columns(2)
with col1:
    st.markdown('Query')
    st.markdown(index.raw_queries[str(q-1)])
    st.markdown('Tokens')
    st.write(query)
with col2:
    docs = list(index.ground_truth[index.ground_truth['Query'] == q]['Relevent document'])
    if docs:
        st.markdown('Relevent documents')
        st.write(docs)
    else:
        st.markdown('No relevent document')