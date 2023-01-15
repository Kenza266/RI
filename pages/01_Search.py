import streamlit as st
from Index import Index

st.set_page_config(layout='wide')

index = Index(('DS/index.json', 'DS/inverted.json', 'DS/queries.json', 'DS/ground_truth.csv', 'DS/raw_queries.json', 'DS/raw_docs.json'), preprocessed=True)
st.title('Metrics')

doc = st.number_input('Choose a document', min_value=0, max_value=len(index.index)-1)
st.write(index.raw_docs[str(doc)])
q = st.number_input('Choose a query', min_value=0, max_value=len(list(index.queries.values()))-1)
query = index.queries[str(q)] 
st.write(index.raw_queries[str(q)])

output = {'Scalar product': index.scalar_prod(str(doc), query), 
          'Cosine measure': index.cosine_measure(str(doc), query),
          'Jaccard measure': index.jaccard_measure(str(doc), query)}
st.write(output)

st.markdown("""---""")

st.title('Vector search')
col1, col2 = st.columns(2)
query = col1.text_input('Enter a query')
similarity = col2.selectbox(
    'Similarity',
    ('jaccard', 'cosine', 'scalar'))

result = index.vector_search_per_q(query, metric=similarity)
tabs = st.tabs([i for i, _ in result]) 

for i, tab in enumerate(tabs):
    with tab:
        st.write('Similarity =', result[i][1])
        st.write(index.raw_docs[result[i][0]])

st.markdown("""---""")

st.title('Probability search')
col1, col2 = st.columns(2)
query = col1.text_input('Enter a query', key=1)

result = index.BM25_per_q(query)[0]
tabs = st.tabs([str(i) for i, _ in result]) 

for i, tab in enumerate(tabs):
    with tab:
        st.write('Score =', result[i][1])
        st.write(index.raw_docs[str(result[i][0])])

st.markdown("""---""")

st.title('Boolean')
query = st.text_input('Enter a boolean query')
docs = index.parse_boolean_query(query)
if docs is not None:
    tabs = st.tabs([i for i in docs]) 

    for i, tab in enumerate(tabs):
        with tab:
            st.write(index.raw_docs[docs[i]])
else:
    st.write('Please rewrite the query')