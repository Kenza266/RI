import json
import math
import re

import nltk
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer, WordNetLemmatizer
from rank_bm25 import BM25Okapi
from tqdm import tqdm


class Index():
    def __init__(self, docs, preprocessed=False):
        if preprocessed:
            index, inverted, queries, ground_truth, raw_queries, raw_docs = docs
            with open(queries) as f:
                queries = json.load(f) 
            ground_truth = pd.read_csv(ground_truth, sep=',')
            with open(index) as f:
                index = json.load(f)
            with open(inverted) as f:
                inverted = json.load(f)
            with open(raw_queries) as f:
                raw_queries = json.load(f)
            with open(raw_docs) as f:
                raw_docs = json.load(f)
            self.index = index
            self.inverted = inverted 
            self.queries = queries
            self.ground_truth = ground_truth
            self.raw_queries = raw_queries
            self.raw_docs = raw_docs
            queries = np.unique(self.ground_truth['Query'])
            self.relevent_docs = [list(self.ground_truth[self.ground_truth['Query'] == q]['Relevent document']) for q in queries]
        else:
            self.docs = docs
        self.wnl = WordNetLemmatizer()
        self.port = PorterStemmer()        

    def filter(self, token):
        t = self.port.stem(token)
        return self.wnl.lemmatize(t)

    def tokenize(self, regex='(?:[A-Za-z]\.)+|\d+(?:\.\d+)?%?|\w+(?:\-\w+)*'):
        regex = nltk.RegexpTokenizer(regex) 
        self.tokens_lists = [regex.tokenize(txt) for txt in self.docs]
        empty_words = nltk.corpus.stopwords.words('english')
        self.tokens_lists = [[token.lower() for token in tokens if token not in empty_words] for tokens in tqdm(self.tokens_lists)]
        self.tokens_lists = [[self.filter(t) for t in tokens_list] for tokens_list in tqdm(self.tokens_lists)] 

    def get_freq(self):
        self.frequencies = []
        for tokens in tqdm(self.tokens_lists):
            dict = {}
            for token in tokens:
                dict[token] = (dict[token] if token in dict.keys() else 0) + 1
            self.frequencies.append(dict) 

    def get_weights(self):
        max = [np.max(list(d.values())) for d in self.frequencies]
        self.weights = []
        for i in tqdm(range(len(max))):
            d = {}
            for k, v in self.frequencies[i].items():
                d[k] = round((v/max[i]) * np.log10(len(max)/len(self.get_docs(k, preprocessed=True))+1), 2)
            self.weights.append(d)
        
    def combine(self, origin):
        out = {}
        sets = set()
        for o in origin:
            sets = sets | set(o)
        frequencies = [{k: origin[i].get(k) if origin[i].get(k) else  0 for k in sets} for i in range(len(origin))]
        for freq, d in tqdm(zip(frequencies, range(len(frequencies)))):
            for k, v in freq.items():
                out[(k, d)] = v
        return out

    def process(self):
        print('Tokenizing...')
        self.tokenize()
        print('Getting frequencies...')
        self.get_freq()
        print('Combining...')
        self.all_frequencies = self.combine(self.frequencies) 
        self.get_inverted_f()
        print('Getting weights...')
        self.get_weights()
        print('Combining...')
        self.all_weights = self.combine(self.weights) 
        self.get_index()
        self.get_inverted()
        
    def get_index(self):
        index = {}
        for doc, (w, f) in enumerate(zip(self.weights, self.frequencies)):
            d = {}
            for token in list(w.keys()):
                d[token] = (f[token], w[token])
            index[doc] = d
        self.index = index

    def get_inverted_f(self):
        inverted = {}
        for doc, f in enumerate(self.frequencies):
            for token in list(f.keys()):
                if token not in inverted.keys():
                    inverted[token] = {}
                inverted[token][doc] = f[token] 
        self.inverted = inverted 

    def get_inverted(self):
        inverted = {}
        for doc, (w, f) in enumerate(zip(self.weights, self.frequencies)):
            for token in list(w.keys()):
                if token not in inverted.keys():
                    inverted[token] = {}
                inverted[token][doc] = (f[token], w[token])
        self.inverted = inverted
        
    def get_docs(self, token, details=False, preprocessed=False):
        if not preprocessed:
            token = self.filter(token.lower())
        if details:
            return self.inverted[token]
        try:
            return list(self.inverted[token].keys())
        except:
            return []

    def get_weight(self, doc, token, preprocessed=False):
        if not preprocessed:
            token = self.filter(token.lower())
        try:
            return self.index[doc][token][1]
        except:
            return 0

    def get_frequen(self, doc, token, preprocessed=False):
        if not preprocessed:
            token = self.filter(token.lower())
        try:
            return self.index[doc][token][0]
        except:
            return 0

    def get_frequency(self, doc, token, preprocessed=False):
        if not preprocessed:
            token = self.filter(token.lower())
        try:
            return self.index[doc][token][0]
        except:
            return 0

    def get_docs_query(self, query):
        tokens = [token for token in query.split()]
        all = {}
        details = {}
        for token in tokens:
            docs = self.get_docs(token)
            for d in docs:
                f, w = self.get_frequency(d, token), self.get_weight(d, token)
                if d not in all.keys():
                    all[d] = [f, w] 
                else:
                    all[d] = [all[d][0] + f, round(all[d][1] + w, 2)]
            details[token] = {d: [f, w] for d in docs}
        return details, all

    def scalar_prod(self, n_doc, query):
        result = 0
        for token in query:
            result += self.get_weight(n_doc, token)
        return result
    
    def cosine_measure(self, n_doc, query):
        w = [self.get_weight(n_doc, token, preprocessed=True) for token in list(self.index[n_doc].keys())]
        result = np.sqrt(len(query)) * np.sqrt(np.dot(w, w))
        return self.scalar_prod(n_doc, query) / result 

    def jaccard_measure(self, n_doc, query):
        w = [self.get_weight(n_doc, token, preprocessed=True) for token in list(self.index[n_doc].keys())]
        result = len(query) + np.dot(w, w) - self.scalar_prod(n_doc, query) 
        return self.scalar_prod(n_doc, query) / result

    def vector_search(self, max_docs=50, metric='scalar'):
        queries = np.unique(self.ground_truth['Query'])
        predicted = {} 
        if metric == 'scalar':
            metric = self.scalar_prod
        elif metric == 'cosine':
            metric = self.cosine_measure
        elif metric == 'jaccard':
            metric = self.jaccard_measure
        for q in tqdm(queries):
            pred = {} 
            query = self.queries[str(q-1)] 
            for doc, _ in self.index.items():
                pred[doc] = metric(doc, query)
            pred = dict(sorted(pred.items(), key=lambda x: x[1], reverse=True))
            predicted[q] = [p for p in list(pred.items())[:max_docs]] 
        return predicted

    def tokenize_q(self, query, regex='(?:[A-Za-z]\.)+|\d+(?:\.\d+)?%?|\w+(?:\-\w+)*'):
        regex = nltk.RegexpTokenizer(regex) 
        tokens_list = regex.tokenize(query)
        tokens_list = [self.filter(t) for t in tokens_list] 
        empty_words = nltk.corpus.stopwords.words('english')
        tokens_list = [token.lower() for token in tokens_list if token not in empty_words]
        return tokens_list

    def vector_search_per_q(self, query, max_docs=10, metric='scalar'):
        query = self.tokenize_q(query)
        if metric == 'scalar':
            metric = self.scalar_prod
        elif metric == 'cosine':
            metric = self.cosine_measure
        elif metric == 'jaccard':
            metric = self.jaccard_measure
        pred = {}
        for doc, _ in self.index.items():
            pred[doc] = metric(doc, query)
        pred = dict(sorted(pred.items(), key=lambda x: x[1], reverse=True))
        predicted = [p for p in list(pred.items())[:max_docs]] 
        return predicted

    def PR(self, pred, relevent):
        precisions = []
        recalls = []
        for p, r in zip(pred, relevent):
            if type(p[0]) == str:
                r = [str(i) for i in r]
            TP = len(set(p) & set(r))
            precisions.append(TP/len(p))
            recalls.append(TP/len(r)) 
        return precisions, recalls, np.mean(precisions), np.mean(recalls)

    def accuracy(self, pred, relevent):
        acc = []
        for p, r in zip(pred, relevent):
            if type(p[0]) == str:
                r = [str(i) for i in r]
            TP = len(set(p) & set((r)))
            acc.append(TP/len(r))
        return acc, np.mean(acc)

    def BM25_per_doc(self, n_doc, doc, query, avgdl, N, b=0.75, k1=1.2):
        score = 0
        for token in query:
            try:
                temp = self.get_docs(token, preprocessed=True)
            except:
                temp = []
            tf = self.get_weight(n_doc, token, preprocessed=True)
            dl = len(doc)
            n = len(temp)
            idf = math.log((N - n + 0.5) / (n + 0.5))
            w = idf * (tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl)))
            score += w
        return score

    def BM25(self):
        pred = []
        queries = np.unique(self.ground_truth['Query'])
        N = len(self.index)
        avgdl = sum([len(t) for _, t in self.index.items()]) / N
        for q in tqdm(queries):
            query = self.queries[str(q-1)] 
            scores = []
            for doc, tokens in self.index.items():
                scores.append(self.BM25_per_doc(doc, tokens, query, avgdl, N))
            pred.append(sorted(zip(range(N), scores), key=lambda x: x[1], reverse=True))
        return pred

    def BM25_per_q(self, query):
        query = self.tokenize_q(query)
        N = len(self.index)
        pred = []
        avgdl = sum([len(t) for _, t in self.index.items()]) / N
        scores = []
        for doc, tokens in self.index.items():
            scores.append(self.BM25_per_doc(doc, tokens, query, avgdl, N))
        pred.append(sorted(zip(range(N), scores), key=lambda x: x[1], reverse=True))
        return pred 
    
    def evaluate(self, stack):
        # print(stack)
        if len(stack) == 1:
            return stack[0]

        if '(' in stack:
            start = stack.index('(')
            jump = 0
            for i in range(start+1, len(stack)):
                if stack[i] == '(':
                    jump += 1
                elif stack[i] == ')':
                    if jump == 0:
                        end = i+1
                        break
                    else:
                        jump -= 1
            res = self.evaluate(stack[start+1: end-1])
            stack[start] = res
            for i in range(start, end-1):
                stack.pop(start+1)
            return self.evaluate(stack)
            
        if 'not' in stack:
            op = stack.index('not')
            right = stack.pop(op + 1) 
            stack[op] = [str(i) for i in list(range(len(self.index))) if str(i) not in right] 
            return self.evaluate(stack)

        if 'and' in stack:
            op = stack.index('and')
            left = stack.pop(op - 1)
            right = stack.pop(op)
            stack[op-1] = list(set(left).intersection(right))
            return self.evaluate(stack)

        if 'or' in stack:
            op = stack.index('or')
            left = stack.pop(op - 1)
            right = stack.pop(op)
            stack[op-1] = list(set(left).union(right))
            return self.evaluate(stack)

    def parse_boolean_query(self, query):
        query = query.lower()
        pattern = re.compile(r'(\(|\)|\w+|and|or|not)')
        stack = []
        parsed = pattern.findall(query)
        
        for token in parsed:
            if token in ['(', ')', 'or', 'and', 'not']:
                stack.append(token)
            else:
                stack.append(self.get_docs(token))

        result = self.evaluate(stack)
        return result