import os
from collections import Counter
import time
import numpy as np
#import nltk
from nltk.stem.porter import PorterStemmer
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk import word_tokenize
from nltk.corpus import stopwords
data = dict()
start = time.clock()
for root, dirs, files in os.walk("20_newsgroups"):  
    for d in dirs:
        data[d]=[]
        for f in (os.scandir("20_newsgroups//"+d)):
            data[d].append(f.name)
            
train =dict()
test =dict()            
for d in data.keys():
    
    l = int(len(data[d])*0.5)    
    train[d] = data[d][0:l]
    test[d] = data[d][l:2*l]
words =[]
tr_words =dict()    
words_count = {}
for i in train:
    tr_words[i]=[]
    for j in train[i]:
        value = open("20_newsgroups//"+i +"//"+j).read().replace("\n"," ")
        for k in ",.<>/?'\";:|\}]{[+=_-)(*&^%$#@!":
            value.replace(k," ")    
        tr_words[i].extend(word_tokenize(value))
    words_count[i] = Counter(tr_words[i])
    words.extend(tr_words[i])  
    print(i)
words = set(words)

#words = [x for x in words if not any(c.isdigit() for c in x)]
#words = [x for x in words if x.isalpha() and len(x) > 3 and len(x) <15 and x not in stopwords.words('english')]

print(len(words))
words = list(words)
dictionary = {}
doc_num = 1
for i in words: 
    dictionary[i] = {}
    for j in tr_words: 
        if i in words_count[j]:
            dictionary[i][j] = words_count[j][i]
        else:
            dictionary[i][j] = 0.1
        dictionary[i][j]/=len(tr_words[j])
        
    print("Class",i)
    
prob = {}
total= 0
for i in train:
    prob[i] = len(train[i])
    total += len(train[i])
for j in prob:
    prob[j] = prob[j]/total



stemmer = PorterStemmer()
doc_prob = {}
acc = 0
c = 0
for d in test:
    for j in test[d]:
        
        value = open("20_newsgroups//"+d +"//"+j).read()
        test_words = word_tokenize(value)
        test_words = set(test_words)
        p = {}
        for k in test:
            p[k] = prob[k]
            for w in test_words:
                if w in dictionary:
                    p[k] = p[k]* dictionary[w][k]*1000
                else:
                    p[k] = p[k]* 0.1/len(tr_words[d])*1000
        max_keys = [k for k, v in p.items() if v == max(p.values())]
        acc += (max_keys[0]==d)
        c+=1
        print("Class",d)
      
print("Accuracy: ", acc*100/c)    
print("Excecution time:",time.clock()-start)
    
    


    
        
