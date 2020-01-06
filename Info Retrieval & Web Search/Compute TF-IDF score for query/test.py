import time
import pickle
import project_part1 as project_part1
s = time.time()
s1 = time.time()
documents = {1: 'According to Los Angeles Times, The Boston Globe will be experiencing another recession in 2020. However, The Boston Globe decales it a hoax.',
             2: 'The Washington Post declines the shares of George Washington.',
             3: 'According to Los Angeles Times, the UNSW COMP6714 students should be able to finish project part-1 now.'}
index = project_part1.InvertedIndex()

index.index_documents(documents)


e1 = time.time()
print(index.tf_entities)
print(index.tf_tokens)
## ## Test cases]:

s2 = time.time()
Q = 'Los The Angeles Boston Times Globe Washington Post'
DoE = {'Los Angeles Times':0, 'The Boston Globe':1,'The Washington Post':2, 'Star Tribune':3}

doc_id = 1

## 2. Split the query...
query_splits = index.split_query(Q, DoE)
e2 = time.time()
s3 = time.time()
##3. Compute the max-score
result = index.max_score_query(query_splits, doc_id)
print(result)
e3 = time.time()
e = time.time()
print("---------p1-----------")
print(e1 - s1)

print("---------p2-----------")
print(e2 - s2)
print("---------p3-----------")
print(e3 - s3)
print("---------total-----------")
print(e - s)