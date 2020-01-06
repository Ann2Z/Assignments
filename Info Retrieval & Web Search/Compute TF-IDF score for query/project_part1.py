## Import Libraries and Modules here...
import spacy
import math


class InvertedIndex:
    def __init__(self):

        ## You should use these variable to store the term frequencies for tokens and entities...
        self.tf_tokens = {}
        self.tf_entities = {}

        ## You should use these variable to store the inverse document frequencies for tokens and entities...
        self.idf_tokens = {}
        self.idf_entities = {}



    ## Your implementation for indexing the documents...
    def index_documents(self, documents):
        doc_len = len(documents)
        nlp = spacy.load("en_core_web_sm")
        for doc_id in documents:
            doc = nlp(documents[doc_id])
            # build tf_ent
            for ent in doc.ents:
                ent_text = ent.text
                if ent_text not in self.tf_entities:
                    # the entity occurs for the first time
                    self.tf_entities[ent_text] = {doc_id: 1}
                else:
                    if doc_id not in self.tf_entities[ent_text]:
                        # the entity occurs in the current doc for the first time
                        self.tf_entities[ent_text][doc_id] = 1
                    else:
                        # accumulate the freq of the ent
                        self.tf_entities[ent_text][doc_id] = self.tf_entities[ent_text][doc_id] + 1
            # build tf_token
            for token in doc:
                if token.is_stop is False and token.is_punct is False:
                    token_text = token.text
                    if token_text not in self.tf_tokens:
                        # the token occurs for the first time
                        self.tf_tokens[token_text] = {doc_id: 1}
                    else:
                        if doc_id not in self.tf_tokens[token_text]:
                            # the token occurs in the current doc for the first time
                            self.tf_tokens[token_text][doc_id] = 1
                        else:
                            # accumulate the freq of the token
                            self.tf_tokens[token_text][doc_id] = self.tf_tokens[token_text][doc_id] + 1

        # remove those tokens occur in simple-word entities
        for ent, doc_cont in self.tf_entities.items():
            split_ent = ent.split()
            if len(split_ent) == 1:
                # remove
                if ent in self.tf_tokens:
                    for doc_id, times in doc_cont.items():
                        if doc_id in self.tf_tokens[ent]:
                            self.tf_tokens[ent][doc_id] = self.tf_tokens[ent][doc_id] - times
                            if self.tf_tokens[ent][doc_id] == 0:
                                self.tf_tokens[ent].pop(doc_id)
                    if self.tf_tokens[ent] == {}:
                        # no more item in the dict
                        self.tf_tokens.pop(ent)

        # build the idf of tokens and entities
        for token, doc_cont in self.tf_tokens.items():
            self.idf_tokens[token] = 1.0 + math.log(doc_len/(1.0 + len(doc_cont)))
        for ent, doc_cont in self.tf_entities.items():
            self.idf_entities[ent] = 1.0 + math.log(doc_len/(1.0 + len(doc_cont)))

        return



    ## Your implementation to split the query to tokens and entities...
    def split_query(self, Q, DoE):
        query = Q.split()

        # find all the ent in the query
        found_ent = []
        for ent in DoE.keys():
            split_ent = ent.split()
            index = 0
            count = 0
            for word in split_ent:
                for q in query[index:]:
                    if q == word:
                        count = count + 1
                        index = query.index(q) + 1
                        break
            if count == len(split_ent):
                # found the ent in the query
                found_ent.append(ent)

        # build up all subsets of ent
        query_splits = {}
        x = 0
        found_ent_len = len(found_ent)
        for i in range(1 << found_ent_len):
            subset = []
            for j in range(found_ent_len):
                if i & (1 << j):
                    subset.append(found_ent[j])
            # check if the subset is qualified
            temp = query.copy()
            qualified = True
            for ent in subset:
                split_ent = ent.split()
                for word in split_ent:
                    if word in temp:
                        temp.remove(word)
                    else:
                        qualified = False
                        break
                if qualified is False:
                    break
            if qualified is True:
                # add the qualified subset into output
                query_splits[x] = {}
                query_splits[x]['tokens'] = temp.copy()
                query_splits[x]['entities'] = subset.copy()
                x = x + 1
        return query_splits



    ## Your implementation to return the max score among all the query splits...
    def max_score_query(self, query_splits, doc_id):
        score = []
        for splits in query_splits.values():
            s1 = 0
            s2 = 0
            for ent in splits['entities']:
                if ent in self.tf_entities and doc_id in self.tf_entities[ent]:
                    s1 = s1 + (1.0 + math.log(self.tf_entities[ent][doc_id])) * self.idf_entities[ent]
            for token in splits['tokens']:
                if token in self.tf_tokens and doc_id in self.tf_tokens[token]:
                    s2 = s2 + (1.0 + math.log(1.0 + math.log(self.tf_tokens[token][doc_id]))) * self.idf_tokens[token]
            score.append((s1 + 0.4 * s2, splits))
        max_score = -9999
        max_t = (0, 0)
        for split in score:
            if split[0] > max_score:
                max_score = split[0]
                max_t = split

        return max_t               ## Output should be a tuple (max_score, {'tokens': [...], 'entities': [...]})

