import numpy as np
import xgboost as xgb
import math
import spacy


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

    ## Your implementation to return the max score among all the query splits...
    def score(self, query_splits, doc_id):
        s = 0
        for token in query_splits:
            if token in self.tf_tokens and doc_id in self.tf_tokens[token]:
                s = s + (1.0 + math.log(1.0 + math.log(self.tf_tokens[token][doc_id]))) * self.idf_tokens[token]
        return s

    def score_men(self, tokens, entities, doc_id):
        s1 = 0
        s2 = 0
        for ent in entities:
            if ent in self.tf_entities and doc_id in self.tf_entities[ent]:
                s1 = s1 + (1.0 + math.log(self.tf_entities[ent][doc_id])) * self.idf_entities[ent]
        for token in tokens:
            if token in self.tf_tokens and doc_id in self.tf_tokens[token]:
                s2 = s2 + (1.0 + math.log(1.0 + math.log(self.tf_tokens[token][doc_id]))) * self.idf_tokens[token]

        return s1 + 0.2 * s2           ## Output should be a tuple (max_score, {'tokens': [...], 'entities': [...]})


def disambiguate_mentions(train_mentions, train_labels, dev_mentions, men_docs, parsed_entity_pages):
    # build group
    group = []
    # build train label
    labels = []

    # index the men docs
    index = InvertedIndex()
    index.index_documents(men_docs)

    # build feature: tf-idf of candidate entities
    tf_idf_can = []
    # build feature: tf-idf of the Wikipedia description pages for each candidate entity
    tf_idf_men = []
    # build feature: entity length
    lengths = []
    # build feature: amount of NUM in the Wikipedia description pages for each candidate entity
    num_NUM = []

    # for each mention
    for mention_id in range(len(train_mentions)):
        mention_id = mention_id + 1
        group.append(len(train_mentions[mention_id]['candidate_entities']))
        label = train_labels[mention_id]['label']
        length = train_mentions[mention_id]['length']
        doc_title = train_mentions[mention_id]['doc_title']

        # for each candidate entity
        for candidate_entity in train_mentions[mention_id]['candidate_entities']:
            tf_idf_can.append(index.score(candidate_entity.split('_'), doc_title))
            words = []
            ent = []
            n = 0
            for word in parsed_entity_pages[candidate_entity]:
                if word[4] != 'O':
                    ent.append(word[1])
                else:
                    words.append(word[1])
                if word[3] == 'NUM':
                    n = n + 1
            num_NUM.append(n)
            tf_idf_men.append(index.score_men(words, ent, doc_title))
            lengths.append(length)
            if candidate_entity == label:
                labels.append(1)
            else:
                labels.append(0)

    group = np.array(group)
    labels = np.array(labels)
    tf_idf_men = np.array(tf_idf_men)
    tf_idf_can = np.array(tf_idf_can)
    lengths = np.array(lengths)
    num_NUM = np.array(num_NUM)

    # merge features
    features = np.dstack((tf_idf_men, tf_idf_can, lengths, num_NUM)).squeeze()

    # train
    xgb_train = xgb.DMatrix(data=features, label=labels)
    xgb_train.set_group(group)

    # evaluation
    dev_group = []
    dev_tf_idf_can = []
    dev_tf_idf_men = []
    dev_lengths = []
    dev_num_NUM = []

    # for each mention
    for mention_id in range(len(dev_mentions)):
        mention_id = mention_id + 1
        dev_group.append(len(dev_mentions[mention_id]['candidate_entities']))
        length = dev_mentions[mention_id]['length']
        doc_title = dev_mentions[mention_id]['doc_title']

        # for each candidate entity
        for candidate_entity in dev_mentions[mention_id]['candidate_entities']:
            dev_tf_idf_can.append(index.score(candidate_entity.split('_'), doc_title))
            dev_lengths.append(length)
            words = []
            ent = []
            n = 0
            for word in parsed_entity_pages[candidate_entity]:
                if word[4] != 'O':
                    ent.append(word[1])
                else:
                    words.append(word[1])
                if word[3] == 'NUM':
                    n = n + 1
            dev_num_NUM.append(n)
            dev_tf_idf_men.append(index.score_men(words, ent, doc_title))

    dev_group = np.array(dev_group)
    dev_tf_idf_men = np.array(dev_tf_idf_men)
    dev_tf_idf_can = np.array(dev_tf_idf_can)
    dev_lengths = np.array(dev_lengths)
    dev_num_NUM = np.array(dev_num_NUM)

    # merge features
    dev_features = np.dstack((dev_tf_idf_men, dev_tf_idf_can, dev_lengths, dev_num_NUM)).squeeze()

    # train
    xgb_dev = xgb.DMatrix(data=dev_features)
    xgb_dev.set_group(dev_group)

    # Model Training and Prediction
    param = {'max_depth': 8, 'eta': 0.05, 'silent': 1, 'objective': 'rank:pairwise',
             'min_child_weight': 0.01, 'lambda': 100, 'n_estimators': 5000}
    classifier = xgb.train(param, xgb_train, num_boost_round=4900)
    preds = classifier.predict(xgb_dev)

    # find the prediction of dev
    index = 0
    preds_index = []
    for length in dev_group:
        max = np.argmax(preds[index:index + length])
        preds_index.append(max)
        index = index + length

    # build output
    output = {}
    for i in range(len(dev_mentions)):
        output[i + 1] = dev_mentions[i + 1]['candidate_entities'][preds_index[i]]

    return output











