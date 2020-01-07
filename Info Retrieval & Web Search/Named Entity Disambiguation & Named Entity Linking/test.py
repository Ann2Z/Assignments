import pickle
import project_part2 as project_part2

## Read the data sets...

### Read the Training Data
train_file = './Dev_set_2/train.pickle'
train_mentions = pickle.load(open(train_file, 'rb'))

### Read the Training Labels...
train_label_file = './Dev_set_2/train_labels.pickle'
train_labels = pickle.load(open(train_label_file, 'rb'))

### Read the Dev Data... (For Final Evaluation, we will replace it with the Test Data)
dev_file = './Dev_set_2/dev2.pickle'
dev_mentions = pickle.load(open(dev_file, 'rb'))

### Read the Parsed Entity Candidate Pages...
fname = './Dev_set_2/parsed_candidate_entities.pickle'
parsed_entity_pages = pickle.load(open(fname, 'rb'))

### Read the Mention docs...
mens_docs_file = "./Dev_set_2/men_docs.pickle"
men_docs = pickle.load(open(mens_docs_file, 'rb'))

result = project_part2.disambiguate_mentions(train_mentions, train_labels, dev_mentions, men_docs, parsed_entity_pages)

for key in list(result)[:5]:
    print('KEY: {} \t VAL: {}'.format(key,result[key]))

## We will be using the following function to compute the accuracy...
def compute_accuracy(result, data_labels):
    assert set(list(result.keys())) - set(list(data_labels.keys())) == set()
    TP = 0.0
    for id_ in result.keys():
        if result[id_] == data_labels[id_]['label']:
            TP +=1
    assert len(result) == len(data_labels)
    return TP/len(result)

### Read the Dev Labels... (For Final Evaluation, we will replace it with the Test Data)
dev_label_file = './Dev_set_2/dev2_labels.pickle'
dev_labels = pickle.load(open(dev_label_file, 'rb'))

accuracy = compute_accuracy(result, dev_labels)
print("Accuracy = ", accuracy)