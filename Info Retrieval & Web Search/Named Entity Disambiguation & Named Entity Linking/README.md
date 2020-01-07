Input to the model are as follows
1. *men_docs.pickle*
    A python dictionary of the documents with mentions pre-identified.
2. *parsed_candidate_entities.pickle*
    A dictionary containing textual description for each candidate entity         (pages from Wikipedia). 
3. *train.pickle*
    Training data.
4. *train_labels.pickle*
    Labels corresponding to the training data.
5. *dev.pickle* 
    Development data. 
6. *dev_labels.pickle* 
    Labels corresponding to the dev data.

**TASK**: 

Given a document *men_doc=[w 1 ,w 2 ,...,w Q ]*; mention span within the document = *{m i }*; and a collection of candidate entities for each mention alongwith corresponding entity description pages: *{e i } n i=1*. Use the:
1. Mention
2. Mention's document (i.e., men_doc)
3. Entity description page for each candidate entity

to come up with a learning-to-rank model to rank the candidate entities corresponding to each mention in such a way that the Ground Truth Entity is ranked higher than the false candidates.
