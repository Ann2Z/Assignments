Inputs to the model are as follows:
1. Documents (**D**) as a dictionary with *key*: doc_id; *value*: document text
2. Query (**Q**), as a string of words
3. Dictionary of Entities (**DoE**), with *key*: entity; *value*: entity_id

The procedure for computation of the **TF-IDF** score follows following steps:
1. **TF-IDF index construction for Entities and Tokens**
2. **Split the query into lists of Entities and Tokens**
3. **Query Score Computation** 

