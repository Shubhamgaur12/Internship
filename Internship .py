#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import AutoTokenizer, AutoModel
import torch


# In[2]:


tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')


# In[3]:


sentences = [
    "Three years later, the coffin was still full of Jello.",
    "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
    "The person box was packed with jelly many dozens of months later.",
    "He found a leprechaun in his walnut shell."
]

# initialize dictionary to store tokenized sentences
tokens = {'input_ids': [], 'attention_mask': []}

for sentence in sentences:
    # encode each sentence and append to dictionary
    new_tokens = tokenizer.encode_plus(sentence, max_length=128,
                                       truncation=True, padding='max_length',
                                       return_tensors='pt')
    tokens['input_ids'].append(new_tokens['input_ids'][0])
    tokens['attention_mask'].append(new_tokens['attention_mask'][0])

# reformat list of tensors into single tensor
tokens['input_ids'] = torch.stack(tokens['input_ids'])
tokens['attention_mask'] = torch.stack(tokens['attention_mask'])


# In[4]:


outputs = model(**tokens)
outputs.keys()


# In[5]:


embeddings = outputs.last_hidden_state
embeddings


# In[6]:


embeddings.shape


# In[7]:


attention_mask = tokens['attention_mask']
attention_mask.shape


# In[8]:


mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
mask.shape


# In[9]:


mask


# In[ ]:




