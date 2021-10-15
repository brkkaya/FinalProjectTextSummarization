#%%
import pandas as pd
import json
from tqdm import tqdm

tqdm.pandas()
def jsonl_loader(file_name: str) -> pd.DataFrame:
    with open(file_name, "r") as json_file:
        json_list = list(json_file)

    json_array = []
    for json_str in json_list:
        json_array.append(json.loads(json_str))
    return pd.DataFrame(json_array, columns=["text", "summary"])


#%%

df_test = jsonl_loader("tu_test.jsonl")
# %%
def build_vocab(sentences):
    vocab = {}
    for sentence in tqdm(sentences):
        for word in sentence:
            try:
                vocab[word] +=1
            except:
                vocab[word] =1
    return vocab
# %%
sentences = df_test['text'].progress_apply(lambda x: x.split(' ')).values

vocab = build_vocab(sentences=sentences)

print({k:vocab[k] for k in list(vocab)[:5]})
# %%
len(vocab)
# %%
sorted(vocab.values(),reverse=True)[:5]
# %%
for item,value in vocab.items():
    if value == 15168:
        print(item)
# %%
import httplib2
from bs4 import BeautifulSoup ,SoupStrainer
http = httplib2.Http()
status, response = http.request('http://tr.wikipedia.org')

for link in BeautifulSoup(response, parse_only=SoupStrainer('a'),features="html.parser"):
    if link.has_attr('href'):
        print(link['href'])

# %%
import wikipedia

# wikipedia.summary('ubuntu')
# %%
wikipedia.set_lang('tr')
# %%
wikipedia.search('Samsun')
# %%
x = wikipedia.random() 
x
#%%
wikipedia.WikipediaPage('Sadakatsiz').content
# %%

wikipedia.summary(x)
# %%
