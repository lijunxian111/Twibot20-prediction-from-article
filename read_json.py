# -*- coding: utf-8 -*-
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
import torch.nn as nn
import torch
import numpy as np
import re

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model=RobertaModel.from_pretrained('roberta-base')

#这里不应该用absolute path
df=pd.read_json('D:\PycharmProjects\pythonProject\TwiBot-20_sample.json')
#print(df['tweet'].values)
#del df['ID']
#print(df.keys())
#print(df['profile'].values[0])
#print(df['tweet'].values)

def clean(text):
    string = text
    string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


#desc=[]
#tweet=[]
#线性层,激活函数
fc1=nn.Linear(768,4)
act1=nn.LeakyReLU()
fc2=nn.Linear(768,4)
pfc=nn.Linear(5,4)
rc=nn.Linear(11,4)
act2=nn.LeakyReLU()

def rb(dff):
    desc=[]
    for item in dff['profile'].values:
        encoded_input = tokenizer(item['description'], return_tensors='pt')
        output = model(**encoded_input)
        desc.append(output.pooler_output)
    desc = torch.cat(desc, dim=0)
    return desc

def rt(dff):
    tweet=[]
    for item in dff['tweet'].values:
        ljx=[0.0 for i in range(768)]
        ljx=np.array(ljx)
    #print(type(item))
        count=0.0
        if(type(item)==type([1])):
            for sentence in item:
                sentence=clean(sentence)
                #sentence=sentence.replace('\n,\t','')
                #print(sentence)
                encoded_input = tokenizer(sentence, return_tensors='pt')
                output = model(**encoded_input)
                ljx+=output.pooler_output.squeeze().detach().numpy().astype(np.float32)
                count+=1
    #ljx=np.concatenate(ljx,axis=0)
    #avg=np.average(ljx,axis=0)
    #print(avg)
    #print(ljx)
        if count!=0:
            ljx=ljx/count
    #print(ljx)
            tweet.append(torch.tensor(ljx))
        else:
            tweet.append(torch.zeros(768))
    tweet = torch.stack(tweet, dim=0)
    return tweet

def rpn(dff):
    prop=[]
    for item in dff['profile'].values:
        feature=[]
        feature.append(int(item['followers_count'].replace(' ','')))
        feature.append(int(item['listed_count'].replace(' ', '')))
        feature.append(int(item['favourites_count'].replace(' ', '')))
        feature.append(int(item['statuses_count'].replace(' ', '')))
        feature.append(len(item['screen_name'].replace(' ','')))
        feature=np.array(feature).astype(np.float32)
        ff=feature
        res=(feature-np.mean(feature))/np.std(ff)
        prop.append(torch.tensor(res).unsqueeze(0))
    prop=torch.cat(prop,dim=0)
    return prop

def rpc(dff):
    one_hot=[]
    for item in dff['profile'].values:
        ctaeg=[]
        ctaeg.append(1 if (item['protected'].replace(' ','')=='True') else 0)
        ctaeg.append(1 if (item['geo_enabled'].replace(' ', '') == 'True') else 0)
        ctaeg.append(1 if (item['verified'].replace(' ', '') == 'True') else 0)
        ctaeg.append(1 if (item['contributors_enabled'].replace(' ', '') == 'True') else 0)
        ctaeg.append(1 if (item['is_translator'].replace(' ', '') == 'True') else 0)
        ctaeg.append(1 if (item['is_translation_enabled'].replace(' ', '') == 'True') else 0)
        ctaeg.append(1 if (item['profile_background_tile'].replace(' ', '') == 'True') else 0)
        ctaeg.append(1 if (item['profile_use_background_image'].replace(' ', '') == 'True') else 0)
        ctaeg.append(1 if (item['has_extended_profile'].replace(' ', '') == 'True') else 0)
        ctaeg.append(1 if (item['default_profile'].replace(' ', '') == 'True') else 0)
        ctaeg.append(1 if (item['default_profile_image'].replace(' ', '') == 'True') else 0)
        ctaeg=np.array(ctaeg).astype(np.float32)
        one_hot.append(torch.tensor(ctaeg).unsqueeze(0))
    one_hot=torch.cat(one_hot,dim=0)
    return one_hot


#prop=rpn(df)
#print(prop.shape)

#desc=torch.cat(desc,dim=0)
#tweet=torch.cat(tweet,dim=0)
#print(desc.shape)

#tweet=rt(df)
#print(tweet)

desc=rb(df)
desc=act1(fc1(desc))
tweet=fc2(rt(df).float())
prop=pfc(rpn(df))
cate=act2(rc(rpc(df)))
#print(desc)
#print(tweet)
#con=torch.cat([desc,tweet],dim=1)
#print(con)

#cate=rpc(df)
#print(cate)

#表征向量，dim:100*16

embeds=torch.cat([desc,tweet,prop,cate],dim=1)
print(embeds)

#输入RGCN应该转置一下
