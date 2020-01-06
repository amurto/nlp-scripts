import nltk
#nltk.download('popular')
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

sentence = 'European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices'

ne_tree = ne_chunk(pos_tag(word_tokenize(sentence)))

#print(ne_tree)

ex = 'European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices'

def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent

def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent

sent = preprocess(ex)
sent

pattern = 'NP: {<DT>?<JJ>*<NN>}'

cp = nltk.RegexpParser(pattern)
cs = cp.parse(sent)
#print(cs)

NPChunker = nltk.RegexpParser(pattern) 
result = NPChunker.parse(sent)
#result.draw()

from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint

iob_tagged = tree2conlltags(cs)
#pprint(iob_tagged)

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

doc = nlp('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices')
#pprint([(X.text, X.label_) for X in doc.ents])

#pprint([(X, X.ent_iob_, X.ent_type_) for X in doc])

from bs4 import BeautifulSoup
import requests
import re

def url_to_string(url):
    res = requests.get(url)
    html = res.text
    soup = BeautifulSoup(html, 'html5lib')
    for script in soup(["script", "style", 'aside']):
        script.extract()
    return " ".join(re.split(r'[\n\t]+', soup.get_text()))

#ny_bb = url_to_string('https://www.nytimes.com/2018/08/13/us/politics/peter-strzok-fired-fbi.html?hp&action=click&pgtype=Homepage&clickSource=story-heading&module=first-column-region&region=top-news&WT.nav=top-news')
ny_bb=input("Enter string:")
article = nlp(ny_bb)
len(article.ents)

labels = [x.label_ for x in article.ents]
Counter(labels)

items = [x.text for x in article.ents]
Counter(items).most_common(3)

sentences = [x for x in article.sents]
#print(sentences[20])
#print(sentences)

#displacy.render(nlp(str(sentences[20])), jupyter=True, style='ent')

#displacy.render(nlp(str(sentences[20])), style='dep', jupyter = True, options = {'distance': 120})

[(x.orth_,x.pos_, x.lemma_) for x in [y 
                                      for y
                                      #in nlp(str(sentences[20])) 
                                      in nlp(str(sentences))
                                      if not y.is_stop and y.pos_ != 'PUNCT']]

#dict([(str(x), x.label_) for x in nlp(str(sentences[20])).ents])
#dict([(str(x), x.label_) for x in nlp(str(sentences)).ents])

#print([(x, x.ent_iob_, x.ent_type_) for x in sentences[20]])
# a=[]
# for i in range(len(labels)+1):
#     a.append(0)
# print(a)
person_list=[]
norp_list=[]
fac_list=[]
org_list=[]
gpe_list=[]
loc_list=[]
product_list=[]
event_list=[]
work_of_art_list=[]
law_list=[]
language_list=[]
date_list=[]
time_list=[]
percent_list=[]
money_list=[]
quantity_list=[]
ordinal_list=[]
cardinal_list=[]

person,norp,fac,org,gpe,loc,product,event,work_of_art,law,language,date,time,percent,money,quantity,ordinal,cardinal,other=0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
for x in sentences[0]:
    if x.ent_type_=='PERSON':
        person_list.append(x)
        person=person+1
    elif x.ent_type_=='NORP':
        norp_list.append(x)
        norp=norp+1
    elif x.ent_type_=='FAC':
        fac_list.append(x)
        fac=fac+1
    elif x.ent_type_=='ORG':
        org_list.append(x)
        org=org+1
    elif x.ent_type_=='GPE':
        gpe_list.append(x)
        gpe=gpe+1
    elif x.ent_type_=='LOC':
        loc_list.append(x)
        loc=loc+1
    elif x.ent_type_=='PRODUCT':
        product_list.append(x)
        product=product+1
    elif x.ent_type_=='EVENT':
        event_list.append(x)
        event=event+1
    elif x.ent_type_=='WORK_OF_ART':
        work_of_art_list.append(x)
        work_of_art=work_of_art+1
    elif x.ent_type_=='LAW':
        law_list.append(x)
        law=law+1
    elif x.ent_type_=='LANGUAGE':
        language_list.append(x)
        language=language+1  
    elif x.ent_type_=='DATE':
        date_list.append(x)
        date=date+1
    elif x.ent_type_=='TIME':
        time_list.append(x)
        time=time+1
    elif x.ent_type_=='PERCENT':
        percent_list.append(x)
        percent=percent+1
    elif x.ent_type_=='MONEY':
        money_list.append(x)
        money=money+1
    elif x.ent_type_=='QUANTITY':
        quantity_list.append(x)
        quantity=quantity+1
    elif x.ent_type_=='ORDINAL':
        ordinal_list.append(x)
        ordinal=ordinal+1
    elif x.ent_type_=='CARDINAL':
        cardinal_list.append(x)
        cardinal=cardinal+1
    else:
        other=other+1

# for i in sentences[0]:
#     if a[i]!=0:
#         print("Persons Count:",p)
if(person!=0):print("Persons Count:",person, "| List of Persons:", person_list)
if(norp!=0):print("NORP Count:",norp, "| List of NORP:", norp_list)
if(fac!=0):print("FAC Count:",fac,"| List of FAC:", fac_list)
if(org!=0):print("ORG Count:",org,"| List of ORG:",org_list)
if(gpe!=0):print("GPE Count:",gpe,"| List of GPE:", gpe_list)
if(loc!=0):print("LOC Count:",loc,"| List of LOC:",loc_list)
if(product!=0):print("PRODUCT Count:",product,"| List of PRODUCT:",product_list)
if(event!=0):print("EVENT Count:",event,"| List of event:",event_list)
if(work_of_art!=0):print("WORK_OF_ART Count:",work_of_art,"| List of WORK_OF_ART:",work_of_art_list)
if(law!=0):print("LAW Count:",law,"| List of LAW:",law_list)
if(language!=0):print("LANGUAGE Count:",language,"| List of Language:",language_list)
if(date!=0):print("DATE Count:",date,"| List of date:",date_list)
if(time!=0): print("TIME Count:",time,"| List of time:",time_list)
if(percent!=0): print("PERCENT Count:",percent,"| List of percent:",percent_list)
if(money!=0): print("MONEY Count:",money,"| List of Money:",money_list)
if(quantity!=0): print("QUANTITY Count:",quantity,"| List of Quantity:",quantity_list)
if(ordinal!=0): print("ORDINAL Count:",ordinal,"| List of ordinal:",ordinal_list)
if(cardinal!=0): print("CARDINAL Count:",cardinal,"| List of cardinal:",cardinal_list)
print("Other:",other)
#print([(x, x.ent_iob_, x.ent_type_) for x in sentences[0]])

#displacy.render(article, jupyter=True, style='ent')