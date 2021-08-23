import pandas as pd
import numpy as np
import math
import string
import copy
from collections import Counter
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings("ignore")

data = pd.read_csv('../comments.csv')
data = data[['comment', 'label']]
data.drop_duplicates(inplace=True)
data['label'].value_counts()
data['label'] = data['label'].astype(np.int8)
data_not_toxic = data[data['label']==0].iloc[:5361]
data_toxic = data[data['label']==1]
data = pd.concat([data_not_toxic, data_toxic])

verbs = set()
nouns = set()
adjectives = set()
adverbs = set()
stop_words = set()

f = open('GEO_ENG_DICT.txt', encoding='utf-16', mode='r')
def generate_data(file=f):
    """
    function to generate data
    """
    for line in f.readlines():
        if 'v' in line.split():
            georgian_verbs = line.split(' v ')[0]
            georgian_verbs = georgian_verbs.split(',')
            for verb in georgian_verbs:
                if len(verb.split()) > 1:
                    verb = verb.split()[-1]
                if verb.endswith('ლი') or verb.endswith('ენი') or verb.endswith('ური'):
                    adjectives.add(verb.strip())
                else:
                    verbs.add(verb.strip())
        
        elif 'n' in line.split():
            georgian_nouns = line.split(' n ')[0]
            georgian_nouns = georgian_nouns.split(',')
            for noun in georgian_nouns:
                if len(noun.split()) > 1:
                    noun = noun.split()[-1]
                nouns.add(noun.strip())
            
        elif 'a' in line.split():
            georgian_adjectives = line.split(' a ')[0]
            georgian_adjectives = georgian_adjectives.split(',')
            for adjective in georgian_adjectives:
                if len(adjective.split()) > 1:
                    ad = adjective.split()[-1]
                    if not (ad.endswith('თ') or ad.endswith('დ') or ad!='გამოიყენა' or ad!='გადაიტენა'):
                        adjective.add(ad.strip())
                else:
                    adjectives.add(adjective.strip())
        
        elif 'adv' in line.split():
            georgian_adverbs = line.split(' adv ')[0]
            georgian_adverbs = georgian_adverbs.split(',')
            for adverb in georgian_adverbs:
                if len(adverb.split()) > 1:
                    adverb = adverb.split()[-1]
                    if adverb.endswith('ბა'):
                        verbs.add(adverb.strip())
                adverbs.add(adverb.strip())
        
        else:
            stop_words.add(line)
    file.close()


nouns_singular = set()
nouns_plural = set()
nouns_going = set()
def generate_noun_unique(nouns=nouns):
    for noun in nouns:
        if len(noun) == 0:
            continue

        #after checking word dictionary
        #some words were nconsistent with 
        #their type (noun) and those words were 
        #removed from the set or the words where 
        #already contained in their lexical form
        if noun == 'ართურმა':
            continue

            
        if noun.endswith('მ'):
            continue

        if noun.endswith('ს'):
            continue
        

        if noun.endswith('თ'):
            continue
        
        if noun.endswith('დ'):
            adjectives.add(noun[:-2] + 'ი')

        if noun.endswith('ები'):
            nouns_plural.add(noun)
            continue
        
        nouns_singular.add(noun)        
            
generate_noun_unique()


nouns_singular_dict = {}
nouns_plural_dict = {}
def generate_noun_dict_singular(nouns_singular=nouns_singular, nouns_plural=nouns_plural):
    #function to generate noun dictionary
    #that would be used for converting
    #words that are in the data into 
    #their lexical form -> lemmatization
    for noun in nouns_singular: 
        
        nouns_singular_dict[noun + 'ა'] = noun + 'არის'
        nouns_singular_dict[noun + 'ც'] = noun
        if noun[-1] == 'ი':
            brunvebi = ['ი', 'მა', 'ს', 'ის', 'ით', 'ად', 'ო']
            tandebulebi = ['ივით', 'ზე', 'ში','თან', 'ამდე','ისკენ' , 'ისთვის', 'იდან', 'ისგან' ]
            suffix_dict = {
                          'ალი' : 'ლ',
                          'ამი' : 'მ',
                          'ანი' : 'ნ',
                          'არი' : 'რ',
                          'ელი' : 'ლ',
                          'ემი' : 'მ',
                          'ენი' : 'ნ',
                          'ერი' : 'რ',
                          'ოლი' : 'ლ',
                          'ომი' : 'მ',
                          'ონი' : 'ნ',
                          'ორი' : 'რ',
            }
            suffix = noun[-3:]
            kumshvadi = 'NO'
            if suffix in suffix_dict.keys():
                kumshvadi = 'Unknown'
                noun_pluralk = noun[:-3] + suffix_dict[suffix] + 'ები'
                noun_plural = noun[:-1] + 'ები'
                if noun_pluralk in nouns_plural:
                    kumshvadi = 'T'
                    nouns_plural_dict[noun_pluralk] = noun
                elif noun_plural in  nouns_plural:
                    kumshvadi = 'F'
                    nouns_plural_dict[noun_plural] = noun
                else:
                    nouns_plural.add(noun_plural)
                    nouns_plural_dict[noun_plural] = noun
                    nouns_plural.add(noun_pluralk)
                    nouns_plural_dict[noun_pluralk] = noun
            else:
                noun_plural = noun[:-1] + 'ები'
                nouns_plural_dict[noun_plural] = noun
                
            nouns_singular_dict[noun[:-1]] = noun
            if kumshvadi=='NO' or kumshvadi=='F' or kumshvadi=='Unknown':
                for brunva in brunvebi:
                    nouns_singular_dict[noun[:-1] + brunva] = noun
                
                for tandebuli in tandebulebi:
                    nouns_singular_dict[noun[:-1] + tandebuli] = noun
            
            if kumshvadi=='T'  or kumshvadi=='Unknown':
                for ix, brunva in enumerate(brunvebi):
                    if ix >= 3 and ix < 6:
                        nouns_singular_dict[noun[:-3] + suffix_dict[suffix] + brunva] = noun
                    else:
                        nouns_singular_dict[noun[:-1] + brunva] = noun

                for ix, tandebuli in enumerate(tandebulebi):
                    if ix >= 5:
                        nouns_singular_dict[noun[:-3] + suffix_dict[suffix] + tandebuli] = noun
                    else:
                        nouns_singular_dict[noun[:-1] + tandebuli] = noun
        
        else:
            brunvebi = ['მ', 'ს', 'ს', 'თ', 'დ']
            tandebulebi = ['სავით', 'ზე', 'ში', 'სთან','მდე', 'სგან', 'სკენ', 'სთვის', 'დან']
            gamonaklisebi = ['მოყვარე','ბეგარა','ტომარა', 'ფანჯარა', 'პეპელა', 'ქარხანა', 'ქვეყანა']
            vowels = ['რ', 'რ', 'რ', 'რ', 'ლ', 'ნ', 'ნ']
            if noun in gamonaklisebi:
                v = gamonaklisebi.index(noun)
                vowel = vowels[v]
                nouns_singular_dict[noun] = noun
                for ix, brunva in enumerate(brunvebi):
                    if ix >= 2 and ix < 4:
                            nouns_singular_dict[noun[:-3] + vowel + 'ი' + brunva] = noun
                    else:
                        nouns_singular_dict[noun + brunva] = noun
                noun_plural = noun[:-3] + vowel + 'ები'
                nouns_plural.add(noun_plural)
                if noun[-1] ==  'ა':
                    nouns_singular_dict[noun + 'ვ'] = noun
                else:
                    nouns_singular_dict[noun + 'ო'] = noun
                continue
            if noun[-1]=='ა':
                kvecadi = 'Unknown'
                if noun[:-1] + 'ები' in nouns_plural:
                    kvecadi = 'T'
                    nouns_plural_dict[noun[:-1] + 'ები'] = noun
                elif noun + 'ები' in nouns_plural:
                    kvecadi = 'F'
                    nouns_plural_dict[noun + 'ები'] = noun
                else:
                    nouns_plural.add(noun[:-1] + 'ები')
                    nouns_plural.add(noun + 'ები')
                    nouns_plural_dict[noun[:-1] + 'ები'] = noun
                    nouns_plural_dict[noun + 'ები'] = noun

                    
                if kvecadi == 'T' or kvecadi == 'Unknown':
                    nouns_singular_dict[noun] = noun
                    for ix, brunva in enumerate(brunvebi):
                        if ix >= 2 and ix < 4:
                            nouns_singular_dict[noun[:-1] + 'ი' + brunva] = noun
                        else:
                            nouns_singular_dict[noun + brunva] = noun
                    nouns_singular_dict[noun + 'ვ'] = noun
                    for ix,tandebuli in enumerate(tandebulebi):
                        if ix>=5:
                            nouns_singular_dict[noun[:-1] + 'ი' + tandebuli] = noun
                        else:
                            nouns_singular_dict[noun + tandebuli] = noun
                            
                if kvecadi == 'F' or kvecadi == 'Unknown':
                    nouns_singular_dict[noun] = noun
                    for ix, brunva in enumerate(brunvebi):
                        nouns_singular_dict[noun + brunva] = noun
                    nouns_singular_dict[noun + 'ვ'] = noun
                    for tandebuli in tandebulebi:
                        nouns_singular_dict[noun + tandebuli] = noun
                    
            if noun[-1] =='ე':
                if noun + 'ები' not in nouns_plural:
                    nouns_plural.add(noun + 'ები')
                nouns_plural_dict[noun + 'ები'] = noun
                nouns_singular_dict[noun] = noun
                for ix, brunva in enumerate(brunvebi):
                    if ix >= 2 and ix < 4:
                        nouns_singular_dict[noun[:-1] + 'ი' + brunva] = noun
                    else:
                        nouns_singular_dict[noun + brunva] = noun
                for ix, brunva in enumerate(brunvebi):
                        nouns_singular_dict[noun + brunva] = noun
                
                nouns_singular_dict[noun + 'ო'] = noun
                for ix,tandebuli in enumerate(tandebulebi):
                    if ix>=5:
                        nouns_singular_dict[noun[:-1] + 'ი' + tandebuli] = noun
                        nouns_singular_dict[noun + tandebuli] = noun
                    else:
                        nouns_singular_dict[noun + tandebuli] = noun
            
            if noun[-1] == 'ო':
                gamonaklisi=False
                if noun == 'ღვინო':
                    gamonaklisi == True
                nouns_plural_dict[noun + 'ები'] = noun
                if gamonaklisi:
                    nouns_singular_dict[noun] = noun
                    for ix, brunva in enumerate(brunvebi):
                        if ix >= 2 and ix < 4:
                                nouns_singular_dict[noun[:-1] + 'ი' + brunva] = noun
                        else:
                            nouns_singular_dict[noun + brunva] = noun
                    nouns_singular_dict[noun + 'ო'] = noun
                
                else:
                    nouns_singular_dict[noun] = noun
                    for ix, brunva in enumerate(brunvebi):
                        nouns_singular_dict[noun + brunva] = noun

                    for tandebuli in tandebulebi:
                        nouns_singular_dict[noun + tandebuli] = noun

            
            if noun[-1]=='უ':
                nouns_singular_dict[noun] = noun
                nouns_plural_dict[noun + 'ები'] = noun
                for ix, brunva in enumerate(brunvebi):
                    nouns_singular_dict[noun + brunva] = noun
                for tandebuli in tandebulebi:
                    nouns_singular_dict[noun + tandebuli] = noun

generate_noun_dict_singular()

plurals = copy.deepcopy(list(nouns_plural_dict.keys()))
del nouns_plural, nouns_singular

def generate_noun_dict_plural(plurals = plurals, nouns_plural_dict=nouns_plural_dict):
    brunvebi = ['ი', 'მა', 'ს', 'ის', 'ით', 'ად', 'ო']
    tandebulebi = ['ივით', 'ზე', 'ში','თან', 'ამდე','ისკენ' , 'ისთვის', 'იდან', 'ისგან' ]
    for plural in plurals:
        noun = nouns_plural_dict[plural]
    
        for brunva in brunvebi:
            nouns_plural_dict[plural[:-1] + brunva] = noun

        for tandebuli in tandebulebi:
            nouns_plural_dict[plural[:-1] + tandebuli] = noun
generate_noun_dict_plural()

adjectives_singular = set()
adjectives_plural = set()
def generate_adjective_unique(adjectives=adjectives):
    for adjective in adjectives:
        if len(adjective) == 0:
            continue
 
        if adjective.endswith('მ'):
            continue
        
        if adjective.endswith('ს'):
            continue
  
        if adjective.endswith('თ'):
            continue

        if adjective.endswith('ა'):
            suffix = adjective[-3:]
            if suffix == 'ისა' or suffix == 'ვით':
                continue
        
        adjectives_singular.add(adjective)
        
            
generate_adjective_unique()

adjectives_singular_dict = {}
adjectives_plural_dict = {}
def generate_adjective_dict_singular(adjectives_singular=adjectives_singular, adjectives_plural=adjectives_plural):
    #generate adjective dictionary
    
    for adjective in adjectives_singular: 
        adjectives_singular_dict[adjective + 'ა'] = adjective + 'არის'
        adjectives_singular_dict[adjective + 'ც'] = adjective
        if adjective[-1] == 'ი':
            brunvebi = ['ი', 'მა', 'ს', 'ის', 'ით', 'ად', 'ო']
            tandebulebi = ['ივით', 'ზე', 'ში','თან', 'ამდე','ისკენ' , 'ისთვის', 'იდან', 'ისგან' ]
            suffix_dict = {
                          'ალი' : 'ლ',
                          'ამი' : 'მ',
                          'ანი' : 'ნ',
                          'არი' : 'რ',
                          'ელი' : 'ლ',
                          'ემი' : 'მ',
                          'ენი' : 'ნ',
                          'ერი' : 'რ',
                          'ოლი' : 'ლ',
                          'ომი' : 'მ',
                          'ონი' : 'ნ',
                          'ორი' : 'რ',
            }
            suffix = adjective[-3:]
            if suffix in suffix_dict.keys():
                kumshvadi = 'Unknown'
                adjective_pluralk = adjective[:-3] + suffix_dict[suffix] + 'ები'
                adjective_plural = adjective[:-1] + 'ები'
                adjectives_plural.add(adjective_plural)
                adjectives_plural_dict[adjective_plural] = adjective
                adjectives_plural.add(adjective_pluralk)
                adjectives_plural_dict[adjective_pluralk] = adjective
            else:
                kumshvadi = 'NO'
                adjective_plural = adjective[:-1] + 'ები'
                adjectives_plural_dict[adjective_plural] = adjective
                adjectives_plural.add(adjective_plural)
                
            adjectives_singular_dict[adjective[:-1]] = adjective
            if kumshvadi=='NO' or kumshvadi=='Unknown':
                for brunva in brunvebi:
                    adjectives_singular_dict[adjective[:-1] + brunva] = adjective
                
                for tandebuli in tandebulebi:
                    adjectives_singular_dict[adjective[:-1] + tandebuli] = adjective
            
            if kumshvadi=='Unknown':
                for ix, brunva in enumerate(brunvebi):
                    if ix >= 3 and ix < 6:
                        adjectives_singular_dict[adjective[:-3] + suffix_dict[suffix] + brunva] = adjective
                    else:
                        adjectives_singular_dict[adjective[:-1] + brunva] = adjective

                for ix, tandebuli in enumerate(tandebulebi):
                    if ix >= 5:
                        adjectives_singular_dict[adjective[:-3] + suffix_dict[suffix] + tandebuli] = adjective
                    else:
                        adjectives_singular_dict[adjective[:-1] + tandebuli] = adjective
        
        else:

            brunvebi = ['მ', 'ს', 'ს', 'თ', 'დ']
            tandebulebi = ['სავით', 'ზე', 'ში', 'სთან','მდე', 'სგან', 'სკენ', 'სთვის', 'დან']
            if adjective[-1]=='ა':
                kvecadi = 'Unknown'
                adjectives_plural.add(adjective[:-1] + 'ები')
                adjectives_plural.add(adjective + 'ები')
                adjectives_plural_dict[adjective[:-1] + 'ები'] = adjective
                adjectives_plural_dict[adjective + 'ები'] = adjective

                    
                if kvecadi == 'Unknown':
                    adjectives_singular_dict[adjective] = adjective
                    for ix, brunva in enumerate(brunvebi):
                        if ix >= 2 and ix < 4:
                            adjectives_singular_dict[adjective[:-1] + 'ი' + brunva] = adjective
                        else:
                            adjectives_singular_dict[adjective + brunva] = adjective
                    adjectives_singular_dict[adjective + 'ვ'] = adjective
                    for ix,tandebuli in enumerate(tandebulebi):
                        if ix>=5:
                            adjectives_singular_dict[adjective[:-1] + 'ი' + tandebuli] = adjective
                        else:
                            adjectives_singular_dict[adjective + tandebuli] = adjective
                            
                if kvecadi == 'F' or kvecadi == 'Unknown':
                    adjectives_singular_dict[adjective] = adjective
                    for ix, brunva in enumerate(brunvebi):
                        adjectives_singular_dict[adjective + brunva] = adjective
                    adjectives_singular_dict[adjective + 'ვ'] = adjective
                    for tandebuli in tandebulebi:
                        adjectives_singular_dict[adjective + tandebuli] = adjective
                    
            if adjective[-1] =='ე':
                if adjective + 'ები' not in adjectives_plural:
                    adjectives_plural.add(adjective + 'ები')
                adjectives_plural_dict[adjective + 'ები'] = adjective
                adjectives_singular_dict[adjective] = adjective
                for ix, brunva in enumerate(brunvebi):
                    if ix >= 2 and ix < 4:
                        adjectives_singular_dict[adjective[:-1] + 'ი' + brunva] = adjective
                    else:
                        adjectives_singular_dict[adjective + brunva] = adjective
                for ix, brunva in enumerate(brunvebi):
                        adjectives_singular_dict[adjective + brunva] = adjective
                
                adjectives_singular_dict[adjective + 'ო'] = adjective
                for ix,tandebuli in enumerate(tandebulebi):
                    if ix>=5:
                        adjectives_singular_dict[adjective[:-1] + 'ი' + tandebuli] = adjective
                        adjectives_singular_dict[adjective + tandebuli] = adjective
                    else:
                        adjectives_singular_dict[adjective + tandebuli] = adjective
            
            if adjective[-1] == 'ო':
                adjective_plural = adjective + 'ები'
                adjectives_plural.add(adjective_plural)
                adjectives_plural_dict[adjective_plural] = adjective
                adjectives_singular_dict[adjective] = adjective
                for ix, brunva in enumerate(brunvebi):
                    adjectives_singular_dict[adjective + brunva] = adjective

                for tandebuli in tandebulebi:
                    adjectives_singular_dict[adjective + tandebuli] = adjective

            
            if adjective[-1]=='უ':
                adjective_plural = adjective + 'ები'
                adjectives_singular_dict[adjective] = adjective
                adjectives_plural_dict[adjective_plural] = adjective
                adjectives_plural.add(adjective_plural)
                adjectives_singular_dict[adjective] = adjective
                for ix, brunva in enumerate(brunvebi):
                    adjectives_singular_dict[adjective + brunva] = adjective
                for tandebuli in tandebulebi:
                    adjectives_singular_dict[adjective + tandebuli] = adjective

generate_adjective_dict_singular()
a_plurals = copy.deepcopy(list(adjectives_plural_dict.keys()))
generate_noun_dict_plural(a_plurals, adjectives_plural_dict)

#create word dictionary
word_dictionary = {**nouns_singular_dict, **nouns_plural_dict,
                   **adjectives_singular_dict, **adjectives_plural_dict
                  }
del nouns_singular_dict, nouns_plural_dict
del adjectives_singular_dict, adjectives_plural_dict

class CommentPreparator(object):
    
    def __init__(self, word_dictionary=word_dictionary, mode = 'LEM'):
        self.word_dictionary = word_dictionary
        self.mode = mode
        self.remove_punct = str.maketrans('', '', string.punctuation)
        self.remove_numbers = str.maketrans('', '', string.digits)
        self.remove_latin = str.maketrans('', '', string.ascii_letters)
        self.stop_words = ['და', 'თუ', 'მაგრამ', 'თორემ', 'ხოლო', 'ან', 'რომ',
              'თუ არა', 'რადგან', 'რათა', 'როგორ', 'რაკი', 'ვიდრე',
              'ვინც', 'რაც', 'სადაც', 'საიდანაც', 'საითკენაც', 'როდესაც'
              'როცა', 'ხომ', 'კი', 'დიახ', 'აბა რა', 'ე.ი.', 'რადგანაც',
              'რისთვისაც', 'როგორაც', 'როგორც', 'ოღონდ', 'ანუ', 'აშ', 'ეი', 'მეთქი',
              'თქო']    
        self.word_dictionary = word_dictionary
    
    def __call__(self, comment):
        def checker(word):
    
            if word.endswith('ა') or word.endswith('ც') or word.endswith('ო'):
                if word[:-1] in self.word_dictionary:
                    return self.word_dictionary[word[:-1]]
            if word.endswith('თქო'):
                if word[:-3] in self.word_dictionary:
                    return self.word_dictionary[word[:-3]]
            if word.endswith('მეთქი'):
                if word[:-5] in self.word_dictionary:
                    return self.word_dictionary[word[:-5]]
            return word
        
        words = comment.split()
        words = [w.translate(self.remove_punct) for w in words]
        words = [w.translate(self.remove_numbers) for w in words]
        words = [w.translate(self.remove_latin) for w in words]
        words = [w for w in words if w not in self.stop_words]
        words = [self.word_dictionary[w]  if w in word_dictionary else checker(w) for w in words]
        words = [w for w in words if len(w) > 0]
        return words

comment_prep = CommentPreparator()
data['comment'] = data['comment'].apply(comment_prep)

counter = set()
data['comment'].apply(counter.update)
n_word_unique = len(counter)
word_unique = list(counter)

word_dict_inx = dict()
for ix, word in enumerate(sorted(word_unique)):
    word_dict_inx[word] = ix + 1

def indicer(sentence):
    l = list()
    for word in sentence:
        l.append(word_dict_inx[word])
    return l

data['comment'] = data['comment'].apply(indicer)
