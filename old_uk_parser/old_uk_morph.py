import pyconll
import re
import difflib
from collections import defaultdict, Counter
import json
from fuzzywuzzy import fuzz, process
import pandas as pd
import numpy as np
from itertools import chain
import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
import sklearn_crfsuite
from sklearn_crfsuite import scorers, metrics
from sklearn_crfsuite.utils import flatten
from sklearn import metrics as skmetrics
from IPython.display import JSON
import load_dictionaries as ld

ns_dict = ld.ns_dict
vs_dict = ld.vs_dict
adj_dict = ld.adj_dict
nominal_all = ld.nominal_all
nbases = ld.nbases
vbases = ld.vbases
adj_bases = ld.adj_bases
propn_bases = ld.propn_bases
sconj = ld.sconj
cconj = ld.cconj
adp = ld.adp
pron = ld.pron
advs = ld.advs
n_roots = ld.n_roots
vroots = ld.vroots
adj_roots = ld.adj_roots
advroots = ld.advroots
sconjroots = ld.sconjroots
cconjroots = ld.cconjroots
adproots = ld.adproots
pronroots = ld.pronroots
propn_roots = ld.propn_roots
detroots = ld.detroots
partroots = ld.partroots
auxroots = ld.auxroots
itjroots = ld.itjroots
numroots = ld.numroots
symroots = ld.symroots

def normalize_chars(word):
    char_map = {
        'ѧ': 'я', 'ꙋ': 'у', 'оу': 'у', 'i': 'и', 'ї': 'и', 'ꙗ': 'я', 'ѯ': 'кс',
        'ѡ': 'о', 'с̑': 'с', 'м̑': 'м', 'є': 'е', 'кг': 'г', '̃': '', '̑': ''
    }
    for old, new in char_map.items():
        word = word.replace(old, new)
    return word

def normalize_feature_dict(features_dict_keys):
    flat_feats = defaultdict(set)
    for feat_tuple in features_dict_keys:
        for k, v in feat_tuple:
            flat_feats[k].update(v)
    return dict(flat_feats)
    

def analyze_noun(word):
    word = normalize_chars(word)
    analyses = []
    for suffix, entries in ns_dict.items():
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            if stem and stem in n_roots: 
                for entry, count in entries.items():
                    feat_dict = defaultdict(set)
                    for key, vals in entry:
                        if vals and vals != ('',):
                            feat_dict[key].update(vals)
                    analyses.append({
                        'word_form': word,
                        'root': stem,
                        'suffix': suffix,
                        'lemma': n_roots.get(stem, 'unknown'),
                        'features': dict(feat_dict)
                    })
            if stem and stem not in n_roots and suffix != '':
                for entry, count in entries.items():
                    feat_dict = defaultdict(set)
                    for key, vals in entry:
                        if vals and vals != ('',):
                            feat_dict[key].update(vals)
                    analyses.append({
                        'word_form': word,
                        'root': stem,
                        'suffix': suffix,
                        'lemma': n_roots.get(stem, 'unknown'),
                        'features': dict(feat_dict)
                    }) 
    if '' in ns_dict and word in n_roots:  
        for entry, count in ns_dict[''].items():
            feat_dict = defaultdict(set)
            for key, vals in entry:
                if vals and vals != ('',):
                    feat_dict[key].update(vals)
            analyses.append({
                'word_form': word,
                'root': word,
                'suffix': '',
                'lemma': n_roots.get(word, 'unknown'),
                'features': dict(feat_dict)
            })
    return analyses

def analyze_verb(word):
    analyses = []
    refl = ('ся', 'се', 'сѧ', 'сѣ', 'сє', 'сь')
    reflex = ''
    if word.endswith(refl):
        reflex = [r for r in refl if word.endswith(r)][0] 
        word = word[:-len(reflex)]
    for suffix, entries in vs_dict.items():
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            if stem and stem in vroots:  
                for entry, count in entries.items():
                    feat_dict = defaultdict(set)
                    for key, vals in entry:
                        if vals and vals != ('',):
                            feat_dict[key].update(vals)
                    analyses.append({
                        'word_form': word + reflex if reflex else word,
                        'root': stem,
                        'suffix': suffix,
                        'lemma': vroots.get(stem, 'unknown'),
                        'features': dict(feat_dict),
                        'reflex': reflex
                    })
            if stem and stem not in vroots and suffix != '':
                for entry, count in entries.items():
                    feat_dict = defaultdict(set)
                    for key, vals in entry:
                        if vals and vals != ('',):
                            feat_dict[key].update(vals)
                    analyses.append({
                        'word_form': word + reflex if reflex else word,
                        'root': stem,
                        'suffix': suffix,
                        'lemma': vroots.get(stem, 'unknown'),
                        'features': dict(feat_dict),
                        'reflex': reflex
                    }) 
    if '' in vs_dict and word in vroots and not reflex:  
        for entry, count in vs_dict[''].items():
            feat_dict = defaultdict(set)
            for key, vals in entry:
                if vals and vals != ('',):
                    feat_dict[key].update(vals)
            analyses.append({
                'word_form': word,
                'root': word,
                'suffix': '',
                'lemma': vroots.get(word, 'unknown'),
                'features': dict(feat_dict),
                'reflex': ''
            })
    return analyses

def analyze_adj(word):
    analyses = []
    for suffix, entries in adj_dict.items():
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            if stem and stem in adj_roots:  
                for entry, count in entries.items():
                    feat_dict = defaultdict(set)
                    for key, vals in entry:
                        if vals and vals != ('',):
                            feat_dict[key].update(vals)
                    analyses.append({
                        'word_form': word,
                        'root': stem,
                        'suffix': suffix,
                        'lemma': adj_roots.get(stem, 'unknown'),
                        'features': dict(feat_dict)
                    })
            if stem and stem not in adj_roots and suffix != '':  
                for entry, count in entries.items():
                    feat_dict = defaultdict(set)
                    for key, vals in entry:
                        if vals and vals != ('',):
                            feat_dict[key].update(vals)
                    analyses.append({
                        'word_form': word,
                        'root': stem,
                        'suffix': suffix,
                        'lemma': adj_roots.get(stem, 'unknown'),
                        'features': dict(feat_dict)
                    })
    if '' in adj_dict and word in adj_roots: 
        for entry, count in adj_dict[''].items():
            feat_dict = defaultdict(set)
            for key, vals in entry:
                if vals and vals != ('',):
                    feat_dict[key].update(vals)
            analyses.append({
                'word_form': word,
                'root': word,
                'suffix': '',
                'lemma': adj_roots.get(word, 'unknown'),
                'features': dict(feat_dict)
            })
    return analyses

def analyze_adv(word):
    analyses = []
    if word in advroots:  
        for entry in advroots[word]:
            features = entry.get('features', {})
            normalized_features = {k: set(v) if isinstance(v, (list, tuple)) else {v} if v else set() for k, v in features.items()}
            analyses.append({
                'word_form': word,
                'root': word,
                'suffix': '',
                'lemma': entry.get('lemma', word),
                'features': normalized_features
            })
    return analyses

def analyze_sconj(word):
    analyses = []
    if word in sconjroots:  
        for entry in sconjroots[word]:
            features = entry.get('features', {})
            normalized_features = {k: set(v) if isinstance(v, (list, tuple)) else {v} if v else set() for k, v in features.items()}
            analyses.append({
                'word_form': word,
                'root': word,
                'suffix': '',
                'lemma': entry.get('lemma', word),
                'features': normalized_features
            })
    return analyses

def analyze_cconj(word):
    analyses = []
    if word in cconjroots: 
        for entry in cconjroots[word]:
            features = entry.get('features', {})
            normalized_features = {k: set(v) if isinstance(v, (list, tuple)) else {v} if v else set() for k, v in features.items()}
            analyses.append({
                'word_form': word,
                'root': word,
                'suffix': '',
                'lemma': entry.get('lemma', word),
                'features': normalized_features
            })
    return analyses

def analyze_adp(word):
    analyses = []
    if word in adproots:  
        for entry in adproots[word]:
            features = entry.get('features', {})
            normalized_features = {k: set(v) if isinstance(v, (list, tuple)) else {v} if v else set() for k, v in features.items()}
            analyses.append({
                'word_form': word,
                'root': word,
                'suffix': '',
                'lemma': entry.get('lemma', word),
                'features': normalized_features
            })
    return analyses

def analyze_pron(word):
    analyses = []
    if word in pronroots:  
        for entry in pronroots[word]:
            features = entry.get('features', {})
            normalized_features = {k: set(v) if isinstance(v, (list, tuple)) else {v} if v else set() for k, v in features.items()}
            analyses.append({
                'word_form': word,
                'root': word,
                'suffix': '',
                'lemma': entry.get('lemma', word),
                'features': normalized_features
            })
    return analyses

def analyze_propn(word):
    analyses = []
    for suffix, entries in nominal_all.items():
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            if stem and stem in propn_roots:  
                for entry in entries:
                    feat_dict = defaultdict(set)
                    for key, vals in entry:
                        if vals and vals != ('',):
                            feat_dict[key].update(vals)
                    analyses.append({
                        'word_form': word,
                        'root': stem,
                        'suffix': suffix,
                        'lemma': propn_roots.get(stem, 'unknown'),
                        'features': dict(feat_dict)
                    })
    if word in propn_roots: 
        analyses.append({
            'word_form': word,
            'root': word,
            'suffix': '',
            'lemma': propn_roots.get(word, 'unknown'),
            'features': {}
        })
    return analyses

def analyze_det(word):
    analyses = []
    if word in detroots: 
        for entry in detroots[word]:
            features = entry.get('features', {})
            normalized_features = {k: set(v) if isinstance(v, (list, tuple)) else {v} if v else set() for k, v in features.items()}
            analyses.append({
                'word_form': word,
                'roof': word,
                'suffix': '',
                'lemma': entry.get('lemma', word),
                'features': normalized_features
            })
    return analyses

def analyze_part(word):
    analyses = []
    if word in partroots:  
        for entry in partroots[word]:
            features = entry.get('features', {})
            normalized_features = {k: set(v) if isinstance(v, (list, tuple)) else {v} if v else set() for k, v in features.items()}
            analyses.append({
                'word_form': word,
                'root': word,
                'suffix': '',
                'lemma': entry.get('lemma', word),
                'features': normalized_features
            })
    return analyses

def analyze_aux(word):
    analyses = []
    if word in auxroots:  
        for entry in auxroots[word]:
            features = entry.get('features', {})
            normalized_features = {k: set(v) if isinstance(v, (list, tuple)) else {v} if v else set() for k, v in features.items()}
            analyses.append({
                'word_form': word,
                'root': word,
                'suffix': '',
                'lemma': entry.get('lemma', word),
                'features': normalized_features
            })
    return analyses

def analyze_num(word):
    analyses = []
    if word in numroots: 
        for entry in numroots[word]:
            features = entry.get('features', {})
            normalized_features = {k: set(v) if isinstance(v, (list, tuple)) else {v} if v else set() for k, v in features.items()}
            analyses.append({
                'word_form': word,
                'root': word,
                'suffix': '',
                'lemma': entry.get('lemma', word),
                'features': normalized_features
            })
    return analyses

def analyze_itj(word):
    analyses = []
    if word in itjroots:
        for entry in itjroots[word]:
            features = entry.get('features', {})
            normalized_features = {k: set(v) if isinstance(v, (list, tuple)) else {v} if v else set() for k, v in features.items()}
            analyses.append({
                'word_form': word,
                'root': word,
                'suffix': '',
                'lemma': entry.get('lemma', word),
                'features': normalized_features
            })
    return analyses

def analyze_sym(word):
    analyses = []
    if word in symroots: 
        for entry in symroots[word]:
            features = entry.get('features', {})
            normalized_features = {k: set(v) if isinstance(v, (list, tuple)) else {v} if v else set() for k, v in features.items()}
            analyses.append({
                'word_form': word,
                'root': word,
                'suffix': '',
                'lemma': entry.get('lemma', word),
                'features': normalized_features
            })
    return analyses
    

def analyse_token(word, upos):
    #norm_word = normalize_chars(word)
    analysis = None
    if upos == 'NOUN':
        analysis = analyze_noun(word)
    elif upos == 'ADJ':
        analysis = analyze_adj(word)
    elif upos == 'VERB':
        analysis = analyze_verb(word)
    elif upos == 'ADV':
        analysis = analyze_adv(word)
    elif upos == 'SCONJ':
        analysis = analyze_sconj(word)
    elif upos == 'CCONJ':
        analysis = analyze_cconj(word)
    elif upos == 'ADP':
        analysis = analyze_adp(word)
    elif upos == 'PROPN':
        analysis = analyze_propn(word)
    elif upos == 'PRON':
        analysis = analyze_pron(word)
    elif upos == 'DET':
        analysis = analyze_det(word)
    elif upos == 'PART':
        analysis = analyze_part(word)
    elif upos == 'AUX':
        analysis = analyze_aux(word)
    elif upos == 'INTJ':
        analysis = analyze_itj(word)
    elif upos == 'NUM':
        analysis = analyze_num(word)
    # elif upos == 'SYM':
    #     analysis = analyze_sym(word)
    return analysis
