import json

def normalize_chars(word):
    char_map = {
        'ѧ': 'я', 'ꙋ': 'у', 'оу': 'у', 'i': 'и', 'ї': 'и', 'ꙗ': 'я', 'ѯ': 'кс',
        'ѡ': 'о', 'с̑': 'с', 'м̑': 'м', 'є': 'е', 'кг': 'г', '̃': '', '̑': ''
    }
    for old, new in char_map.items():
        word = word.replace(old, new)
    return word

def convert_strings_to_tuples(data):
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            if isinstance(key, str) and '|' in key:
                key_parts = [tuple(part.split('=', 1)) for part in key.split('|')]
                key = tuple((k, tuple(v.split(','))) if ',' in v else (k, (v,)) for k, v in key_parts)
            elif isinstance(value, list) and key in ['norm_propn_suf']:
                def make_hashable(item):
                    if isinstance(item, list):
                        return tuple(make_hashable(sub_item) for sub_item in item)
                    return item
                value = set(make_hashable(item) for item in value)
            new_dict[key] = convert_strings_to_tuples(value)
        return new_dict
    elif isinstance(data, list):
        return [convert_strings_to_tuples(item) for item in data]
    else:
        return data

def deserialize_dict(data):
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            if isinstance(key, str) and '|' in key:
                key_parts = [tuple(part.split('=', 1)) if '=' in part else (part, '') for part in key.split('|')]
                key = tuple((k, tuple(v.split(','))) if ',' in v else (k, v) for k, v in key_parts)
            new_dict[key] = deserialize_dict(value)
        return new_dict
    elif isinstance(data, list):
        return [deserialize_dict(item) for item in data]
    else:
        return data



with open('norm_noun_suf.json', 'r', encoding='utf-8') as fp:
    ns_dict = convert_strings_to_tuples(json.load(fp))
with open('norm_verb_suf.json', 'r', encoding='utf-8') as fp:
    vs_dict = convert_strings_to_tuples(json.load(fp))
with open('norm_adj_suf.json', 'r', encoding='utf-8') as fp:
    adj_dict = convert_strings_to_tuples(json.load(fp))
with open('norm_propn_suf.json', 'r', encoding='utf-8') as fp:
    nominal_all = convert_strings_to_tuples(json.load(fp))


with open('norm_refined_nbases.json', 'r', encoding='utf-8') as fp:
    nbases = convert_strings_to_tuples(json.load(fp))
with open('norm_refined_vbases.json', 'r', encoding='utf-8') as fp:
    vbases = convert_strings_to_tuples(json.load(fp))
with open('norm_refined_adjbases.json', 'r', encoding='utf-8') as fp:
    adj_bases = convert_strings_to_tuples(json.load(fp))
with open('norm_propn_bases.json', 'r', encoding='utf-8') as fp:
    propn_bases = convert_strings_to_tuples(json.load(fp))


with open('norm_sconj.json', 'r', encoding='utf-8') as fp:
    sconj = deserialize_dict(json.load(fp))
with open('norm_cconj.json', 'r', encoding='utf-8') as fp:
    cconj = deserialize_dict(json.load(fp))
with open('norm_adp.json', 'r', encoding='utf-8') as fp:
    adp = deserialize_dict(json.load(fp))
with open('norm_pron.json', 'r', encoding='utf-8') as fp:
    pron = deserialize_dict(json.load(fp))
with open('norm_advs.json', 'r', encoding='utf-8') as fp:
    advs = deserialize_dict(json.load(fp))
with open('norm_det.json', 'r', encoding='utf-8') as fp:
    det = deserialize_dict(json.load(fp))
with open('norm_part.json', 'r', encoding='utf-8') as fp:
    part = deserialize_dict(json.load(fp))
with open('norm_aux.json', 'r', encoding='utf-8') as fp:
    aux = deserialize_dict(json.load(fp))
with open('norm_itj.json', 'r', encoding='utf-8') as fp:
    itj = deserialize_dict(json.load(fp))
with open('norm_num.json', 'r', encoding='utf-8') as fp:
    num = deserialize_dict(json.load(fp))
with open('norm_sym.json', 'r', encoding='utf-8') as fp:
    sym = deserialize_dict(json.load(fp))


n_roots = {}
for k, v in nbases.items():
    for x in v:
        n_roots.setdefault(x, []).append(k)

vroots = {}
for k, v in vbases.items():
    for x in v:
        vroots.setdefault(x, []).append(k)

adj_roots = {}
for k, v in adj_bases.items():
    for x in v:
        adj_roots.setdefault(x, []).append(k)

advroots = {}
for k, v in advs.items():
    for x in v:
        advroots.setdefault(x['word_form'], []).append({'lemma': k, 'features': x['features']})

sconjroots = {}
for k, v in sconj.items():
    for x in v:
        sconjroots.setdefault(x['word_form'], []).append({'lemma': k, 'features': x['features']})

cconjroots = {}
for k, v in cconj.items():
    for x in v:
        cconjroots.setdefault(x['word_form'], []).append({'lemma': k, 'features': x['features']})

adproots = {}
for k, v in adp.items():
    for x in v:
        adproots.setdefault(x['word_form'], []).append({'lemma': k, 'features': x['features']})

pronroots = {}
for k, v in pron.items():
    for x in v:
        pronroots.setdefault(x['word_form'], []).append({'lemma': k, 'features': x['features']})

propn_roots = {}
for k, v in propn_bases.items():
    for x in v:
        propn_roots.setdefault(x, []).append(k)

detroots = {}
for k,v in det.items(): 
    for x in v:
        detroots.setdefault(x['word_form'], []).append({'lemma': k, 'features': x['features']})

partroots = {}
for k,v in part.items(): 
    for x in v:
        partroots.setdefault(x['word_form'], []).append({'lemma': k, 'features': x['features']})

auxroots = {}
for k,v in aux.items(): 
    for x in v:
        auxroots.setdefault(x['word_form'], []).append({'lemma': k, 'features': x['features']})

symroots = {}
for k,v in sym.items(): 
    for x in v:
        symroots.setdefault(x['word_form'], []).append({'lemma': k, 'features': x['features']})

numroots = {}
for k,v in num.items(): 
    for x in v:
        numroots.setdefault(x['word_form'], []).append({'lemma': k, 'features': x['features']})

itjroots = {}
for k,v in itj.items(): 
    for x in v:
        itjroots.setdefault(x['word_form'], []).append({'lemma': k, 'features': x['features']})