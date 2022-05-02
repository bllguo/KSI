import click
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split


def get_wiki_codes(file):
    wikivoc = {} # wikivoc
    count = 0
    wiki_codes = defaultdict(list) # codewiki
    with open(file, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            if line[0:4] == 'XXXd':
                line=line.strip('\n').split()
                for code in line:
                    if code[0:2] == 'd_':
                        wiki_codes[code].append(count)
                        wikivoc[code] = 1
                count += 1
            line=f.readline()
    return wikivoc, wiki_codes


def get_XY(file):
    features=[]
    labels=[]
    with open(file, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            line = line.strip('\n').split()
            if line[0] == 'codes:':
                labels.append(line[1:])
                line = f.readline()
                line = line.strip('\n').split()
                if line[0] == 'notes:':
                    feat = []
                    line=f.readline()
                    while line!='end!\n':
                        line=line.strip('\n').split()
                        feat += line
                        line = f.readline()
                    features.append(feat)
            line=f.readline()
    return features, labels


def update_wikivec(wikivec,
                   wikivoc,
                   wiki_codes,
                   labels,
                   combine_vecs=True,
                   vectorizer_type='binary'):
    label_to_ix = {}
    ix_to_label = {}
    for codes in labels:
        for code in codes:
            if code not in label_to_ix:
                label_to_ix[code]=len(label_to_ix)
                ix_to_label[label_to_ix[code]]=code

    tempwikivec=[]
    for i in range(len(ix_to_label)):
        if ix_to_label[i] in wikivoc:
            if combine_vecs:
                vecs = [wikivec[j] for j in wiki_codes[ix_to_label[i]]]
                temp = np.sum(vecs, axis=0)
                if vectorizer_type == 'binary':
                    temp[temp > 1] = 1
            else:
                temp = wikivec[wiki_codes[ix_to_label[i]][0]]
            tempwikivec.append(temp)
        else:
            tempwikivec.append([0.0]*wikivec.shape[1])
    return np.array(tempwikivec)


def produce_multihot_labels(data, wikivoc, label_to_ix):
    new_data = []
    for doc, note, codes in data:
        label = np.zeros(len(label_to_ix))
        for code in codes:
            if code in wikivoc.keys():
                label[label_to_ix[code]] = 1.
        new_data.append((doc, note, label))
    return np.array(new_data)

def save_data(features,
              labels,
              wikivec,
              notevec,
              wikivoc,
              out_dir='data/',
              test_split=.2,
              val_split=.125,
              seed=42):
    data = []
    for i in range(len(features)):
        data.append((features[i], notevec[i], labels[i]))

    data = np.array(data)

    label_to_ix = {}
    ix_to_label={}
    for _, _, codes in data:
        for code in codes:
            if code not in label_to_ix:
                if code in wikivoc:
                    label_to_ix[code]=len(label_to_ix)
                    ix_to_label[label_to_ix[code]]=code
    np.save(f'{out_dir}label_to_ix', label_to_ix)
    np.save(f'{out_dir}ix_to_label', ix_to_label)

    data = produce_multihot_labels(data, wikivoc, label_to_ix)
    label_vec = []
    for item in data:
        _, _, label = item
        label_vec.append(label)
    label_vec = np.array(label_vec)
    code_frequencies = label_vec.sum(axis=0)
    bin_10 = np.argwhere((code_frequencies <= 10) & (code_frequencies > 0)).squeeze()
    bin_50 = np.argwhere((code_frequencies <= 50) & (code_frequencies > 10)).squeeze()
    bin_100 = np.argwhere((code_frequencies <= 100) & (code_frequencies > 50)).squeeze()
    bin_500 = np.argwhere((code_frequencies <= 500) & (code_frequencies > 100)).squeeze()
    bin_remaining = np.argwhere(code_frequencies > 500).squeeze()

    bin_data = np.array([bin_10, bin_50, bin_100, bin_500, bin_remaining])
    np.save('data/bin_data', bin_data)

    training_data, test_data = train_test_split(data, test_size=test_split, random_state=seed)
    training_data, val_data = train_test_split(training_data, test_size=val_split, random_state=seed)

    np.save(f'{out_dir}training_data', training_data)
    np.save(f'{out_dir}test_data', test_data)
    np.save(f'{out_dir}val_data', val_data)

    word_to_ix = {}
    ix_to_word = {}
    ix_to_word[0] = 'OUT'

    for doc, _, codes in training_data:
        for word in doc:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)+1
                ix_to_word[word_to_ix[word]] = word
    np.save(f'{out_dir}word_to_ix', word_to_ix)
    np.save(f'{out_dir}ix_to_word', ix_to_word)

    code_dict = {}
    for codes in labels:
        for code in codes:
            if code not in code_dict:
                code_dict[code] = len(code_dict)
    newwikivec=[]
    for i in range(len(ix_to_label)):
        newwikivec.append(wikivec[code_dict[ix_to_label[i]]])
    newwikivec=np.array(newwikivec)
    np.save(f'{out_dir}newwikivec', newwikivec)


def preprocess(file_wiki,
               file_mimic,
               file_wikivec,
               file_notevec,
               out_dir='data/',
               test_split=0.2,
               val_split=0.125,
               seed=42,
               original=False,
               vectorizer_type='binary'):
    wikivec=np.load(file_wikivec)
    notevec=np.load(file_notevec)

    wikivoc, wiki_codes = get_wiki_codes(file_wiki)
    if original:
        wiki_codes['d_072']=[214]
        wiki_codes['d_698']=[125]
        wiki_codes['d_305']=[250]
        wiki_codes['d_386']=[219]
    np.save(f'{out_dir}wikivoc', wikivoc)
    features, labels = get_XY(file_mimic)

    wikivec = update_wikivec(wikivec, wikivoc, wiki_codes, labels,
                             combine_vecs=not original, vectorizer_type=vectorizer_type)
    save_data(features, labels, wikivec, notevec, wikivoc, out_dir,
              test_split, val_split, seed)


@click.command()
@click.option('--file_wiki', default='data/wikipedia_knowledge')
@click.option('--file_mimic', default='data/combined_dataset')
@click.option('--file_wikivec', default='data/wikivec.npy')
@click.option('--file_notevec', default='data/notevec.npy')
@click.option('--out_dir', default='data/')
@click.option('--test_split', default=0.2)
@click.option('--val_split', default=0.125)
@click.option('--seed', default=42)
@click.option('--original', default=False)
@click.option('--vectorizer_type', default='binary', type=click.Choice(['binary', 'count', 'tfidf']))
def preprocess_(file_wiki,
                file_mimic,
                file_wikivec,
                file_notevec,
                out_dir,
                test_split,
                val_split,
                seed,
                original,
                vectorizer_type):
    preprocess(file_wiki,
               file_mimic,
               file_wikivec,
               file_notevec,
               out_dir,
               test_split,
               val_split,
               seed,
               original,
               vectorizer_type)


if __name__ == "__main__":
    preprocess_()
