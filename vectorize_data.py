import click
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

        
def get_wiki_vocab(file):
    """
    Build a vocabulary from Wikipedia articles.
    
    Parameters
    ----------
    file : str
        File path to Wikipedia article text
    
    Returns
    -------
    dict
        Dictionary of words and their frequencies
    """
    vocab = {}
    with open(file, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            if line[0:3]!='XXX':
                line=line.strip('\n').split()
                for token in line:
                    vocab[token.lower()] = vocab.get(token.lower(), 0) + 1
            line=f.readline()
    return vocab


def get_notes_vocab(file):
    """
    Build a vocabulary from MIMIC notes text.
    
    Parameters
    ----------
    file : str
        File path to combined_dataset from 1_preprocess_mimic.py
    
    Returns
    -------
    dict
        Dictionary of words and their frequencies
    """
    vocab = {}
    with open(file, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            line = line.strip('\n').split()
            if line[0] == 'codes:':
                line = f.readline()
                line = line.strip('\n').split()
                if line[0] == 'notes:':
                    line = f.readline()
                    while line != 'end!\n':
                        line = line.strip('\n').split()
                        for token in line:
                            vocab[token.lower()] = vocab.get(token.lower(), 0) + 1
                        line = f.readline()
            line = f.readline()
    return vocab


def docs_to_strings(documents):
    """
    Convert a list of lists of tokens to a list of strings.
    
    Parameters
    ----------
    documents : list
        List of lists of tokens
    
    Returns
    -------
    list
        List of strings
    """
    data = []
    for doc in documents:
        text = ''
        for token in doc:
            text += token + ' '
        data.append(text)
    return data


def get_wiki_documents(file, vocab):
    """
    Get a list of lists of tokens from Wikipedia articles.
    
    Parameters
    ----------
    file : str
        File path to Wikipedia article texts
    vocab : dict
        Dictionary of words and their frequencies
        
    Returns
    -------
    list
        List of lists of tokens
    """
    documents = []
    with open(file, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            if line[0:4] == 'XXXd':
                doc = []
                line = f.readline()
                while line[0:4] != 'XXXe':
                    line = line.strip('\n').split()
                    for token in line:
                        if token.lower() in vocab:
                            doc.append(token.lower())
                    line = f.readline()
                documents.append(doc)
            line = f.readline()
    
    return documents


def get_notes(file, vocab):
    """
    Get a list of lists of tokens from MIMIC notes.
    
    Parameters
    ----------
    file : str
        File path to combined_dataset from 1_preprocess_mimic.py
    vocab : dict
        Dictionary of words and their frequencies
        
    Returns
    -------
    list
        List of lists of tokens
    """
    documents = []
    with open(file, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            line = line.strip('\n').split()
            if line[0] == 'codes:':
                line = f.readline()
                line = line.strip('\n').split()
                if line[0] == 'notes:':
                    doc = []
                    line = f.readline()
                    while line != 'end!\n':
                        line = line.strip('\n').split()
                        for token in line:
                            if token.lower() in vocab:
                                doc.append(token)
                        line = f.readline()
                    documents.append(doc)
            line = f.readline()
    
    return documents


def vectorize(data, output_file, vectorizer):
    """
    Apply a vectorizer to a list of strings.
    
    Parameters
    ----------
    data : list
        List of strings
    output_file : str
        File path to save vectorized data
    vectorizer : sklearn.feature_extraction.text.CountVectorizer or sklearn.feature_extraction.text.TfidfVectorizer
        Vectorizer to apply
    """
    embed = vectorizer.fit_transform(data)
    embed = embed.A
    embed = np.array(embed, dtype=float)
    np.save(output_file, embed)


@click.command()
@click.option('--file_wiki', default='data/wikipedia_knowledge')
@click.option('--file_mimic', default='data/combined_dataset')
@click.option('--output_wiki', default='data/wikivec')
@click.option('--output_mimic', default='data/notevec')
@click.option('--vectorizer_type', default='binary', type=click.Choice(['binary', 'tfidf']))
def process_data(file_wiki,
                 file_mimic,
                 output_wiki,
                 output_mimic,
                 vectorizer_type):
    wiki_vocab = get_wiki_vocab(file_wiki)
    note_vocab = get_notes_vocab(file_mimic)
    
    vocab = set(note_vocab.keys()).intersection(set(wiki_vocab.keys()))
    
    wiki_docs = get_wiki_documents(file_wiki, vocab)
    notes = get_notes(file_mimic, vocab)
    
    vocab_map = {}
    for note in notes:
        for token in note:
            if token.lower() not in vocab_map.keys():
                vocab_map[token.lower()] = len(vocab_map)
                
    wiki_docs = docs_to_strings(wiki_docs)
    notes = docs_to_strings(notes)
    
    if vectorizer_type == 'binary':
        vectorizer = CountVectorizer(min_df=1, vocabulary=vocab_map, binary=True)
    elif vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(min_df=1, vocabulary=vocab_map)
    
    vectorize(wiki_docs, output_wiki, vectorizer)
    vectorize(notes, output_mimic, vectorizer)


if __name__ == "__main__":
    process_data()
