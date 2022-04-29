from collections import defaultdict
import csv
import string
from stop_words import get_stop_words    # download stop words package from https://pypi.org/project/stop-words/
import numpy as np
import click
import s3fs

STOPWORDS = get_stop_words('english')


def parse_note_events(file='NOTEEVENTS.csv'):
    """
    Parse MIMIC-III NOTEEVENTS file to extract subject_id to discharge summary mapping.

    Parameters
    ----------
    file : str
        File path to NOTEEVENTS.csv

    Returns
    -------
    dict
        Dictionary of subject_id to discharge summary
    """
    subject_id_to_summary = defaultdict(list)
    is_aws = file.startswith('s3')
    if is_aws:
        fs = s3fs.S3FileSystem(anon=False)
        with fs.open(file, 'r') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            for row in reader:
                if row[6]=='Discharge summary':
                    subject_id = row[2]
                    summary = row[-1].replace('\n',' ').translate(str.maketrans('','',string.punctuation)).lower()
                    subject_id_to_summary[subject_id].append(summary)
    else:
        with open(file, 'r') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            for row in reader:
                if row[6]=='Discharge summary':
                    subject_id = row[2]
                    summary = row[-1].replace('\n',' ').translate(str.maketrans('','',string.punctuation)).lower()
                    subject_id_to_summary[subject_id].append(summary)

    return subject_id_to_summary


def freq_counts(texts, stop_words=STOPWORDS, threshold=10):
    """
    Build a vocabulary from a dictionary of texts. Excludes stop words, digits, and words that occur less
    than `threshold` times.

    Parameters
    ----------
    texts : dict
        Dictionary of texts to be used for building vocabulary
    stop_words : list
        List of stop words to be excluded from vocabulary
    threshold : int
        Minimum number of times a word must occur to be included in vocabulary

    Returns
    -------
    dict
        Dictionary of words and their frequencies
    """
    vocab = defaultdict(int)
    for _, v in texts.items():
        for text in v:
            tokens = text.strip('\n').split()
            for token in tokens:
                vocab[token] += 1

    vocab = {k: v for k, v in vocab.items() if v > threshold and not k.isdigit() and k not in stop_words}
    return vocab


def parse_diagnoses(file='DIAGNOSES_ICD.csv'):
    """
    Parse MIMIC-III DIAGNOSES_ICD file to map hospital admission ID to ICD codes. Also computes ICD code
    frequencies.

    Parameters
    ----------
    file : str
        File path to DIAGNOSES_ICD.csv

    Returns
    -------
    dict
        Dictionary of hospital admission ID to list of associated ICD codes
    dict
        Dictionary of ICD codes to their frequencies
    """
    hadmid_to_codes = defaultdict(list)
    is_aws = file.startswith('s3')
    if is_aws:
        fs = s3fs.S3FileSystem(anon=False)
        with fs.open(file, 'r') as f:
            f.readline()
            line = f.readline()
            while line:
                line = line.strip().split(',')
                icd9_code = line[4]
                hadm_id = line[2]
                if icd9_code[1:-1] != '':
                    hadmid_to_codes[hadm_id].append("d_"+icd9_code[1:-1])
                line=f.readline()
    else:
        with open(file, 'r') as f:
            f.readline()
            line = f.readline()
            while line:
                line = line.strip().split(',')
                icd9_code = line[4]
                hadm_id = line[2]
                if icd9_code[1:-1] != '':
                    hadmid_to_codes[hadm_id].append("d_"+icd9_code[1:-1])
                line=f.readline()

    code_freqs = defaultdict(int)
    for _, v in hadmid_to_codes.items():
        for code in v:
            code_freqs[code] += 1

    return hadmid_to_codes, code_freqs


def combine_datasets(subject_id_to_summary,
                     hadmid_to_codes,
                     vocab,
                     code_freqs,
                     id_list,
                     code_freq_threshold=0,
                     file='combined_dataset'):
    """
    Combine datasets from processed MIMIC-III NOTEEVENTS and DIAGNOSES_ICD files to create a dataset with
    hospital admission ID, associated ICD codes, and (processed) note text.

    Parameters
    ----------
    subject_id_to_summary : dict
        Dictionary of subject_id to discharge summary
    hadmid_to_codes : dict
        Dictionary of hospital admission ID to list of associated ICD codes
    vocab : dict
        Dictionary of words and their frequencies
    code_freqs : dict
        Dictionary of ICD codes to their frequencies
    id_list : list
        List of hospital admission IDs to be included in dataset
    code_freq_threshold : int
        Minimum number of times a ICD code must occur to be included in dataset
    file : str
        File path to output dataset
    """
    with open(file, 'w') as f:
        for id in id_list:
            if len(hadmid_to_codes[id]) > 0:
                f.write('start! '+id+'\n')
                f.write('codes: ')
                codes = list({code[0:5] for code in hadmid_to_codes[id]
                              if code_freqs[code] >= code_freq_threshold})
                for code in codes:
                    f.write(code + ' ')
                f.write('\nnotes:\n')
                for line in subject_id_to_summary[id]:
                    stripped = line.strip('\n').split()
                    for token in stripped:
                        if vocab.get(token):
                            f.write(token + ' ')
                    f.write('\n')
                f.write('end!\n')


def process_mimic(file_notes,
                  file_diagnoses,
                  word_threshold,
                  code_freq_threshold,
                  output_file):
    subject_id_to_summary = parse_note_events(file_notes)
    hadmid_to_codes, code_freqs = parse_diagnoses(file_diagnoses)
    vocab = freq_counts(subject_id_to_summary, threshold=word_threshold)
    id_list = np.load('data/IDlist.npy', encoding='bytes').astype(str)
    combine_datasets(subject_id_to_summary,
                     hadmid_to_codes,
                     vocab=vocab,
                     code_freqs=code_freqs,
                     id_list=id_list,
                     code_freq_threshold=code_freq_threshold,
                     file=output_file)


@click.command()
@click.option('--file_notes', default='data/NOTEEVENTS.csv')
@click.option('--file_diagnoses', default='data/DIAGNOSES_ICD.csv')
@click.option('--word_threshold', default=10)
@click.option('--code_freq_threshold', default=0)
@click.option('--output_file', default='data/combined_dataset')
def process_mimic_(file_notes,
                   file_diagnoses,
                   word_threshold,
                   code_freq_threshold,
                   output_file):
    process_mimic(file_notes,
                  file_diagnoses,
                  word_threshold,
                  code_freq_threshold,
                  output_file)

if __name__ == "__main__":
    process_mimic_()
