from preprocess_mimic import process_mimic
from vectorize_data import process_data
from preprocess_final import preprocess
import os
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'aws':
        file_notes = 's3://dl4h-ksi/data/NOTEEVENTS.csv'
        file_diagnoses = 's3://dl4h-ksi/data/DIAGNOSES_ICD.csv'
    else:
        file_notes = 'data/NOTEEVENTS.csv'
        file_diagnoses = 'data/DIAGNOSES_ICD.csv'
    process_mimic(file_notes=file_notes,
                  file_diagnoses=file_diagnoses,
                  word_threshold=10,
                  code_freq_threshold=0,
                  output_file='data/combined_dataset')
    # original dataset from paper
    file_wiki = 'data/wikipedia_knowledge'
    subdir = 'original/'
    os.makedirs(f'data/{subdir}', exist_ok=True)
    process_data(file_wiki=file_wiki,
                 file_mimic='data/combined_dataset',
                 output_wiki=f'data/{subdir}wikivec',
                 output_mimic=f'data/{subdir}notevec',
                 vectorizer_type='binary')
    preprocess(file_wiki=file_wiki,
               file_mimic='data/combined_dataset',
               file_wikivec=f'data/{subdir}wikivec.npy',
               file_notevec=f'data/{subdir}notevec.npy',
               out_dir=f'data/{subdir}',
               original=True,
               vectorizer_type='binary')

    # original dataset from paper, normalized word frequencies
    subdir = 'original_freqs/'
    os.makedirs(f'data/{subdir}', exist_ok=True)
    process_data(file_wiki=file_wiki,
                 file_mimic='data/combined_dataset',
                 output_wiki=f'data/{subdir}wikivec',
                 output_mimic=f'data/{subdir}notevec',
                 vectorizer_type='count')
    preprocess(file_wiki=file_wiki,
               file_mimic='data/combined_dataset',
               file_wikivec=f'data/{subdir}wikivec.npy',
               file_notevec=f'data/{subdir}notevec.npy',
               out_dir=f'data/{subdir}',
               original=False,
               vectorizer_type='count')

    # original dataset from paper, tfidf
    subdir = 'original_tfidf/'
    os.makedirs(f'data/{subdir}', exist_ok=True)
    process_data(file_wiki=file_wiki,
                 file_mimic='data/combined_dataset',
                 output_wiki=f'data/{subdir}wikivec',
                 output_mimic=f'data/{subdir}notevec',
                 vectorizer_type='tfidf')
    preprocess(file_wiki=file_wiki,
               file_mimic='data/combined_dataset',
               file_wikivec=f'data/{subdir}wikivec.npy',
               file_notevec=f'data/{subdir}notevec.npy',
               out_dir=f'data/{subdir}',
               original=False,
               vectorizer_type='tfidf')