from preprocess_mimic import process_mimic
from vectorize_data import process_data
from preprocess_final import preprocess


if __name__ == "__main__":
    process_mimic(file_notes='data/NOTEEVENTS.csv',
                  file_diagnoses='data/DIAGNOSES_ICD.csv',
                  word_threshold=10,
                  code_freq_threshold=0,
                  output_file='data/combined_dataset')
    # original dataset from paper
    file_wiki = 'data/wikipedia_knowledge'
    subdir = 'original/'
    process_data(file_wiki=file_wiki,
                 file_mimic='data/combined_dataset',
                 output_wiki=f'data/{subdir}wikivec',
                 output_mimic=f'data/{subdir}notevec',
                 vectorizer_type='binary')
    preprocess(file_wiki=file_wiki,
                file_mimic='data/combined_dataset',
                output_wikivoc=f'data/{subdir}wikivoc',
                file_wikivec=f'data/{subdir}wikivec.npy',
                file_notevec=f'data/{subdir}notevec.npy',
                file_newwikivec=f'data/{subdir}newwikivec.npy',
                original=True,
                vectorizer_type='binary')

    # original dataset from paper, but with multiple codes per note
    subdir = 'original_mc/'
    process_data(file_wiki=file_wiki,
                 file_mimic='data/combined_dataset',
                 output_wiki=f'data/{subdir}wikivec',
                 output_mimic=f'data/{subdir}notevec',
                 vectorizer_type='binary')
    preprocess(file_wiki=file_wiki,
               file_mimic='data/combined_dataset',
               output_wikivoc=f'data/{subdir}wikivoc',
               file_wikivec=f'data/{subdir}wikivec.npy',
               file_notevec=f'data/{subdir}notevec.npy',
               file_newwikivec=f'data/{subdir}newwikivec.npy',
               original=False,
               vectorizer_type='binary')

    # original dataset from paper, tfidf
    subdir = 'original_tfidf/'
    process_data(file_wiki=file_wiki,
                 file_mimic='data/combined_dataset',
                 output_wiki=f'data/{subdir}wikivec',
                 output_mimic=f'data/{subdir}notevec',
                 vectorizer_type='tfidf')
    preprocess(file_wiki=file_wiki,
               file_mimic='data/combined_dataset',
               output_wikivoc=f'data/{subdir}wikivoc',
               file_wikivec=f'data/{subdir}wikivec.npy',
               file_notevec=f'data/{subdir}notevec.npy',
               file_newwikivec=f'data/{subdir}newwikivec.npy',
               original=False,
               vectorizer_type='tfidf')
