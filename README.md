# KSI framework
This repository is forked from https://github.com/tiantiantu/KSI and extends the source code for the Knowledge Source Intergration (KSI) framework described in the following paper:
* **Bai, T., Vucetic, S., Improving Medical Code Prediction from Clinical Text via Incorporating Online Knowledge Sources, The Web Conference (WWW'19), 2019.**

Before running the code, you need to apply for [MIMIC-III](https://mimic.physionet.org/gettingstarted/access/) dataset and place the files "NOTEEVENTS.csv" and "DIAGNOSES_ICD.csv" under the `/data` directory of the project.

Afterwards, to build the datasets, run `build_datasets.py` from the root directory of the project. This will generate three datasets under `/data` containing:
* the dataset in its original form from the original repo
* a modified version of that dataset that supports multiple Wiki articles associated to a code rather than just one article per code
* a version of the dataset using normalized count vector representations of text rather than binary vectors encoding word presence. Meant to be used with the `ModifiedKSI` mechanism.
* a version of the dataset using tf-idf vector representations of text rather than binary vectors encoding word presence. Meant to be used with the `ModifiedKSI` mechanism.

For more flexibility, you can also use the individual preprocessing scripts, based off of the scripts in the original repo. The order is:
1. `preprocess_mimic.py`
2. `vectorize_mimic.py`
3. `preprocess_final.py`

You can also rebuild the datasets using different external knowledge sources. To that end, `wiki_scraper.ipynb` is a Jupyter notebook that scrapes Wikipedia to build an updated dataset of Wiki articles associated with ICD-9 codes. It can be used in place of the originally provided `wikipedia_knowledge` file.
