# -*- coding: utf-8 -*-
import logging
import os
import re
import shutil
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from collections import defaultdict
from nltk import (
    download as nltk_download,
    sent_tokenize,
)
from nltk.data import path as nltk_path
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer

logger = logging.getLogger(__name__)


BASE_DIR = os.path.dirname(__file__)

FILES_PATH = os.path.join(BASE_DIR, 'test docs')
OUTPUT_PATH = os.path.join(BASE_DIR, 'result')
PROCESSED_FILES_PATH = os.path.join(BASE_DIR, 'processed_files')
NLKT_DATA_PATH = os.path.join(BASE_DIR, 'nltk_data')

OCCURRENCES_LIMIT = 20
COMPRESSION = 'gzip'
RESULT_CSV_FILENAME = 'result.csv'
INDEX_COLUMN = 'word'


class FindCommonWords:

    def __init__(self, result_path, processed_files_path, occurrences_limit,
                 include_stopwords):

        self.result_path = result_path
        self.occurrences_limit = occurrences_limit
        self.processed_files_path = processed_files_path
        self.include_stopwords = include_stopwords

    def find_common_words(self, source_path):

        # Use os.scandir() so we don't have to list all files at once
        for dir_entry in os.scandir(source_path):

            with open(dir_entry.path, 'rb') as file_object:
                filename = dir_entry.name

                dataframe = self._create_file_dataframe(file_object)

                if dataframe.empty:
                    continue

                filename = os.path.splitext(filename)
                csv_name = f'{filename}.{COMPRESSION}'
                dataframe.to_csv(os.path.join(self.processed_files_path, csv_name), index=False)

        result_dataframe = self._get_final_dataframe(self.processed_files_path, self.occurrences_limit)

        # Remove the index 'word' and treat it as a regular column
        result_dataframe.reset_index(INDEX_COLUMN, inplace=True)
        result_dataframe.to_csv(RESULT_CSV_FILENAME, index=False)

    @staticmethod
    def _get_final_dataframe(process_file_path, occurency_limit):

        result_columns = ['word', 'docs', 'total', 'sentences']
        merged_df = pd.DataFrame(columns=result_columns).set_index(INDEX_COLUMN)

        def concat(*args, delimeter=''):
            strs = [str(arg) for arg in args if not pd.isnull(arg)]
            return f'{delimeter}'.join(strs) if strs else np.nan

        np_concat = np.vectorize(concat)

        for dir_entry in os.scandir(process_file_path):
            doc_dataframe = pd.read_csv(dir_entry.path, skipinitialspace=True)

            # Set 'word' column as index
            doc_dataframe.set_index(INDEX_COLUMN, inplace=True)

            # Assuming the file extensions are in .txt
            doc_dataframe['docs'] = os.path.splitext(dir_entry.name)[0]

            merged_df = pd.merge(merged_df, doc_dataframe, on=[INDEX_COLUMN],
                                 how='outer', suffixes=('_result', '_doc'))

            # Concatenate columns

            # Split docs values with comma
            merged_df['docs'] = np_concat(merged_df['docs_result'], merged_df['docs_doc'], delimeter=',')

            # Split new sentences with new line
            merged_df['sentences'] = np_concat(merged_df['sentences_result'],
                                               merged_df['sentences_doc'], delimeter='\n')

            # Count all occurences across the files
            merged_df['total'] = merged_df.loc[:, ['total_result', 'total_doc']].sum(axis=1)

            # Drop unused columns for the next iteration
            unwanted_cols = set(list(merged_df)).difference(result_columns)
            if unwanted_cols:
                merged_df.drop(list(unwanted_cols), axis=1, inplace=True)

        merged_df.fillna('')
        merged_df = merged_df[merged_df['total'] >= occurency_limit]
        merged_df.sort_values(by=['total'], ascending=False, inplace=True)
        return merged_df

    def _create_file_dataframe(self, file_object):

        words_details = self._get_words_details(file_object)

        dataframe = pd.DataFrame.from_records(words_details, columns=['word', 'total', 'sentences'])
        return dataframe

    def _get_words_details(self, file_object):

        # create a defaultdict with number of occurrences
        # and set of sentences (if a word appears more than once in one sentence)
        words_details = defaultdict(lambda: [0, set()])

        for line in file_object:

            line = line.decode('utf-8')
            sentences = (sentence for sentence in sent_tokenize(line))
            for sentence in sentences:
                # Lower all words first to count the words easier
                sentence = sentence.lower()

                words = self._get_words(sentence)
                for word in words:
                    # total words in one sentence
                    total = len(re.findall(r'\b{0}\b'.format(word), sentence))

                    words_details[word][0] += total

                    # We use set so we won't have duplicates
                    # if more than one word exist in the sentence
                    words_details[word][1].add(sentence)

        result = []

        # Preparing data for dataframe,
        for _word, details in words_details.items():

            total = details[0]
            sentences = '\n'.join(list(details[1]))
            result.append((_word, total, sentences))

        return result

    def _get_words(self, sentence):
        """Get and format words from a sentence
        @:param sentence: str: a sentence from a line
        @return generator: list of words (with/without stop words)"""

        # Remove apostrophes in contractions words with
        # Syntactically they are one word so I will treat them like that
        # The other option would be changing explicitly contractions words to two seperate words
        sentence = re.sub("""(?<=\w)['’`"](?=\w)""", '', sentence)

        # Instead create a new wheel, we can use a nltk library for tokenization
        toktok = ToktokTokenizer()
        words = toktok.tokenize(sentence)

        # Here we can include or exclude stop words, they are probably the most common words
        if not self.include_stopwords:
            # This throws ResourceWarning. It opened the stream but devs forgot to include close()
            stop_words = stopwords.words('english')
            words = set(words).difference(stop_words)

        # Ignore punctuations and return generator with words
        return (word for word in words if not re.match('[^\w]', word))


def setup_directories(processed_path, nlkt_data_path):
    """Just in case delete/create a directory for processed files
    It's okay to store files data in memory if they are small.
    However, I would save/upload them them somewhere if the data is large"""

    if os.path.exists(processed_path):
        shutil.rmtree(processed_path)

    os.makedirs(processed_path)

    if not os.path.exists(nlkt_data_path):
        os.makedirs(nlkt_data_path)
        nltk_download(['punkt', 'stopwords'], download_dir=nlkt_data_path)

    nltk_path.append(nlkt_data_path)


def handle_parser(args):

    processed_file_path = args.processed_file_path
    setup_directories(processed_file_path, args.nlkt_data_path)

    FindCommonWords(args.output,
                    processed_file_path,
                    args.occurrences_limit,
                    args.include_stopwords
                    ).find_common_words(args.docs_path)


def create_parser():
    parser = ArgumentParser(description='Get common words from text documents',
                            epilog='python common_words.py -docs_path {full_path}/test docs')
    parser.add_argument('-docs_path', help='Path for documents to process', default=FILES_PATH)
    parser.add_argument('-processed_file_path', help='Path for each processed document. '
                                                'The script uses it as a transit directory .Default: result/', default=PROCESSED_FILES_PATH)
    parser.add_argument('-output', help='File output directory. Default: result/', default=OUTPUT_PATH)
    parser.add_argument('-occurrences_limit', help=f'Occurrences limit. Default: {OCCURRENCES_LIMIT}', default=OCCURRENCES_LIMIT)
    parser.add_argument('-nlkt_data_path', help=f'Nlkt path for nlkt module. Default: {NLKT_DATA_PATH}', default=NLKT_DATA_PATH)
    parser.add_argument('-include_stopwords', help=f'Include stop words?. Default: False', required=False)
    handle_parser(parser.parse_args())


if __name__ == "__main__":
    create_parser()
