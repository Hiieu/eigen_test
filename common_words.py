import os
import re
import shutil
from argparse import ArgumentParser

import nltk
import pandas as pd

from collections import defaultdict

BASE_DIR = os.path.dirname(__file__)

FILES_PATH = os.path.join(BASE_DIR, 'test docs')
OUTPUT_PATH = os.path.join(BASE_DIR, 'result')
PROCESSED_FILES_PATH = os.path.join(BASE_DIR, 'processed_files')
NLKT_DATA_PATH = os.path.join(BASE_DIR, 'nltk_data')

OCCURRENCES_LIMIT = 50
COMPRESSION = 'gzip'


class FindCommonWords:

    def __init__(self, result_path, processed_files_path, occurrences_limit):

        self.result_path = result_path
        self.occurrences_limit = occurrences_limit
        self.processed_files_path = processed_files_path

    def find_common_words(self, path):

        # Use os.scandir() so we don't have to list all files at once
        for dir_entry in os.scandir(path):

            with open(dir_entry.path, 'rb') as file_object:
                filename = dir_entry.name

                dataframe = self._get_dataframe(file_object)

                csv_name = os.path.splitext(filename)[0]
                csv_name = f'{csv_name}.{COMPRESSION}'
                dataframe.to_csv(os.path.join(self.processed_files_path, csv_name),
                                 compression=COMPRESSION, index=False)

        self._get_common_words()

    def _get_common_words(self):

        result_dataframe = pd.DataFrame(columns=['word', 'docs', 'sentences'])

        for dir_entry in os.scandir(self.processed_files_path):
            doc_dataframe = pd.read_csv(dir_entry.path, compression=COMPRESSION)

            merged_df = pd.merge(result_dataframe, doc_dataframe, on=['word'],
                                 how='outer', suffixes=('_result', '_doc'))

            # Merge non empty values from doc's sentence column and result column
            # Split new sentences with a new line for readability
            merged_df.loc[(merged_df['sentences_result'].notnull())
                          & (merged_df['sentences_doc'].notnull()), 'sentences_result'] += '\----\\'

            # Merge doc's sentence to result column if result column is
            merged_df['sentences_result'] = merged_df['sentences_result'].fillna(merged_df['sentences_doc'])



    def _get_dataframe(self, file_object):

        dataframe = pd.DataFrame(columns=['word', 'total', 'sentences'])

        words_details = self._get_words_details(file_object)

        for word, details in words_details:
            total = details[0]
            sentences = details[1]
            dataframe.loc[len(dataframe.index) + 1] = word, total, list(sentences)


        dataframe = dataframe.loc[dataframe['total'] >= self.occurrences_limit]
        dataframe = dataframe.sort_values(by=['total'], ascending=False)
        return dataframe

    def _get_words_details(self, file_object):

        # create a defaultdict with number of occurrences
        # and set of sentences (if a word appears more than once in one sentence)
        words_details = defaultdict(lambda: [0, set()])
        for line in file_object:

            line = line.decode('utf-8')
            sentences = nltk.sent_tokenize(line)

            for sentence in sentences:
                words = self._get_words(sentence)
                for word in words:
                    words_details[word][0] += 1
                    words_details[word][1].add(sentence)

        return words_details.items()

    @staticmethod
    def _get_words(sentence):
        """Get and format words from a sentence
        @:param sentence: str: a sentence from a line
        @return list(str): list of words"""

        sentence = sentence.lower()

        # Remove punctuations
        sentence = re.sub('[^\w\s]', '', sentence)
        words = nltk.word_tokenize(sentence, preserve_line=True)
        return words


def _setup_directories(processed_path, nlkt_data_path):
    """Just in case delete/create a directory for processed files
    It's okay to store files data in memory if they are small.
    However, I would save them them somewhere if the data is large"""

    if os.path.exists(processed_path):
        shutil.rmtree(processed_path)

    os.makedirs(processed_path)

    if not os.path.exists(nlkt_data_path):
        os.makedirs(nlkt_data_path)
        nltk.download('punkt', download_dir=nlkt_data_path)

    nltk.data.path.append(nlkt_data_path)


def handle_parser(args):


    processed_file_path = args.processed_file_path
    _setup_directories(processed_file_path, args.nlkt_data_path)

    FindCommonWords(args.output, processed_file_path, args.occurrences_limit).find_common_words(args.docs_path)


def create_parser():
    parser = ArgumentParser(description='Get common words from text documents',
                            epilog='python common_words.py -docs_path {full_path}/test docs')
    parser.add_argument('-docs_path', help='Path for documents to process', default=FILES_PATH)
    parser.add_argument('-processed_file_path', help='Path for each processed document. '
                                                'The script uses it as a transit directory .Default: result/', default=PROCESSED_FILES_PATH)
    parser.add_argument('-output', help='File output directory. Default: result/', default=OUTPUT_PATH)
    parser.add_argument('-occurrences_limit', help=f'Occurrences limit. Default: {OCCURRENCES_LIMIT}', default=OCCURRENCES_LIMIT)
    parser.add_argument('-nlkt_data_path', help=f'Nlkt path for nlkt module. Default: {NLKT_DATA_PATH}', default=NLKT_DATA_PATH)
    handle_parser(parser.parse_args())


if __name__ == "__main__":
    create_parser()