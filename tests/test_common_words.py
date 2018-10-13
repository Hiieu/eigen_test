import os
import shutil

from argparse import Namespace
from unittest import TestCase
from unittest.mock import patch
from tempfile import mkdtemp

from common_words import (
    BASE_DIR,
    create_parser,
    FILES_PATH,
    FindCommonWords,
    handle_parser,
    NLKT_DATA_PATH,
    OCCURRENCES_LIMIT,
    PROCESSED_FILES_PATH,
    setup_directories
)

from .sample import (
    DOC_1,
    DOC_2
)


class BaseTest(TestCase):

    def setUp(self):
        super(BaseTest, self).setUp()
        self.test_instance = FindCommonWords('', '', 5, False)

        self.test_processed_path = mkdtemp('test_processed_files')

        setup_directories(self.test_processed_path, NLKT_DATA_PATH)

    def tearDown(self):
        # Delete temp test directory
        shutil.rmtree(self.test_processed_path)


class TestParser(BaseTest):

    def setUp(self):
        super(TestParser, self).setUp()
        self.parser_params = {
            'docs_path': FILES_PATH,
            'include_stopwords': None,
            'occurrences_limit': OCCURRENCES_LIMIT,
            'output_path': BASE_DIR,
            'processed_files_path': PROCESSED_FILES_PATH,
            'nlkt_data_path': NLKT_DATA_PATH
        }

    @patch('common_words.nltk_path')
    @patch('common_words.nltk_download')
    def test_setup_directories(self, download_mock, path_mock):

        nltk_path = 'non_existing/'

        setup_directories(self.test_processed_path, nltk_path)

        self.assertFalse(os.path.exists(self.test_processed_path))
        path_mock.append.assert_called_once_with(nltk_path)
        download_mock.assert_called_once_with(['punkt', 'stopwords'], download_dir=nltk_path)

    @patch('common_words.FindCommonWords')
    @patch('common_words.setup_directories')
    def test_handle_parser(self, setup_mock, find_mock):
        params = self.parser_params

        find_mock.find_common_words.return_value = lambda x: None

        handle_parser(Namespace(**params))

        setup_mock.assert_called_once_with(PROCESSED_FILES_PATH, NLKT_DATA_PATH)

        find_mock.assert_called_once_with(
            params['output_path'],
            params['processed_files_path'],
            params['occurrences_limit'],
            params['include_stopwords']
        )

        find_mock().find_common_words.assert_called_once_with(
            params['docs_path']
        )

    @patch('common_words.handle_parser')
    def test_create_parser(self, handle_parser_mock):
        create_parser()

        params = Namespace(**self.parser_params)
        handle_parser_mock.assert_called_once_with(params)


class TestFinalDataFrame(BaseTest):
    """Check if we have correct data in the final dataframe before saving it to a csv file"""

    def setUp(self):
        super(TestFinalDataFrame, self).setUp()
        docs_content = {'doc1.txt': DOC_1, 'doc2.txt': DOC_2}

        for filename, file_content in docs_content.items():
            with open(f'{self.test_processed_path}/{filename}.csv', 'wb') as fp:
                fp.write(file_content.encode('utf-8'))

    def test_final_dataframe(self):

        df = self.test_instance._get_final_dataframe(self.test_processed_path, 0)

        self.assertEqual(df.loc['cat']['docs'], 'doc2.txt,doc1.txt')
        self.assertEqual(df.loc['cat']['total'], float(4))
        self.assertEqual(df.loc['cat']['sentences'],
                         'I saw a black cat.\nThe cat doesnt like me.\nI saw a cat\nThe cat likes the mouse.')

        self.assertEqual(df.loc['dog']['docs'], 'doc2.txt,doc1.txt')
        self.assertEqual(df.loc['dog']['total'], float(6))
        self.assertEqual(df.loc['dog']['sentences'], 'dog dog dog dog\ndog dog')

        self.assertEqual(df.loc['bird']['docs'], 'doc1.txt')
        self.assertEqual(df.loc['bird']['total'], float(4))
        self.assertEqual(df.loc['bird']['sentences'], 'bird bird bird bird')

        self.assertEqual(df.loc['turtle']['docs'], 'doc2.txt')
        self.assertEqual(df.loc['turtle']['total'], float(5))
        self.assertEqual(df.loc['turtle']['sentences'], 'turtle turtle turtle turtle turtle')

    def test_custom_occurences_limit(self):
        df = self.test_instance._get_final_dataframe(self.test_processed_path, 5)
        self.assertEqual(list(df.index.values), ['dog', 'turtle'])


class TestWords(BaseTest):

    def test_contraction_words(self):
        sentence = "I'll eat some L`Apostrophe from 2004's, " \
                   "I don't know if it's even a food, I'm gonna find out."
        words = self.test_instance._get_words(sentence)
        words = list(words)
        words.sort()
        self.assertEqual( words,
            ['2004s', 'I', 'Ill', 'Im', 'LApostrophe', 'dont', 'eat', 'even',
             'find', 'food', 'gonna', 'know']
        )

    def test_multi_exclamation_marks(self):
        sentence = "To be continued..."
        words = self.test_instance._get_words(sentence)
        words = list(words)
        words.sort()
        self.assertEqual(words, ['To', 'continued'])
