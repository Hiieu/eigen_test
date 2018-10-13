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
    PROCESSED_DOC_1,
    PROCESSED_DOC_2
)


class BaseTest(TestCase):

    def setUp(self):
        super(BaseTest, self).setUp()
        self.test_instance = FindCommonWords('', '', 5, False)

        self.test_path = mkdtemp('test_path')

        setup_directories(self.test_path, NLKT_DATA_PATH)

    def tearDown(self):
        # Delete temp test directory
        shutil.rmtree(self.test_path)


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

        nltk_path = mkdtemp('test_nltk_path')
        shutil.rmtree(nltk_path)

        setup_directories(self.test_path, nltk_path)

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


class TestFindCommonWords(BaseTest):
    """Check if we have correct data in the final dataframe before saving it to a csv file"""

    def _create_files(self, mapping):
        for filename, file_content in mapping.items():
            full_path = f'{self.test_path}/{filename}'

            with open(full_path, 'wb') as fp:
                fp.write(file_content.encode('utf-8'))

    def test_get_words_details_func(self):

        self._create_files({
            'doc1.txt': DOC_1
        })

        with open(f'{self.test_path}/doc1.txt', 'rb') as fp:
            result = self.test_instance._get_words_details(fp)

        result = dict(result)

        self.assertEqual(result, {
            'saw': [2, {'i saw a cat', 'i saw a black cat.'}],
            'black': [1, {'i saw a black cat.'}],
            'cat': [3, {'i saw a cat', 'i saw a black cat.', 'the cat doesnt like me.'}],
            'doesnt': [1, {'the cat doesnt like me.'}],
            'like': [1, {'the cat doesnt like me.'}]
        })

    def test_final_dataframe(self):
        self._create_files({
            'processed_doc1.txt.gzip': PROCESSED_DOC_1,
            'processed_doc2.txt.gzip': PROCESSED_DOC_2,
        })

        df = self.test_instance._get_final_dataframe(self.test_path, 0)

        cat_sentences = df.loc['cat']['sentences'].split('\n')
        cat_docs = df.loc['cat']['docs'].split(',')
        cat_docs.sort()
        cat_sentences.sort()
        self.assertEqual(cat_docs, ['processed_doc1.txt', 'processed_doc2.txt'])
        self.assertEqual(df.loc['cat']['total'], float(4))
        self.assertEqual(cat_sentences, ['I saw a black cat.', 'I saw a cat',
                                         'The cat doesnt like me.', 'The cat likes the mouse.'])

        dog_docs = df.loc['dog']['docs'].split(',')
        dog_sentences = df.loc['dog']['sentences'].split('\n')
        dog_sentences.sort()
        dog_docs.sort()
        self.assertEqual(dog_docs, ['processed_doc1.txt', 'processed_doc2.txt'])
        self.assertEqual(df.loc['dog']['total'], float(6))
        self.assertEqual(dog_sentences, ['dog dog', 'dog dog dog dog'])

        self.assertEqual(df.loc['bird']['docs'], 'processed_doc1.txt')
        self.assertEqual(df.loc['bird']['total'], float(4))
        self.assertEqual(df.loc['bird']['sentences'], 'bird bird bird bird')

        self.assertEqual(df.loc['turtle']['docs'], 'processed_doc2.txt')
        self.assertEqual(df.loc['turtle']['total'], float(5))
        self.assertEqual(df.loc['turtle']['sentences'], 'turtle turtle turtle turtle turtle')

    def test_custom_occurences_limit(self):
        self._create_files({
            'processed_doc1.txt.gzip': PROCESSED_DOC_1,
            'processed_doc2.txt.gzip': PROCESSED_DOC_2,
        })
        df = self.test_instance._get_final_dataframe(self.test_path, 5)
        self.assertEqual(list(df.index.values), ['dog', 'turtle'])


class TestWords(BaseTest):

    def test_contraction_words(self):
        sentence = "I'll eat some L`Apostrophe from 2004's, " \
                   "I don't know if it's even a food, I'm gonna find out."
        words = self.test_instance._get_words(sentence)
        words = list(words)
        words.sort()
        self.assertEqual(words,
                         ['2004s', 'I', 'Ill', 'Im', 'LApostrophe', 'dont', 'eat', 'even',
                          'find', 'food', 'gonna', 'know'])

    def test_multi_exclamation_marks(self):
        sentence = "To be continued..."
        words = self.test_instance._get_words(sentence)
        words = list(words)
        words.sort()
        self.assertEqual(words, ['To', 'continued'])

    def test_stop_words(self):
        sentence = "To be continued..."
        self.test_instance.include_stopwords = True
        words = self.test_instance._get_words(sentence)
        words = list(words)
        words.sort()
        self.assertEqual(words, ['To', 'be', 'continued'])
