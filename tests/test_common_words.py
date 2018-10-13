import os
import shutil
from io import StringIO

import pandas as pd

from unittest import TestCase, main
from unittest.mock import patch, PropertyMock
from tempfile import mkdtemp

from common_words import (
    create_parser,
    FindCommonWords,
    NLKT_DATA_PATH,
    OCCURRENCES_LIMIT,
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
        # Delelte temp test directory
        shutil.rmtree(self.test_processed_path)


class TestParser(BaseTest):

    def test(self):
        return
        create_parser()

        # self.assertEqual(parsed.something, 'test')


class TestWords(BaseTest):

    def test_final_dataframe(self):

        docs_content = {'doc1': DOC_1, 'doc2': DOC_2}

        for filename, file_content in docs_content.items():
            with open(f'{self.test_processed_path}/{filename}.csv', 'wb') as fp:
                fp.write(file_content.encode('utf-8'))

        df = self.test_instance._get_final_dataframe(self.test_processed_path, OCCURRENCES_LIMIT)

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
