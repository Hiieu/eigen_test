import os
import shutil

from unittest import TestCase, main
from unittest.mock import patch, PropertyMock

from tempfile import mkdtemp

from common_words import (
    FILES_PATH,
    PROCESSED_FILES_PATH,
    NLKT_DATA_PATH,
    create_parser,
    FindCommonWords,
    setup_directories,
)


class BaseTest(TestCase):

    def setUp(self):
        super(BaseTest, self).setUp()
        self.test_instance = FindCommonWords('', '', 5, False)

        self.processed_file_path = mkdtemp('test_processed_files')
        setup_directories(self.processed_file_path, NLKT_DATA_PATH)

    def tearDown(self):
        # Delelte temp test directory
        shutil.rmtree(self.processed_file_path)


class TestParser():

    def test(self):
        create_parser()

        self.assertEqual(parsed.something, 'test')


class TestWords(BaseTest):

    def test_apostrophes(self):
        sentence = "I'll eat some L`Apostrophe from 2004's, I don't know if it's even a food."
        words = self.test_instance._get_words(sentence)
        words = list(words)
        words.sort()
        self.assertEqual(
            words,
            ['2004s', 'I', 'Ill', 'LApostrophe', 'dont', 'eat', 'even', 'food', 'know']
        )

    def test_multi_exclamation_marks(self):
        sentence = "To be continued..."
        words = self.test_instance._get_words(sentence)
        words = list(words)
        words.sort()
        self.assertEqual(words, ['To', 'continued'])