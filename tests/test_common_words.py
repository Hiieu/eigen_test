import os
from unittest import TestCase, main
from unittest.mock import patch, PropertyMock

from tempfile import mkstemp

from common_words import (
    FILES_PATH,
    create_parser,
)


class TestAverageDuration(TestCase):


    def test(self):
        create_parser()

        self.assertEqual(parsed.something, 'test')


