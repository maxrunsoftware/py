import unittest

from .mrs_gui import *


class MRSGUITests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(MRSGUITests, self).__init__(*args, **kwargs)
        logging_setup()
        self.log = logger(cls=self.__class__)

    @staticmethod
    def _reset():
        WindowKey.clear_keys()

    def test_WindowKey_init(self):
        self.__class__._reset()
        k = WindowKey('a')
        self.assertEqual(k.parts, ('a',))

        self.__class__._reset()
        k = WindowKey('a', 'b', 'c')
        print(k)
        self.assertEqual(k.parts, ('a', 'b', 'c'))
