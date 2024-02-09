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
        self.assertEqual(k.parts, ('a', 'b', 'c'))

    def test_WindowKey_str(self):
        self.__class__._reset()

        k = WindowKey('a', 'b', 'c')
        self.assertEqual(k.__str__(), 'a.b.c')
        self.assertEqual(str(k), 'a.b.c')

    def test_WindowKey_eq(self):
        self.__class__._reset()

        k = WindowKey('a', 'b', 'c')
        self.assertTrue(k == 'a.b.c')
