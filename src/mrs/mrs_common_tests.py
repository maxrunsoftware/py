import unittest

from src.mrs.mrs_common import *


class MRSCommonTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(MRSCommonTests, self).__init__(*args, **kwargs)
        logging_setup()
        self.log = logger(cls=self.__class__)

    def test_iterable_flatten(self):
        data = 'a'
        self.assertEqual(['a'], iterable_flatten(data))

        data = 'abc'
        self.assertEqual(['abc'], iterable_flatten(data))

        data = ('a',)
        self.assertEqual(['a'], iterable_flatten(data))

        data = ('abc',)
        self.assertEqual(['abc'], iterable_flatten(data))

        data = ('abc', 'xyz')
        self.assertEqual(['abc', 'xyz'], iterable_flatten(data))

        data = ('abc', 1, 'xyz', 2)
        self.assertEqual(['abc', 1, 'xyz', 2], iterable_flatten(data))

        data = ('aaa', 'bbb', ['ccc', 'ddd'], 'eee')
        self.assertEqual(['aaa', 'bbb', 'ccc', 'ddd', 'eee'], iterable_flatten(data))

        data = [0, 1, 2, [3, 4, [5, 6, 7, (8, 9), 10], 11, (12,), 13], 14]
        self.assertEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], iterable_flatten(data))

    def test_iterable_flatten_recursive_single(self):
        data = [0, 1, 2, [3, 'aaa', [5, 6, 'bbb', (8, 'c'), 10], 11, (12,), 'ddd'], 14]

        def parser(x):
            if isinstance(x, int):
                return f"i{x}"
            else:
                return x

        self.assertEqual(
            ['i0', 'i1', 'i2', 'i3', 'aaa', 'i5', 'i6', 'bbb', 'i8', 'c', 'i10', 'i11', 'i12', 'ddd', 'i14'], iterable_flatten(data, parser=parser)
        )

    def test_iterable_flatten_recursive_multiple(self):
        data = [0, 1, 2]

        def parser(x):
            if isinstance(x, int):
                return [('myInt', f"{x}"), 'woo']
            else:
                return x

        self.assertEqual(['myInt', '0', 'woo', 'myInt', '1', 'woo', 'myInt', '2', 'woo'], iterable_flatten(data, parser=parser))

    def test_xstr_iterable_empty(self):
        self.assertEqual([], xstr([]))

    def test_xstr_iterable_None(self):
        self.assertEqual(['a', '', 'c'], xstr(['a', None, 'c']))
