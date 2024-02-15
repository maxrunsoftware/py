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

    def test_compare_iterable_equal(self):
        data = [
            ([1, 2, 3], [1, 2, 3]),
            ([1, 2, 3], (1, 2, 3)),
            ({1, 2, 3}, (1, 2, 3)),
            ([1, 2, 3], {1: 1, 2: 2, 3: 3}.keys()),
            ({1: 1, 2: 2, 3: 3}.values(), [1, 2, 3]),
            ([1, None, 3], [1, None, 3]),
            ([], []),
            ([None, None, None], [None, None, None]),
            (['aaa', 'bb', 'c'], ['aaa', 'bb', 'c']),
        ]

        for x, y in data:
            with self.subTest(x=x, y=y):
                self.assertEqual(0, compare_iterable(x, y), f"compare_iterable({x}, {y}) != 0")

    def test_compare_iterable_not_equal(self):
        data = [
            ([1, 2, 3], [1, 2, 9]),
            ([1, 2, None], [1, 2, 1]),
            ([1, 2, 3], [1, 2, 3, 4]),
            ([1, 2, 3, None], [1, 2, 3, None, None]),
            ([], [1]),
            (['aaa', 'bb', 'c'], ['aaa', 'bb', 'cc']),
            (['a', 'b', 'c'], ['a', 'b', 'd']),
            (['a', 'b', None], ['a', 'b', 'c']),
            ([None], [None, None]),
            ([None], ['a']),
        ]

        for x, y in data:
            with self.subTest(x=x, y=y):
                self.assertEqual(-1, compare_iterable(x, y), f"compare_iterable({x}, {y}) != -1")
                self.assertEqual(1, compare_iterable(y, x), f"compare_iterable({y}, {x}) != 1")
