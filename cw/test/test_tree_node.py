from cw.tree_node import TreeNode
from cw.object_hierarchies import object_hierarchy_equals
import unittest
import re
from pathlib import PurePosixPath
import pprint

pprint = pprint.PrettyPrinter(indent=2).pprint


class TestTreeNode(unittest.TestCase):
    def test_path_property(self):
        root = TreeNode(value=111)
        child1 = TreeNode(key="foo", parent=root)
        child2 = TreeNode(key="bar", parent=child1)
        child3 = TreeNode(key="baz", parent=child2)
        child4 = TreeNode(value=333, key="qux", parent=child3)

        child5 = TreeNode(key="qux", parent=child2)
        child6 = TreeNode(key="quux", parent=child5)
        child7 = TreeNode(value=333, key="quuz", parent=child6)

        self.assertEqual(PurePosixPath("/foo/bar/baz/qux"), child4.path)
        self.assertEqual(PurePosixPath("/foo/bar/qux/quux/quuz"), child7.path)

    def test_node_from_path(self):
        root = TreeNode(value=111)
        child1 = TreeNode(key="foo", parent=root)
        child2 = TreeNode(key="bar", parent=child1)
        child3 = TreeNode(key="baz", parent=child2)
        child4 = TreeNode(value=333, key="qux", parent=child3)

        child5 = TreeNode(key="qux", parent=child2)
        child6 = TreeNode(key="quux", parent=child5)
        child7 = TreeNode(value=333, key="quuz", parent=child6)

        self.assertEqual(str(root.node_from_path(PurePosixPath())), "<TreeNode path='/' key='' value='111'>")
        self.assertEqual(str(root.node_from_path(PurePosixPath(".."))), "<TreeNode path='/' key='' value='111'>")
        self.assertEqual(str(child1.node_from_path(PurePosixPath(".."))), "<TreeNode path='/' key='' value='111'>")
        self.assertEqual(str(child4.node_from_path(PurePosixPath("../../../.."))),
                         "<TreeNode path='/' key='' value='111'>")

        self.assertEqual(str(root.node_from_path(PurePosixPath("/foo/bar/baz/.."))),
                         "<TreeNode path='/foo/bar' key='bar' value='None'>")
        self.assertEqual(str(child4.node_from_path(PurePosixPath("../../qux/quux/quuz"))),
                         "<TreeNode path='/foo/bar/qux/quux/quuz' key='quuz' value='333'>")

        self.assertEqual(str(child6.node_from_path(PurePosixPath("/foo/bar"))),
                         "<TreeNode path='/foo/bar' key='bar' value='None'>")

    def test_node_from_path_with_missing(self):
        root = TreeNode(default=lambda: None)
        self.assertEqual(str(root.node_from_path(PurePosixPath("./hello/world"))),
                         "<TreeNode path='/hello/world' key='world' value='None'>")

    def test_node_from_path_with_missing_integer_key(self):
        root = TreeNode(default=lambda: None)
        self.assertEqual(str(root.node_from_path(PurePosixPath("./hello/[0]/world"))),
                         "<TreeNode path='/hello/[0]/world' key='world' value='None'>")
        zero_node = root.node_from_path(PurePosixPath("./hello/[0]"))
        self.assertEqual(str(zero_node),
                         "<TreeNode path='/hello/[0]' key='0' value='None'>")
        self.assertIsInstance(zero_node.key, int)

    def test_node_from_path_with_missing_failing(self):
        root = TreeNode()
        with self.assertRaisesRegex(KeyError,
                                     re.escape(r"Node with key 'hello' in path 'hello/world' does not exist.")):
            root.node_from_path(PurePosixPath("./hello/world"))

    def test_node_from_path_integer_keys(self):
        root = TreeNode()
        child1 = TreeNode(key="foo", parent=root)
        child2 = TreeNode(value=333, key=0, parent=child1)
        child3 = TreeNode(value=333, key=1, parent=child1)
        child4 = TreeNode(value=333, key=2, parent=child1)
        correct_tree = {'foo': [333, 333, 333]}
        errors = object_hierarchy_equals(root.object_hierarchy(), correct_tree)
        self.assertEqual(len(errors), 0)

    def test_object_hierarchy(self):
        root = TreeNode(value=111)
        child1 = TreeNode(key="foo", parent=root)
        child2 = TreeNode(value=111, key=0, parent=child1)
        child3 = TreeNode(value=222, key=1, parent=child1)
        child4 = TreeNode(key=2, parent=child1)

        child5 = TreeNode(key="bar", parent=child4)
        child6 = TreeNode(key="quux", parent=child5)
        child7 = TreeNode(value=333, key="quuz", parent=child6)

        correct_tree = {
            'foo': [
                111,
                222,
                {
                    'bar': {
                        'quux': {
                            'quuz': 333
                        }
                    }
                }
            ]
        }

        errors = object_hierarchy_equals(root.object_hierarchy(), correct_tree)
        self.assertEqual(len(errors), 0)

    def test_from_path_value_pairs(self):
        path_value_pairs = [
            (PurePosixPath("hello/world"), "!!!"),
            (PurePosixPath("foo/[0]"), "spam"),
            (PurePosixPath("/foo/[2]"), "eggs"),
            (PurePosixPath("foo/[1]"), "ham"),
        ]

        root = TreeNode.from_path_value_pairs(path_value_pairs)
        correct_tree = {'foo': ['spam', 'ham', 'eggs'], 'hello': {'world': '!!!'}}
        errors = object_hierarchy_equals(root.object_hierarchy(), correct_tree)
        self.assertEqual(len(errors), 0)

    def test_from_path_dict(self):
        path_value_pairs = [
            (PurePosixPath("hello/world"), "!!!"),
            (PurePosixPath("foo/[0]"), "spam"),
            (PurePosixPath("/foo/[2]"), "eggs"),
            (PurePosixPath("foo/[1]"), "ham"),
        ]

        root = TreeNode.from_dict(dict(path_value_pairs))
        correct_tree = {'foo': ['spam', 'ham', 'eggs'], 'hello': {'world': '!!!'}}
        errors = object_hierarchy_equals(root.object_hierarchy(), correct_tree)
        self.assertEqual(len(errors), 0)
