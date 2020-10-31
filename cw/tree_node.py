from pathlib import PurePosixPath
from typing import Dict, Any, Union, Sequence, Iterable, Tuple
import re

from cw.cached import cached

TreeNodeKey = Union[str, int]
TreeNodePath = Union[PurePosixPath, TreeNodeKey, Sequence[TreeNodeKey]]

int_re = re.compile(r"^\s*\[(\d+)\]\s*$")


class TreeNode:
    """

    """
    def __init__(self, *, value: Any=None, key: TreeNodePath = None, parent: "TreeNode"=None, default=None):
        self.children: Dict[TreeNodeKey, TreeNode] = {}
        if key is None and (parent is not None):
            raise ValueError("Only the root node may not have a key.")

        # At this point NoneType is allowed because it may only be the root node, which is unnamed.
        if not isinstance(key, (str, int, type(None))):
            raise TypeError("The key may only be of type str or int.")

        # Set the parent. This will be None for the root node.
        self.parent = parent

        if parent is not None:
            self.key = key
            self.root = parent.root
            self.parent.add_child(self)
        else:
            self.root = self
            self.key = ""

        self.value = value
        self.__default = default

    @classmethod
    def from_path_value_pairs(cls, path_value_pairs: Iterable[Tuple[TreeNodePath, Any]]):
        root = cls(default=lambda:  None)
        for path, value in path_value_pairs:
            root[path] = value
        root.set_default(None)
        return root

    @classmethod
    def from_dict(cls, path_value_pairs: Dict[TreeNodePath, Any]):
        return cls.from_path_value_pairs(path_value_pairs.items())

    @cached
    def default(self):
        # If __default is None, then return the default of the parent node.
        # If this is the root Node, return None.
        if self.__default is None:
            if self.is_root:
                return None
            else:
                return self.parent.default
        else:
            return self.__default

    def set_default(self, default):
        self.__default = default
        del self.default
        for _, child in self.items():
            if hasattr(child, "default"):
                del child.default

    @cached
    def path(self):
        node = self
        path_parts = []
        while node.parent is not None:
            path_parts.append(node.path_key)
            node = node.parent
        return PurePosixPath("/", *reversed(path_parts))

    @cached
    def path_key(self):
        return self.escape_key(self.key)

    @cached
    def is_root(self):
        return self is self.root

    @staticmethod
    def escape_key(key):
        if isinstance(key, int):
            return f"[{key}]"
        elif isinstance(key, str):
            return key
        else:
            raise TypeError("The key may only be of type str or int.")

    @staticmethod
    def unescape_key(escaped_key):
        match = int_re.match(escaped_key)
        if match:
            return int(match.group(1))
        else:
            return escaped_key

    @staticmethod
    def generate_path_obj(path: TreeNodePath) -> PurePosixPath:
        if isinstance(path, PurePosixPath):
            path = path
        elif isinstance(path, (list, tuple)):
            path = PurePosixPath(*(f"[{x}]" if isinstance(x, int) else x for x in path))
        else:
            path = PurePosixPath(f'[{path}]' if isinstance(path, int) else path)
        return path

    def add_child(self, child: "TreeNode"):
        self.children[child.key] = child

    def node_from_path(self, path: TreeNodePath):
        path = self.generate_path_obj(path)
        path_parts = path.parts

        # If the path_parts tuple is empty, then the path points to this object.
        if not path_parts:
            return self

        next_node_key = path_parts[0]

        # If the path is absolute set the search node it the root node.
        # Otherwise set the child node to the next child in line.
        if path.is_absolute():
            next_node = self.root
        else:
            # If the next node name is "..", go to the parent.
            if next_node_key == "..":
                next_node = self.parent or self
            else:
                if next_node_key in self.children:
                    next_node = self.children[self.escape_key(path_parts[0])]
                else:
                    if callable(self.default):
                        next_node = TreeNode(key=self.unescape_key(next_node_key), parent=self, value=self.default())
                    else:
                        raise KeyError(f"Node with key '{next_node_key}' in path '{path}' does not exist.")

        # Keep searching for the target node in at the next_node.
        return next_node.node_from_path(PurePosixPath(*path_parts[1:]))

    def __getitem__(self, path: TreeNodePath):
        return self.node_from_path(path).value

    def __setitem__(self, path: TreeNodePath, value):
        node = self.node_from_path(path)
        node.value = value

    def __iter__(self):
        yield from self.children

    def __len__(self):
        return len(self.children)

    def items(self):
        yield from self.children.items()

    def path_values(self):
        for _, node in self.children:
            yield node.path, node.weight

    def path_nodes(self):
        for _, node in self.children:
            yield node.path, node

    # It finally happened, this function ran without bugs on the first try.
    # This only means one thing. Programming karma will be striking back soon.
    #
    # Never mind, found two bugs already. Three. Four.
    # Seventh try's the charm.
    def object_hierarchy(self):
        # If all the child nodes have integers as keys and all of these keys are
        # positive ranging from 0 to the largest key value without gaps, then
        # it's a list.

        # If this node has children (and is not a list) then it's a dictionary

        # Otherwise this node is an endpoint node and contains a value.

        if len(self):
            # The node has children, so it's a container (either list or dictionary).
            key_set = set(self)
            if all((isinstance(key, int) for key in key_set)):
                # All children have integer keys, so it might be a list.
                # Check if all the keys range from 0 to max key value without gaps.
                if key_set == set(range(max(key_set) + 1)):
                    # It's a list, get the children's object_hierarchy() value, make sure
                    # they are put in the list in the correct order
                    return list((self.children[i].object_hierarchy() for i in range(max(key_set) + 1)))

            # If this code was reached, it's a dictionary.
            # Create a dictionary with the values being the return value of the
            # children's object_hierarchy() function.
            return dict(((key, child.object_hierarchy()) for key, child in self.items()))
        else:
            # The node has no children, so return the value.
            return self.value

    def __str__(self):
        return f"<TreeNode path='{self.path}' key='{self.key}' value='{self.value}'>"
