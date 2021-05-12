"""
File: linkedbst.py
Author: Ken Lambert
"""

from abstractcollection import AbstractCollection
from bstnode import BSTNode
from linkedstack import LinkedStack
from linkedqueue import LinkedQueue
from math import log
import random
import time
import sys


class LinkedBST(AbstractCollection):
    """An link-based binary search tree implementation."""

    def __init__(self, sourceCollection=None):
        """Sets the initial state of self, which includes the
        contents of sourceCollection, if it's present."""
        self._root = None
        AbstractCollection.__init__(self, sourceCollection)

    # Accessor methods
    def __str__(self):
        """Returns a string representation with the tree rotated
        90 degrees counterclockwise."""

        def recurse(node, level):
            s = ""
            if node != None:
                s += recurse(node.right, level + 1)
                s += "| " * level
                s += str(node.data) + "\n"
                s += recurse(node.left, level + 1)
            return s

        return recurse(self._root, 0)

    def __iter__(self):
        """Supports a preorder traversal on a view of self."""
        if not self.isEmpty():
            stack = LinkedStack()
            stack.push(self._root)
            while not stack.isEmpty():
                node = stack.pop()
                yield node.data
                if node.right != None:
                    stack.push(node.right)
                if node.left != None:
                    stack.push(node.left)

    def preorder(self):
        """Supports a preorder traversal on a view of self."""
        return None

    def inorder(self):
        """Supports an inorder traversal on a view of self."""
        lyst = list()

        def recurse(node):
            if node != None:
                recurse(node.left)
                lyst.append(node.data)
                recurse(node.right)

        recurse(self._root)
        return iter(lyst)

    def postorder(self):
        """Supports a postorder traversal on a view of self."""
        return None

    def levelorder(self):
        """Supports a levelorder traversal on a view of self."""
        return None

    def __contains__(self, item):
        """Returns True if target is found or False otherwise."""
        return self.find(item) != None

    def find(self, item):
        """If item matches an item in self, returns the
        matched item, or None otherwise."""

        def recurse(node):
            if node is None:
                return None
            elif item == node.data:
                return node.data
            elif item < node.data:
                return recurse(node.left)
            else:
                return recurse(node.right)

        return recurse(self._root)
    
    def find_with_loop(self, item):
        """If item matches an item in self, returns the
        matched item, or None otherwise. Implemented using loop"""
        curr_node = self._root
        
        while curr_node:
            if curr_node.data == item:
                return curr_node.data
            if item > curr_node.data:
                curr_node = curr_node.right
            else:
                curr_node = curr_node.left

        return None

    # Mutator methods
    def clear(self):
        """Makes self become empty."""
        self._root = None
        self._size = 0

    def add(self, item):
        """Adds item to the tree."""

        # Helper function to search for item's position
        def recurse(node):
            # New item is less, go left until spot is found
            if item < node.data:
                if node.left == None:
                    node.left = BSTNode(item)
                else:
                    recurse(node.left)
            # New item is greater or equal,
            # go right until spot is found
            elif node.right == None:
                node.right = BSTNode(item)
            else:
                recurse(node.right)
                # End of recurse

        # Tree is empty, so new item goes at the root
        if self.isEmpty():
            self._root = BSTNode(item)
        # Otherwise, search for the item's spot
        else:
            recurse(self._root)
        self._size += 1
    
    def add_with_loop(self, item):
        """Adds utem to the tree, with the help of loop"""
        # Tree is empty, so new item goes at the root
        if self.isEmpty():
            self._root = BSTNode(item)
        # Otherwise, search for the item's spot
        else:
            curr_node = self._root
            while True:
                if item < curr_node.data:
                    if curr_node.left is None:
                        curr_node.left = BSTNode(item)
                        break
                    else:
                        curr_node = curr_node.left
                else:
                    if curr_node.right is None:
                        curr_node.right = BSTNode(item)
                        break
                    else:
                        curr_node = curr_node.right
        self._size += 1

    def remove(self, item):
        """Precondition: item is in self.
        Raises: KeyError if item is not in self.
        postcondition: item is removed from self."""
        if not item in self:
            raise KeyError("Item not in tree.""")

        # Helper function to adjust placement of an item
        def liftMaxInLeftSubtreeToTop(top):
            # Replace top's datum with the maximum datum in the left subtree
            # Pre:  top has a left child
            # Post: the maximum node in top's left subtree
            #       has been removed
            # Post: top.data = maximum value in top's left subtree
            parent = top
            currentNode = top.left
            while not currentNode.right == None:
                parent = currentNode
                currentNode = currentNode.right
            top.data = currentNode.data
            if parent == top:
                top.left = currentNode.left
            else:
                parent.right = currentNode.left

        # Begin main part of the method
        if self.isEmpty(): return None

        # Attempt to locate the node containing the item
        itemRemoved = None
        preRoot = BSTNode(None)
        preRoot.left = self._root
        parent = preRoot
        direction = 'L'
        currentNode = self._root
        while not currentNode == None:
            if currentNode.data == item:
                itemRemoved = currentNode.data
                break
            parent = currentNode
            if currentNode.data > item:
                direction = 'L'
                currentNode = currentNode.left
            else:
                direction = 'R'
                currentNode = currentNode.right

        # Return None if the item is absent
        if itemRemoved == None: return None

        # The item is present, so remove its node

        # Case 1: The node has a left and a right child
        #         Replace the node's value with the maximum value in the
        #         left subtree
        #         Delete the maximium node in the left subtree
        if not currentNode.left == None \
                and not currentNode.right == None:
            liftMaxInLeftSubtreeToTop(currentNode)
        else:

            # Case 2: The node has no left child
            if currentNode.left == None:
                newChild = currentNode.right

            # Case 3: The node has no right child
            else:
                newChild = currentNode.left

            # Case 2 & 3: Tie the parent to the new child
            if direction == 'L':
                parent.left = newChild
            else:
                parent.right = newChild

        # All cases: Reset the root (if it hasn't changed no harm done)
        #            Decrement the collection's size counter
        #            Return the item
        self._size -= 1
        if self.isEmpty():
            self._root = None
        else:
            self._root = preRoot.left
        return itemRemoved

    def replace(self, item, newItem):
        """
        If item is in self, replaces it with newItem and
        returns the old item, or returns None otherwise."""
        probe = self._root
        while probe != None:
            if probe.data == item:
                oldData = probe.data
                probe.data = newItem
                return oldData
            elif probe.data > item:
                probe = probe.left
            else:
                probe = probe.right
        return None

    def height(self):
        '''
        Return the height of tree
        :return: int
        '''

        def height1(top):
            '''
            Helper function
            :param top:
            :return:
            '''
            if top is None:
                return -1
            else:
                return 1 + max(height1(top.left), height1(top.right))
        
        return height1(self._root)

    def is_balanced(self):
        '''
        Return True if tree is balanced
        :return:
        '''
        check_balance = 2 * log(len(self) + 1, 2) - 1
        return self.height() < check_balance

    def range_find(self, low, high):
        '''
        Returns a list of the items in the tree, where low <= item <= high."""
        :param low:
        :param high:
        :return:
        '''
        # helper method
        def find_on_one_side(node, low, high):
            '''Recursively find elements that fit in the given range'''
            found = []
            if (node is None):
                return found
            else:
                if node.data in range(low, high + 1):
                    found.append(node.data)

                found += find_on_one_side(node.left, low, high)
                found += find_on_one_side(node.right, low, high)
            return found

        return find_on_one_side(self._root, low, high)

    def rebalance(self):
        '''
        Rebalances the tree.
        :return:
        '''
        def make_tree(elements, start, end, tree):
            if start > end:
                return tree
            else:
                mid = (start + end) // 2

                tree.add(elements[mid])
                make_tree(elements, start, mid - 1, tree)
                make_tree(elements, mid + 1, end, tree)
            return tree

        el_sorted = [el for el in self.inorder()]
        num_el = len(self)
        self.clear()
        return make_tree(el_sorted, 0, num_el - 1, self)

    def successor(self, item):
        """
        Returns the smallest item that is larger than
        item, or None if there is no such item.
        :param item:
        :type item:
        :return:
        :rtype:
        """
        current_node = self._root
        current_val = self._root.data
        temp_lowest_val = None

        while current_node is not None:
            if item < current_val:
                current_node = current_node.left
                if (temp_lowest_val is None) or (current_val < temp_lowest_val):
                    temp_lowest_val = current_val
            else:
                current_node = current_node.right
            if current_node: 
                current_val = current_node.data

        return temp_lowest_val

    def predecessor(self, item):
        """
        Returns the largest item that is smaller than
        item, or None if there is no such item.
        :param item:
        :type item:
        :return:
        :rtype:
        """
        if not item in self:
            return None
        
        current_node = self._root
        current_val = self._root.data
        temp_biggest_val = None

        while current_node is not None:
            if item > current_val:
                current_node = current_node.right
                if (temp_biggest_val is None) or (current_val > temp_biggest_val):
                    temp_biggest_val = current_val
            elif item <= current_val:
                current_node = current_node.left
            if current_node: 
                current_val = current_node.data
        
        return temp_biggest_val
        
    def demo_bst(self, path):
        """
        Demonstration of efficiency binary search tree for the search tasks.
        :param path:
        :type path:
        :return:
        :rtype:
        """
        words_lst = []
        with open(path, mode='r', encoding='utf-8') as words:
            for word in words:
                words_lst.append(word[:-1])

        to_search = random.choices(words_lst, k=100)
        to_search = sorted(to_search, key=lambda wrd: wrd.upper())
        
        words_lst = words_lst[:1000]
        # find each word in list
        start = time.time()
        for srch_wrd in to_search:
            srch_wrd in words_lst
        end = time.time()
        lst_srch_time = end - start

        # find each word in bst, first add words to bst from words_lst
        tree = LinkedBST()
        for word in words_lst:
            tree.add_with_loop(word)

        start = time.time()
        for srch_wrd in to_search:
            tree.find_with_loop(srch_wrd)
        end = time.time()
        from_dict_to_bst_srch_time = end - start

        # find each word in bst, first randomly add words to bst from words_lst
        tree2 = LinkedBST()
        shuflled_words_lst = random.sample(words_lst, len(words_lst))
        for word in shuflled_words_lst:
            tree2.add_with_loop(word)

        start = time.time()
        for srch_wrd in to_search:
            tree2.find_with_loop(srch_wrd)
        end = time.time()
        from_unsorted_dict_to_bst_srch_time = end - start
        
        # find each word in balanced bst.
        tree = tree.rebalance()
        start = time.time()
        for srch_wrd in to_search:
            tree.find_with_loop(srch_wrd)
        end = time.time()
        rebalanced_tree_search_time = end - start
       
        res = f'''FIND 10k random words in:
sorted list: {lst_srch_time}\n
BST, which was built with sorted list of words: {from_dict_to_bst_srch_time}\n
BST, which was built with shuffled list of words: {from_unsorted_dict_to_bst_srch_time}\n
balanced BSD: {rebalanced_tree_search_time}'''

        return res
