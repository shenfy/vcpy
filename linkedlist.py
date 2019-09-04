from typing import TypeVar, Generic, Optional

T = TypeVar('T')

class LinkedListNode(Generic[T]):
  def __init__(self, value: T) -> None:
    self.value = value
    self.pre: Optional[LinkedListNode[T]] = None
    self.next: Optional[LinkedListNode[T]] = None

class LinkedList(Generic[T]):
  def __init__(self) -> None:
    self.__head: Optional[LinkedListNode[T]] = None
    self.__tail: Optional[LinkedListNode[T]] = None

  def __len__(self) -> int:
    result = 0
    cur = self.__head
    while cur:
      result += 1
      cur = cur.next
    return result

  def __str__(self) -> str:
    result = '<'
    cur = self.__head
    while cur:
      result += repr(cur.value)
      if cur.next:
        result += ' -> '
      cur = cur.next
    result += '>'
    return result

  def empty(self) -> bool:
    return len(self) == 0

  def __insert_node(self, pos: Optional[LinkedListNode[T]], node: LinkedListNode[T]) -> None:
    # first node
    if self.empty():
      self.__head = node
      self.__tail = node
      node.pre = None
      node.next = None
      return

    assert self.__head and self.__tail

    # insert after tail
    if pos is None:
      node.pre = self.__tail
      self.__tail.next = node
      self.__tail = node
      node.next = None
      return

    # insert before a node
    node.pre = pos.pre
    if pos.pre:
      pos.pre.next = node
    node.next = pos
    pos.pre = node
    if pos is self.__head:
      self.__head = node

  def front(self) -> LinkedListNode[T]:
    if self.empty():
      raise IndexError()
    assert self.__head
    return self.__head

  def back(self) -> LinkedListNode[T]:
    if self.empty():
      raise IndexError()
    assert self.__tail
    return self.__tail

  def clear(self) -> None:
    while not self.empty():
      assert self.__head
      self.erase(self.__head)

  def insert(self, pos: Optional[LinkedListNode[T]], value: T) -> LinkedListNode[T]:
    node = LinkedListNode(value)
    self.__insert_node(pos, node)
    return node

  def erase(self, node: LinkedListNode[T]) -> Optional[LinkedListNode[T]]:
    # only element in this list
    if node is self.__head and node is self.__tail:
      self.__head = None
      self.__tail = None
      return None

    # erase head
    if node is self.__head:
      assert node.next
      self.__head = node.next
      node.next.pre = None
      node.next = None
      return self.__head

    # erase tail
    if node is self.__tail:
      assert node.pre
      self.__tail = node.pre
      node.pre.next = None
      node.pre = None
      return None

    # an element in the middle
    result = node.next
    assert node.pre and node.next
    node.pre.next = node.next
    node.next.pre = node.pre
    node.pre = None
    node.next = None
    return result

  def push_back(self, value: T) -> None:
    self.insert(None, value)

  def pop_back(self) -> None:
    self.erase(self.back())

  def push_front(self, value: T) -> None:
    self.insert(self.__head, value)

  def pop_front(self) -> None:
    self.erase(self.front())

  def move_to_front(self, node: LinkedListNode[T]) -> None:
    self.erase(node)
    self.__insert_node(self.__head, node)

  def move_to_back(self, node: LinkedListNode[T]) -> None:
    self.erase(node)
    self.__insert_node(None, node)

import unittest

class TestLinkedList(unittest.TestCase):
  def test(self):
    # test constructor
    l = LinkedList()
    self.assertEqual(len(l), 0)
    self.assertTrue(l.empty())
    self.assertRaises(IndexError, lambda: l.front())
    self.assertRaises(IndexError, lambda: l.back())
    self.assertRaises(IndexError, lambda: l.pop_front())
    self.assertRaises(IndexError, lambda: l.pop_back())
    # test push back
    cases = [1, '2', (3,), 40, 500]
    for c in cases:
      l.push_back(c)
    self.assertEqual(len(l), len(cases))
    self.assertEqual(l.front().value, cases[0])
    self.assertEqual(l.back().value, cases[-1])
    # test clear
    l.clear()
    self.assertTrue(l.empty())
    # test push front
    for c in cases:
      l.push_front(c)
    self.assertEqual(len(l), len(cases))
    self.assertEqual(l.front().value, cases[-1])
    self.assertEqual(l.back().value, cases[0])
    # test pop
    l.pop_front()
    self.assertEqual(l.front().value, cases[-2])
    self.assertEqual(l.back().value, cases[0])
    l.pop_back()
    self.assertEqual(l.front().value, cases[-2])
    self.assertEqual(l.back().value, cases[1])
    # test insert
    insert_value = 10
    pos = l.front()
    while not pos.value == cases[3]:
      pos = pos.next
    node = l.insert(pos, insert_value)
    self.assertEqual(node.value, insert_value)
    self.assertEqual(node.next, pos)
    self.assertEqual(pos.pre, node)
    # test erase
    l.clear()
    for c in cases:
      l.push_back(c)
    pos = l.front()
    while not pos.value == cases[3]:
      pos = pos.next
    node = l.erase(pos)
    self.assertEqual(node.value, cases[4])
    self.assertEqual(node.pre.value, cases[2])
    self.assertEqual(node.pre.next, node)
    # test move
    value = l.front().value
    l.move_to_back(l.front())
    self.assertEqual(value, l.back().value)
    l.move_to_front(l.back())
    self.assertEqual(value, l.front().value)

if __name__ == '__main__':
  unittest.main()
