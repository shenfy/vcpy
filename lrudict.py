from typing import TypeVar, Dict, Tuple, MutableMapping, Iterator
from vcpy.linkedlist import LinkedList, LinkedListNode

class ContainerFullError(Exception):
  pass

KT = TypeVar('KT')
VT = TypeVar('VT')

class LRUDict(MutableMapping[KT, VT]):
  def __init__(self, capacity: int) -> None:
    if capacity <= 0:
      raise ValueError
    self.dict: Dict[KT, Tuple[VT, LinkedListNode[KT]]] = {}
    self.queue: LinkedList[KT] = LinkedList()
    self.capacity = capacity

  def __contains__(self, key: object) -> bool:
    return key in self.dict

  def __len__(self) -> int:
    return len(self.dict)

  def __iter__(self) -> Iterator[KT]:
    return iter(self.dict)

  def __getitem__(self, key: KT) -> VT:
    return self.dict[key][0]

  def __setitem__(self, key: KT, value: VT) -> None:
    # update value if already exists
    if key in self.dict:
      self.dict[key] = (value, self.dict[key][1])
      self.queue.move_to_back(self.dict[key][1])
      return

    # check if container is full
    if self.full():
      del self[self.queue.front().value]

    # insert new value
    self.queue.push_back(key)
    self.dict[key] = (value, self.queue.back())

  def __delitem__(self, key: KT) -> None:
    self.queue.erase(self.dict[key][1])
    del self.dict[key]

  def full(self) -> bool:
    return len(self) >= self.capacity

  def front(self) -> Tuple[KT, VT]:
    key = self.queue.front().value
    return (key, self.dict[key][0])

  def add_item(self, key: KT, value: VT) -> None:
    if self.full():
      raise ContainerFullError()
    self[key] = value

import unittest

class TestLRUDict(unittest.TestCase):
  def test(self):
    cases = {
      1: 2,
      '3': (4,)
    }
    addtional_key = 5
    addtional_value = 6
    d = LRUDict(len(cases))
    # test setitem
    for key, value in cases.items():
      d[key] = value
    # test setitem update
    for key, value in cases.items():
      d[key] = value
    # test contains and getitem
    for key, value in cases.items():
      self.assertIn(key, d)
      self.assertEqual(d[key], value)
    # test len
    self.assertEqual(len(d), len(cases))
    # test iter
    visit_count = {key: 0 for key in cases}
    for key in d:
      visit_count[key] += 1
    self.assertEqual(len(visit_count), len(cases))
    for key in visit_count:
      self.assertEqual(visit_count[key], 1)
    # test delitem
    for key in cases:
      del d[key]
      self.assertNotIn(key, d)
    # test full check
    for key, value in cases.items():
      d[key] = value
    self.assertTrue(d.full())
    self.assertRaises(ContainerFullError, lambda: d.add_item(addtional_key, addtional_value))
    # test front
    front_key, front_value = d.front()
    key, value = next(iter(cases.items()))
    self.assertEqual(front_key, key)
    self.assertEqual(front_value, value)
    # test fifo
    d[addtional_key] = addtional_value
    self.assertEqual(len(d), len(cases))
    self.assertIn(addtional_key, d)
    self.assertEqual(d[addtional_key], addtional_value)
    self.assertNotIn(next(iter(cases.keys())), d)

if __name__ == '__main__':
  unittest.main()
