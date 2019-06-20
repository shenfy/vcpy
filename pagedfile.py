import sys
import struct
import lz4.block

class PageDesc():
  kFile = 0
  kDirectory = 1
  kSymLink = 2
  kHardLink = 3
  kPlain = 0
  kLZ4 = 1 << 8

  def __init__(self):
    self.format = 0
    self.start = 0
    self.length = 0
    self.is_compressed = False
    self.uncompressed_length = 0
    self.name = None

class PagedFile():
  def __init__(self):
    self.num_pages = 0
    self.page_table = None
    self.page_order = None
    self.disk_file = None
    self.tail_pos = 0
    self.mode = ''

  @staticmethod
  def __dtype_2_fmt(dtype):
    if dtype == 'int8' or dtype == 'char':
      return 'c'
    elif dtype == 'uint8' or dtype == 'unsigned char':
      return 'B'
    elif dtype == 'bool':
      return '?'
    elif dtype == 'int16' or dtype == 'short':
      return 'h'
    elif dtype == 'uint16' or dtype == 'unsigned short':
      return 'H'
    elif dtype == 'int32' or dtype == 'int':
      return 'i'
    elif dtype == 'uint32' or dtype == 'unsigned int':
      return 'I'
    elif dtype == 'long':
      return 'l'
    elif dtype == 'unsigned long':
      return 'L'
    elif dtype == 'int64' or dtype == 'long long':
      return 'q'
    elif dtype == 'uint64' or dtype == 'unsigned long long':
      return 'Q'
    elif dtype == 'ssize_t':
      return 'n'
    elif dtype == 'size_t':
      return 'N'
    elif dtype == 'half':
      return 'e'
    elif dtype == 'float' or dtype == 'float32':
      return 'f'
    elif dtype == 'double' or dtype == 'float64':
      return 'd'
    elif dtype == 'string':
      return 's'
    elif dtype == 'pointer':
      return 'P'
    return None

  @staticmethod
  def __is_compressed(format):
    return ((format >> 8) & 0xff) > 0

  def __read_unpack(self, dtype, count=1):
    fmt = self.__dtype_2_fmt(dtype)
    if fmt is not None:
      if count > 1:
        fmt = str(count) + fmt
      type_length = struct.calcsize(fmt)
      length = type_length * count
      chunk = self.disk_file.read(length)
      return struct.unpack(fmt, chunk)

  def __pack_write(self, value, dtype, count=1):
    fmt = self.__dtype_2_fmt(dtype)
    if count > 1:
      fmt = str(count) + fmt
    chunk = struct.pack(fmt, value)
    self.disk_file.write(chunk)

  def load(self, filename):
    try:
      self.disk_file = open(filename, 'rb')
    except IOError:
      print('Error: failed to open file {}'.format(filename))
      return False

    # verify magic number
    chunk = self.disk_file.read(4)  # magic number (uint32)
    if chunk.decode('utf-8') != 'PFAR':
      print('Error: not a PFAR file!')
      return False

    self.mode = 'r'

    self.page_table = {}
    self.page_order = []

    # read page table length (int64)
    self.disk_file.seek(-8, 2)
    header_length = self.__read_unpack('int64')[0]
    print('header length: {}'.format(header_length))

    # page table begin
    self.disk_file.seek(-header_length - 8, 2)
    self.tail_pos = self.disk_file.tell()

    # num of pages
    self.num_pages = self.__read_unpack('uint32')[0]
    print('Page count: {}'.format(self.num_pages))

    # read all page descs
    for loc in range(0, self.num_pages):
      desc = PageDesc()
      page_idx = self.__read_unpack('uint32')[0]
      desc.start = self.__read_unpack('uint64')[0]
      desc.length = self.__read_unpack('uint64')[0]
      desc.format = self.__read_unpack('uint16')[0]
      if (self.__is_compressed(desc.format)):
        desc.is_compressed = True
        desc.uncompressed_length = self.__read_unpack('uint64')[0]

      name_length = self.__read_unpack('uint16')[0]
      if name_length > 0:
        desc.name = self.disk_file.read(name_length).decode('utf-8')

      # add to page table
      self.page_table[page_idx] = desc
      self.page_order.append(page_idx)

    return True

  def close(self):
    if self.disk_file is not None:
      if self.mode == 'w':
        self.__write_header()

      self.disk_file.close()
    self.clear()

  def clear(self):
    self.disk_file = None
    self.page_table = {}
    self.page_order = []
    self.num_pages = 0

  def read_page(self, page_idx):
    if page_idx in self.page_table:
      desc = self.page_table[page_idx]
      self.disk_file.seek(desc.start, 0)
      data = self.disk_file.read(desc.length)
      if not desc.is_compressed:
        return data
      else:
        uncompressed = lz4.block.decompress(data, uncompressed_size=desc.uncompressed_length)
        return uncompressed

  def read_page_by_name(self, page_name):
    for page_idx, desc in self.page_table.items():
      if desc.name == page_name:
        return self.read_page(page_idx)

  def print_page_table(self):
    if self.page_table is not None:
      for page_idx, desc in self.page_table.items():
        print('[{:d}] {}: {}({})'.format(page_idx, desc.name, desc.start, desc.length))

  #####################################################################################

  def create(self, filename):
    self.close()
    try:
      self.disk_file = open(filename, 'wb')
    except IOError as e:
      print('Error: failed to create file {}'.format(filename))
      return False

    self.disk_file.write('PFAR'.encode('utf-8'))
    self.tail_pos = self.disk_file.tell()

    self.mode = 'w'

    return True

  def __write_header(self):
    if (self.disk_file is None) or (self.mode != 'w'):
      return False

    self.disk_file.seek(self.tail_pos, 0)
    self.__pack_write(self.num_pages, 'uint32')

    for page_idx in self.page_order:
      desc = self.page_table[page_idx]
      self.__pack_write(page_idx, 'uint32')
      self.__pack_write(desc.start, 'uint64')
      self.__pack_write(desc.length, 'uint64')
      self.__pack_write(desc.format, 'uint16')
      if desc.is_compressed:
        self.__pack_write(desc.uncompressed_length, 'uint64')

      name_length = len(desc.name)
      self.__pack_write(name_length, 'uint16')
      if name_length > 0:
        self.disk_file.write(desc.name.encode('utf-8'))

    header_length = self.disk_file.tell() - self.tail_pos
    self.__pack_write(header_length, 'int64')

    return True

  def add_page(self, idx, name, chunk, compress=False):
    if (self.disk_file is None) or (self.mode != 'w'):
      return False

    if compress:
      compressed = lz4.block.compress(chunk, store_size=False)
      compressed_length = len(compressed)
      print(compressed, compressed_length, len(chunk))

    # add desc
    desc = PageDesc()
    desc.start = self.tail_pos
    desc.is_compressed = compress
    desc.name = name
    if compress:
      desc.format = PageDesc.kLZ4 | PageDesc.kFile
      desc.length = compressed_length
      desc.uncompressed_length = len(chunk)
    else:
      desc.format = PageDesc.kPlain | PageDesc.kFile
      desc.length = len(chunk)
      desc.uncompressed_length = 0

    self.page_table[idx] = desc
    self.num_pages = len(self.page_table)
    self.page_order.append(idx)

    # write data to file
    self.disk_file.seek(self.tail_pos, 0)
    if compress:
      self.disk_file.write(compressed)
    else:
      self.disk_file.write(chunk)
    self.tail_pos = self.disk_file.tell()

    return True

if __name__ == '__main__':
  pass

  # # WRITE EXAMPLE
  # paged_file = PagedFile()
  # paged_file.create('test.pf')
  # paged_file.add_page(0, 'a.txt', 'hello python!'.encode('utf-8'), True)
  # paged_file.add_page(1, 'b.txt', 'bonjour paged file.'.encode('utf-8'), True)
  # paged_file.close()

  # # READ EXAMPLE
  # paged_file = PagedFile()
  # paged_file.load('test.pf')
  # paged_file.print_page_table()
  # data = paged_file.read_page(1)
  # print(data.decode('utf-8'))
  # paged_file.close()