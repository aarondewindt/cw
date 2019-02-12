import msgpack
import msgpack_numpy
from functools import partial

msgpack_numpy.patch()

dumps = msgpack.dumps
dump = msgpack.dump

loads = partial(msgpack.loads, encoding="utf-8")
load = partial(msgpack.load, encoding="utf-8")
