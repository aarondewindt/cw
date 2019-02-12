import msgpack
import msgpack_numpy
from functools import partial

msgpack_numpy.patch()

dumps = msgpack.dumps
dump = msgpack.dump

loads = partial(msgpack.loads, raw=False)
load = partial(msgpack.load, raw=False)
