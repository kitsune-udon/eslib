import eslib
import numpy as np
import contextlib

def _cyclic_copy(src, dst, src_offset, dst_offset, size):
    src_cursor = src_offset
    dst_cursor = dst_offset
    remain = size
    while remain > 0:
        copy_size = min(remain, len(src)-src_cursor, len(dst)-dst_cursor)
        dst[dst_cursor:dst_cursor+copy_size] = src[src_cursor:src_cursor+copy_size]
        src_cursor = (src_cursor + copy_size) % len(src)
        dst_cursor = (dst_cursor + copy_size) % len(dst)
        remain -= copy_size

class Perturbation:
    def __init__(self, table_length=10000, random_seed=0):
        state = np.random.get_state()
        np.random.seed(random_seed)
        self.random_table = np.random.normal(size=table_length).astype(eslib.dtype)
        np.random.set_state(state)
        self.t = 0

    def generate(self, shape):
        if not self.within_generation_scope:
            raise RuntimeError

        total_length = np.prod(shape)
        dst = np.empty(total_length, dtype=eslib.dtype)
        src = self.random_table
        src_offset = self.cursor
        dst_offset = 0
        _cyclic_copy(src, dst, src_offset, dst_offset, total_length)
        self.cursor += total_length
        r = dst.reshape(shape)
        if self.current_ptb_id % 2 == 1:
            r *= -1
        return r

    def init_state(self, ptb_id):
        self.current_ptb_id = ptb_id
        i = ptb_id if ptb_id % 2 == 0 else (ptb_id - 1)
        self.cursor = (i + self.t) % len(self.random_table)

    @property
    def within_generation_scope(self):
        return getattr(self, '_within_generation_scope', False)

    @contextlib.contextmanager
    def generation_scope(self, ptb_id):
        self.init_state(ptb_id)
        if self.within_generation_scope:
            raise RuntimeError
        else:
            self._within_generation_scope = True
        try:
            yield
        finally:
            self._within_generation_scope = False

    def age(self):
        self.t += 1


