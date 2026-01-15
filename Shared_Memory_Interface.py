from multiprocessing import shared_memory, Lock
from multiprocessing.managers import BaseManager
import numpy as np

class LockManager(BaseManager):
    pass

class SharedMemoryInterface:
    def __init__(self, is_server, shm_name, shape=(7,), dtype=np.float32, address=('localhost', 50000), authkey=b'abc'):
        self.is_server = is_server
        self.shm_name = shm_name
        self.shape = shape
        self.dtype = dtype
        self.address = address
        self.authkey = authkey

        if self.is_server:
            # Create shared memory
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=True, size=np.zeros(self.shape, dtype=self.dtype).nbytes)
            # Create Lock
            lock = Lock()
            LockManager.register('get_lock', callable=lambda: lock)
            self.manager = LockManager(address=self.address, authkey=self.authkey)
            self.manager.start()
            self.lock = self.manager.get_lock()

        else:
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=False)
            LockManager.register('get_lock')
            self.manager = LockManager(address=self.address, authkey=self.authkey)
            self.manager.connect()
            self.lock = self.manager.get_lock()
        self.buffer = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
        
    def write(self, data):
        self.lock.acquire()
        try:
            self.buffer[:] = data[:]
        finally:
            self.lock.release()

    def read(self):
        self.lock.acquire()
        try:
            return self.buffer[:]
        finally:
            self.lock.release()
            
    def close(self):
        self.shm.close()
        if self.is_server:
            self.shm.unlink()
        self.manager.shutdown()