import Shared_Memory_Interface
import numpy as np

def main():
    smi = Shared_Memory_Interface.SharedMemoryInterface(is_server=False, shm_name="test_shared_memory", shape=(7,), dtype=np.float32)
    try:
        while True:
            pos = smi.read()
            print("Read Position Data:", pos)
            # time.sleep(1.0)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        print("Final Position Data:", pos)
        smi.close()
        print("Shared memory closed.")

if __name__ == "__main__":
    main()