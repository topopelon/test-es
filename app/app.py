# class to wait indefinitely

import time


class WaitForever:
    def __init__(self):
        pass

    def wait(self):
        while True:
            time.sleep(100)


# Wait forever
if __name__ == "__main__":
    wait_forever = WaitForever()
    wait_forever.wait()
