import threading
import time


def task(name):
    print(f"Task {name} starting")
    time.sleep(2)
    print(f"Task {name} completed")


def main():
    threads = []
    for i in range(5):
        thread = threading.Thread(target=task, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
