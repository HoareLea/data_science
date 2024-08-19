import time
from multiprocessing import Pool, cpu_count

# Suppose we have a computationally expensive function that takes a long time to complete, and we want to apply this function to a list.
# Waiting for it to run across each element of the list would take ages. However, we can speed it up by running multiple version of the
# function in parallel.


def computationally_expensive_function(a):
    time.sleep(3)
    return a


if __name__ == "__main__":
    # List of numbers
    numbers = range(50)

    # Create a multiprocessing pool with processes set to the number of CPUs
    # Use 'with' statement to automatically manage pool lifecycle and close it when finished
    with Pool(processes=cpu_count()) as p:
        results = p.map(computationally_expensive_function, numbers)

    print(results)


# Note that while multiprocessing can improve performance for CPU-bound tasks, it incurs overhead due to process creation, data
# serialization/deserialization (when passing data between processes), and inter-process communication. Measure performance gains against
# these overheads.

# This syntax initializes a Pool of worker processes and assigns it to the variable pool. The with statement ensures that the Pool is
# properly initialized (__enter__), and it automatically calls pool.close() and pool.join() when leaving the block (__exit__), ensuring
# that all processes are cleaned up correctly.

