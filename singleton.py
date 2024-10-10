import numpy as np


class RandomStateSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RandomStateSingleton, cls).__new__(cls)
            cls._instance.random_state = np.random.RandomState(seed=42)
        return cls._instance


# Usage
random_instance1 = RandomStateSingleton()
random_instance2 = RandomStateSingleton()

# Ensuring that both instances are the same
assert random_instance1 is random_instance2

# Generating random numbers using the singleton
print(random_instance1.random_state.rand(3))  # [0.37454012 0.95071431 0.73199394]
print(random_instance2.random_state.rand(3))  # [0.59865848 0.15601864 0.15599452]
