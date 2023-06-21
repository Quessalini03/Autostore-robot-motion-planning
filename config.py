class args:
    epsilon = 0.8
    gamma = 0.9
    epsilon_decrement = 5e-4
    num_actions = 5
    num_columns = 12
    state_dimension = 65
    num_rows = 12
    num_agents = 1

    learning_rate = 2e-3
    sync_rate = 20
    replay_size = 10000
    batch_size = 32
    patient_factor = 10
    warmup_steps = 1000


class Action:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STATIC = 4