class args:
    epsilon = 0.8
    gamma = 0.9
    epsilon_decrement = 0.5
    num_actions = 5
    num_columns = 20
    num_rows = 20
    state_dimension = 65
    num_agents = 14

    num_epochs = 2000
    learning_rate = 2e-2
    sync_rate = 40
    replay_size = 10000
    batch_size = 64
    patient_factor = 10
    warmup_steps = 1000

    visualize_ckpt = './lightning_logs/version_5/checkpoints/epoch=1999-step=16000.ckpt'
    train_ckpt = './lightning_logs/version_4/checkpoints/epoch=1999-step=16000.ckpt'


class Action:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STATIC = 4