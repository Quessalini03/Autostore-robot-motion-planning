class args:
    epsilon = 0.8
    gamma = 0.8
    epsilon_decrement = 0.00001
    num_actions = 5
    num_columns = 13
    num_rows = 13
    state_dimension = 65
    num_agents = 10

    num_epochs = 10000
    learning_rate = 0.0001
    sync_rate = 5
    replay_size = 1000
    batch_size = 128
    patient_factor = 10
    warmup_steps = 1000
    time_to_live = 150

    visualize_ckpt = './lightning_logs/version_5/checkpoints/epoch=1999-step=16000.ckpt'
    train_ckpt = './lightning_logs/version_4/checkpoints/epoch=1999-step=16000.ckpt'

    pt_checkpoint = './saved_models/1.pt'


class Action:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STATIC = 4
