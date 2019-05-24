import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym
from collections import deque


# Parameters
learning_rate = 0.001
epsilon = 1
epsilon_decay = 0.001
epsilon_min = 0.01
epsilon_step = 0
gamma = 0.99
batch_size = 20
memory_max = 2000

gym_id = "CartPole-v0"
env = gym.make(gym_id)
action_space = env.action_space.n
state_space = env.observation_space.shape[0]
render_env = False
memories = deque(maxlen=memory_max)

snapshot_freq = 10
save_model_freq = 50
render_by_score_condition = 170  # Renders environment after this average score is reached
model = None


def init_agent(load_path=None):
    global model

    model = keras.Sequential([
        keras.layers.Dense(input_shape=(state_space,), units=64, activation=keras.activations.relu),
        keras.layers.Dense(units=32, activation=keras.activations.relu),
        keras.layers.Dense(units=action_space, activation=keras.activations.linear)
    ])

    if load_path:
        model.load_weights(load_path)

    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate), loss='mse')


def act(state):
    if np.random.rand() > epsilon:
        return np.argmax(model.predict(state))
    else:
        return np.random.randint(0, action_space)


def decay():
    global epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
        epsilon = epsilon_min + (1 - epsilon_min) * np.exp(-epsilon_decay * epsilon_step)


def learn():
    rand_indices = np.random.randint(0, len(memories), size=batch_size)
    memory_batch = [memories[i] for i in rand_indices]

    for memory in memory_batch:

        target = memory["reward"]

        q_vals = model.predict(memory["state"])
        if not memory["done"]:
            target += gamma * np.amax(model.predict(memory["next_state"]))

        q_vals[0][memory["action"]] = target
        model.fit(memory["state"], q_vals, epochs=1, verbose=0)


def memorize(state, action, reward, done, next_state):
    episode = {
        "state": state,
        "action": action,
        "reward": reward,
        "done": done,
        "next_state": next_state
    }
    memories.append(episode)


def save(episode):
    model.save_weights("./saved_models/cartpole_model_episodes_{}".format(episode), overwrite=True)


def load(path):
    model.load_weights(path)


# Initialise game
def run_game(episodes, training=False):
    global render_env, epsilon_step

    scores = []
    snapshots = []
    for episode in range(episodes):

        state = env.reset()
        state = np.array(state.reshape(1, state_space))

        score = 0
        done = False
        while not done:

            if render_env:
                env.render()

            action = act(state)
            decay()

            next_state, reward, done, _ = env.step(action)
            next_state = np.array(next_state.reshape(1, state_space))

            if training:
                memorize(state, action, reward, done, next_state)

            state = next_state

            if len(memories) >= batch_size and training:
                learn()

            score += 1
            epsilon_step += 1

        scores.append(score)

        if (episode + 1) % snapshot_freq == 0:
            snapshot_avg = sum(scores[episode + 1 - snapshot_freq:]) / snapshot_freq
            snapshots.append(snapshot_avg)

            avg = sum(scores) / len(scores)
            print("Episode: {}/{} \t Last score: {:.2f} \t Snapshot Avg: {} \t Total Avg: {:.2f} \t Epsilon: {:.4f}"
                  .format(episode + 1, episodes, score, snapshot_avg, avg, epsilon))

            # After average score hits a target env will be rendered and training stopped
            if snapshot_avg >= render_by_score_condition:
                render_env = True
                training = False
            save(episode)


if __name__ == "__main__":
    print("Starting game...")
    init_agent()
    run_game(1000, True)
