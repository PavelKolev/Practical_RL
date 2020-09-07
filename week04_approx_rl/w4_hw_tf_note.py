import sys, os
if 'google.colab' in sys.modules and not os.path.exists('.setup_complete'):
#     %tensorflow_version 1.x
    !wget -q https://raw.githubusercontent.com/yandexdataschool/Practical_RL/spring20/setup_colab.sh -O- | bash
        
    !wget -q https://raw.githubusercontent.com/yandexdataschool/Practical_RL/master/week04_approx_rl/atari_wrappers.py
    !wget -q https://raw.githubusercontent.com/yandexdataschool/Practical_RL/master/week04_approx_rl/utils.py
    !wget -q https://raw.githubusercontent.com/yandexdataschool/Practical_RL/master/week04_approx_rl/replay_buffer.py
    !wget -q https://raw.githubusercontent.com/yandexdataschool/Practical_RL/master/week04_approx_rl/framebuffer.py

    !touch .setup_complete

if type(os.environ.get("DISPLAY")) is not str or len(os.environ.get("DISPLAY")) == 0:
    !bash ../xvfb start
    os.environ['DISPLAY'] = ':1'


import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gym
# spawn game instance for tests
env = gym.make("BreakoutNoFrameskip-v4")  # create raw env

observation_shape = env.observation_space.shape
n_actions = env.action_space.n


class PreprocessAtari(ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops, scales image into the desired shapes and optionally grayscales it."""
        ObservationWrapper.__init__(self, env)

        self.img_size = (64, 64, 1)
        self.observation_space = Box(0.0, 1.0, self.img_size)

    def observation(self, img):
        """what happens to each observation"""

        # Here's what you need to do:
        #  * Crop image, remove irrelevant parts.
        #  * Resize image to self.img_size. Use cv2.resize or any other library you want,
        #    e.g. PIL or Keras. Do not use skimage.transform.resize because it is roughly
        #    6x slower than cv2.resize.
        #  * Cast image to grayscale.
        #  * Convert image pixels to (0, 1) range, float32 type.

        img = img[50:-5,5:-5]
        img = cv2.resize(img, (64, 64))
        img = rgb2gray(img)
        img = np.expand_dims(img,-1)
        img = img_as_float(img)
        img = np.float32(img)
        
        return img


import gym
# spawn game instance for tests
env = gym.make("BreakoutNoFrameskip-v4")  # create raw env
env = PreprocessAtari(env)

observation_shape = env.observation_space.shape
n_actions = env.action_space.n

obs = env.reset()




from framebuffer import FrameBuffer

def make_env():
    env = gym.make("BreakoutNoFrameskip-v4")
    env = PreprocessAtari(env)
    env = FrameBuffer(env, n_frames=4, dim_order='tensorflow')
    return env

env = make_env()
env.reset()

n_actions = env.action_space.n
state_dim = env.observation_space.shape





import tensorflow as tf
tf.reset_default_graph()
sess = tf.InteractiveSession()



class DQNAgent:
    def __init__(self, name, state_shape, n_actions, epsilon=0, reuse=False):
        """A simple DQN agent"""
        with tf.variable_scope(name, reuse=reuse):

            #<YOUR CODE: define your network body here. Please make sure you don't use any layers created elsewhere>
            self.model = tf.keras.models.Sequential([
                        tf.keras.layers.Conv2D(16, (3,3), strides=2,
                                               activation='relu', 
                                               input_shape = state_shape),
                        #tf.keras.layers.MaxPooling2D(2, 2),
                        tf.keras.layers.Conv2D(32, (3,3), strides=2,
                                               activation='relu'),
                        #tf.keras.layers.MaxPooling2D(2, 2),
                        tf.keras.layers.Conv2D(64, (3,3), strides=2,
                                               activation='relu'),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(256, activation='relu'),
                        tf.keras.layers.Dense(n_actions)
            ])

            # prepare a graph for agent step
            self.state_t = tf.placeholder('float32', [None, ] + list(state_shape))
            # self.qvalues_t = self.model(self.state_t)
            self.qvalues_t = self.get_symbolic_qvalues(self.state_t)

        self.weights = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        self.epsilon = epsilon

    def get_symbolic_qvalues(self, state_t):
        """takes agent's observation, returns qvalues. Both are tf Tensors"""
        #<YOUR CODE: apply your network layers here>
        qvalues = self.model(state_t)

        assert tf.is_numeric_tensor(qvalues) and qvalues.shape.ndims == 2, \
            "please return 2d tf tensor of qvalues [you got %s]" % repr(qvalues)
        assert int(qvalues.shape[1]) == n_actions

        return qvalues

    def get_qvalues(self, state_t):
        """Same as symbolic step except it operates on numpy arrays"""
        sess = tf.get_default_session()
        return sess.run(self.qvalues_t, {self.state_t: state_t})

    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape
        
        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)
        should_explore = np.random.choice([0, 1], batch_size, p=[1-epsilon, epsilon])

        return np.where(should_explore, random_actions, best_actions)

agent = DQNAgent("dqn_agent", state_dim, n_actions, epsilon=0.5)
sess.run(tf.global_variables_initializer())



def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    for _ in range(n_games):
        s = env.reset()
        reward = 0
        for _ in range(t_max):
            qvalues = agent.get_qvalues([s])
            action = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            s, r, done, _ = env.step(action)
            reward += r
            if done:
                break

        rewards.append(reward)
    return np.mean(rewards)

evaluate(env, agent, n_games=1)



def play_and_record(agent, env, exp_replay, n_steps=1):
    """
    Play the game for exactly n steps, record every (s,a,r,s', done) to replay buffer. 
    Whenever game ends, add record with done=True and reset the game.
    It is guaranteed that env has done=False when passed to this function.

    PLEASE DO NOT RESET ENV UNLESS IT IS "DONE"

    :returns: return sum of rewards over time
    """
    # initial state
    s = env.framebuffer

    # Play the game for n_steps as per instructions above
    sum_rewards = 0.
    
    for _ in range(n_steps):
        qvalues = agent.get_qvalues([s])
        a = agent.sample_actions(qvalues)[0]
        s_next, r, done, _ = env.step(a)
        sum_rewards += r

        exp_replay.add(s, a, r, s_next, done)
        s = s_next

        if done:
            s = env.reset()

    return sum_rewards


# testing your code. This may take a minute...
exp_replay = ReplayBuffer(20000)

play_and_record(agent, env, exp_replay, n_steps=10000)



# SECOND MODEL
target_network = DQNAgent("target_network", state_dim, n_actions)


def load_weigths_into_target_network(agent, target_network):
    """ assign target_network.weights variables to their respective agent.weights values. """
    assigns = []
    for w_agent, w_target in zip(agent.weights, target_network.weights):
        assigns.append(tf.assign(w_target, w_agent, validate_shape=True))
    
    #tf.get_default_session().run(assigns)
    return assigns

# create the tf copy graph only once.
copy_step = load_weigths_into_target_network(agent, target_network)
sess.run(copy_step)



# DEFINE THE LOSS FUNCTION !!!


# placeholders that will be fed with exp_replay.sample(batch_size)
obs_ph = tf.placeholder(tf.float32, shape=(None,) + state_dim)
actions_ph = tf.placeholder(tf.int32, shape=[None])
rewards_ph = tf.placeholder(tf.float32, shape=[None])
next_obs_ph = tf.placeholder(tf.float32, shape=(None,) + state_dim)
is_done_ph = tf.placeholder(tf.float32, shape=[None])

is_not_done = 1 - is_done_ph
gamma = 0.99


current_qvalues = agent.get_symbolic_qvalues(obs_ph)
current_action_qvalues = tf.reduce_sum(tf.one_hot(actions_ph, n_actions) * current_qvalues, axis=1)


# *** Q-Learning with Source and Target Model ***

#next_qvalues_target = <YOUR CODE: compute q-values for NEXT states with target network>
next_qvalues_target = target_network.get_symbolic_qvalues(next_obs_ph)

#next_state_values_target = <YOUR CODE: compute state values by taking max over next_qvalues_target for all actions>
best_actions = tf.reduce_max(next_qvalues_target, axis=1)

#reference_qvalues = <YOUR CODE: compute Q_reference(s,a) as per formula above>
reference_qvalues = rewards_ph + gamma * best_actions * is_not_done

# Define loss function for sgd.
td_loss = (current_action_qvalues - tf.stop_gradient(reference_qvalues)) ** 2
td_loss = tf.reduce_mean(td_loss)

train_step = tf.train.AdamOptimizer(1e-4).minimize(td_loss, var_list=agent.weights)

sess.run(tf.global_variables_initializer())




"""### Main loop

It's time to put everything together and see if it learns anything.
"""

# Commented out IPython magic to ensure Python compatibility.
from tqdm import trange
import pandas as pd
from IPython.display import clear_output
import matplotlib.pyplot as plt
# %matplotlib inline

def moving_average(x, span=100, **kw):
    return pd.DataFrame({'x': np.asarray(x)}).x.ewm(span=span, **kw).mean().values

mean_rw_history = []
td_loss_history = []

exp_replay = ReplayBuffer(10**5)
play_and_record(agent, env, exp_replay, n_steps=10000)


def sample_batch(exp_replay, batch_size):
    obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = exp_replay.sample(batch_size)
    return {
        obs_ph: obs_batch,
        actions_ph: act_batch,
        rewards_ph: reward_batch,
        next_obs_ph: next_obs_batch,
        is_done_ph: is_done_batch,
    }

for i in trange(10**5):
    # play
    play_and_record(agent, env, exp_replay, 10)

    # train
    _, loss_t = sess.run([train_step, td_loss], sample_batch(exp_replay, batch_size=64))
    td_loss_history.append(loss_t)

    # adjust agent parameters
    if i % 500 == 0:
        # You could think that loading weights onto a target network is simply
        #     load_weigths_into_target_network(agent, target_network)
        # but actually calling this function repeatedly creates a TF copy operator
        # again and again, which bloats memory consumption with each training step.
        # Instead, you should create 'copy_step' once.
        sess.run(copy_step)
        agent.epsilon = max(agent.epsilon * 0.99, 0.01)
        mean_rw_history.append(evaluate(make_env(), agent, n_games=3))

    if i % 100 == 0:
        clear_output(True)
        print("buffer size = %i, epsilon = %.5f" % (len(exp_replay), agent.epsilon))

        plt.subplot(1, 2, 1)
        plt.title("mean reward per game")
        plt.plot(mean_rw_history)
        plt.grid()

        assert not np.isnan(loss_t)
        plt.figure(figsize=[12, 4])
        plt.subplot(1, 2, 2)
        plt.title("TD loss history (moving average)")
        plt.plot(moving_average(np.array(td_loss_history), span=100, min_periods=100))
        plt.grid()
        plt.show()

assert np.mean(mean_rw_history[-10:]) > 10.
print("That's good enough for tutorial.")








agent.epsilon = 0

# Record sessions

import gym.wrappers

with gym.wrappers.Monitor(make_env(), directory="videos", force=True) as env_monitor:
    sessions = [evaluate(env_monitor, agent, n_games=1) for _ in range(100)]

# Show video. This may not work in some setups. If it doesn't
# work for you, you can download the videos and view them locally.

from pathlib import Path
from IPython.display import HTML

video_names = sorted([s for s in Path('videos').iterdir() if s.suffix == '.mp4'])

HTML("""
<video width="640" height="480" controls>
  <source src="{}" type="video/mp4">
</video>
""".format(video_names[-1]))  # You can also try other indices

