
n_parallel_games = 5
gamma = 0.99

agent = FeedforwardAgent("agent", obs_shape, n_actions)



# for each of n_parallel_games, take 10 steps
rollout_obs, rollout_actions, rollout_rewards, rollout_mask = pool.interact(10)



class FeedforwardAgent:

    def __init__(self, name, obs_shape, n_actions, reuse=False):
        """A simple actor-critic agent"""

        with tf.variable_scope(name, reuse=reuse):
            # Note: number of units/filters is arbitrary, you can and should change it at your will
            self.conv0 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')
            self.conv1 = Conv2D(64, (3, 3), strides=(2, 2), activation='relu')
            self.conv2 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')

            self.flatten = Flatten()
            self.hid     = Dense(128, activation='relu')
            self.logits  = Dense(n_actions)
            
            self.state_value = Dense(1)

            # prepare a graph for agent step
            _initial_state = self.get_initial_state(1)

            # prepare placeholders: prev_state and obs_t
            self.prev_state_placeholders = [
                tf.placeholder(mem.dtype, [None] + [mem.shape[i] for i in range(1, mem.ndim)])
                for mem in _initial_state
            ]

            self.obs_t = tf.placeholder('float32', [None, ] + list(obs_shape))


            # make a symbolic_step
            self.next_state, self.agent_outputs = self.symbolic_step(self.prev_state_placeholders, self.obs_t)


    def symbolic_step(self, prev_state, obs_t):
        """Takes agent's previous step and observation, returns next state and whatever it needs to learn (tf tensors)"""

        return new_state, (logits, state_value)


    def get_initial_state(self, batch_size):
        """Return a list of agent memory states at game start. Each state is a np array of shape [batch_size, ...]"""
        # feedforward agent has no state
        return []


    def step(self, prev_state, obs_t):
        """Same as symbolic state except it operates on numpy arrays"""
        sess = tf.get_default_session()
        feed_dict = {self.obs_t: obs_t}
        
        for state_ph, state_value in zip(self.prev_state_placeholders, prev_state):
            feed_dict[state_ph] = state_value
        return sess.run([self.next_state, self.agent_outputs], feed_dict)


    def sample_actions(self, agent_outputs):
        """pick actions given numeric agent outputs (np arrays)"""
        logits, state_values = agent_outputs
        policy = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        return np.array([np.random.choice(len(p), p=p) for p in policy])

