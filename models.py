import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec
from collections import deque
import sys

import tools

class reverseAction(tools.Module):
    def __init__(self, hidden_depth, hidden_dim, joint_dim, action_size, dist=None, act=tf.nn.relu, min_std=1e-4, init_std=5, mean_scale=5):
        self._hidden_depth = hidden_depth
        self._hidden_dim = hidden_dim
        self._joint_dim = joint_dim
        self._action_size = action_size
        self._dist = dist
        self._act = act
        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

    def __call__(self, feature_1, feature_2):
        x1 = feature_1
        x2 = feature_2
        x = self.get('concat', tf.keras.layers.Concatenate)([x1, x2])
        x = self.get(f'h1', tf.keras.layers.Dense, self._joint_dim, self._act)(x)
        x = self.get(f'dropout1', tf.keras.layers.Dropout, rate=0.5)(x)
        x = self.get(f'h2', tf.keras.layers.Dense, self._joint_dim, self._act)(x)
        x = self.get(f'dropout2', tf.keras.layers.Dropout, rate=0.5)(x)
        if self._dist is None:
            x = self.get(f'hout', tf.keras.layers.Dense, self._action_size)(x)
            return x
        elif self._dist == 'tanh':
            raw_init_std = np.log(np.exp(self._init_std) - 1)
            x = self.get(f'hout', tf.keras.layers.Dense, 2 * self._action_size)(x)
            mean, std = tf.split(x, 2, -1)
            mean = self._mean_scale * tf.tanh(mean / self._mean_scale)
            std = tf.nn.softplus(std + raw_init_std) + self._min_std
            dist = tfd.Normal(mean, std)
            dist = tfd.TransformedDistribution(dist, tools.TanhBijector())
            dist = tfd.Independent(dist, 1)
            dist = tools.SampleDist(dist)
            return dist
        else:
            raise NotImplementedError(self._dist)
        return

class latentRSSM(tools.Module):
    def __init__(self, stoch_size, deter_size=200, hidden_size=200, activation=tf.nn.relu, num_latent_sample=1):
        super().__init__()
        self._activation = activation
        self._stoch_size = stoch_size
        self._deter_size = deter_size
        self._hidden_size = hidden_size
        self._gru = tf.keras.layers.GRUCell(self._deter_size)
        self._num_latent_sample = num_latent_sample
    
    def initial(self, batch_size):
        dtype = prec.global_policy().compute_dtype
        return dict(
            mean=tf.zeros([batch_size, self._stoch_size], dtype), 
            std=tf.zeros([batch_size, self._stoch_size], dtype), 
            stoch=tf.zeros([batch_size, self._stoch_size], dtype), 
            internal=self._gru.get_initial_state(None, batch_size, dtype))
    
    @tf.function
    def observe(self, embed, action, k=1, state=None, num_latent_sample=None):
        if state is None:
            state = self.initial(tf.shape(action)[0])
        if num_latent_sample is None:
            num_latent_sample = self._num_latent_sample
        embed = tf.transpose(embed, [1, 0, 2]) 
        action = tf.transpose(action, [1, 0, 2])
        assert isinstance(k, int)
        last_prior = None
        if k == 1:
            post, prior = tools.static_scan(
                lambda prev, inputs: self.obs_step(prev[0], *inputs, num_latent_sample=num_latent_sample), 
                (action, embed), 
                (state, state)
            )
            post = {k: tf.transpose(v, [1, 0, 2]) for k, v in post.items()}
            prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
            last_prior = prior
        else:
            embed = embed[(k-1):]
            action = tools.k_steps_action(action, k, action.shape[0])
            action = tf.stack(action, axis=0)
            num_valid_states = len(action)
            prior_list, post_list, last_prior_list = [], [], []
            for i in range(num_valid_states):
                post, priors, last_prior = self.obs_k_step(state, action[i], embed[i], num_latent_sample)
                if i >= k:
                    state = post_list[-k]
                # priors = {k: tf.transpose(v, [1, 0, 2]) for k, v in priors.items()}
                prior_list.append(priors)
                post_list.append(post)
                last_prior_list.append(last_prior)
            post = {k: tf.stack([p[k] for p in post_list], axis=0) for k in post_list[0].keys()}
            last_prior = {k: tf.stack([p[k] for p in last_prior_list], axis=0) for k in last_prior_list[0].keys()}
            post = {k: tf.transpose(v, [1, 0, 2]) for k, v in post.items()}
            last_prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in last_prior.items()}
            # prior = {k: tf.stack([p[k] for p in prior_list], axis=0) for k in priors.keys()}
            prior = prior_list
        return post, prior, last_prior

    @tf.function
    def retrace(self, reverse_actor, priors, posts, latent_prediction_steps=1, num_latent_sample=None):
        if num_latent_sample is None:
            num_latent_sample = self._num_latent_sample
        if latent_prediction_steps == 1:
            next_state = {k: v[:, 1:, :] for k, v in posts.items()}
            prev_state = {k: v[:, :-1, :] for k, v in priors.items()}

            next_feature, prev_feature = self.get_feature(next_state), self.get_feature(prev_state)
            reverse_action = reverse_actor(next_feature, prev_feature)
            retraced_states = self.imagine_step(next_state, reverse_action, num_latent_sample)
        else:
            num_valid_states = len(priors)
            K = tf.shape(priors[0]['mean'])[1] # latent learning horizon
            retraced_states = deque(maxlen=(num_valid_states))
            for i in range(num_valid_states):
                prior = priors[i]
                # prior = {k: v[:, [i], ...] for k, v in prior.items()}
                prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
                prev_state = {k: v[:, i, ...][None] for k, v in posts.items()}
                # tf.print(i, output_stream=sys.stderr)
                # tf.print(i, output_stream=sys.stdout)
                for j in reversed(range(1, 3)):
                    # tf.print(j, output_stream=sys.stdout)
                    next_prior, prev_prior = {k: v[j, ...][None] for k, v in prior.items()}, {k: v[j-1, ...][None] for k, v in prior.items()}
                    next_feature, prev_feature = self.get_feature(next_prior), self.get_feature(prev_prior)
                    reverse_action = reverse_actor(next_feature, prev_feature)
                    prev_state = self.imagine_step(prev_state, reverse_action, num_latent_sample)
                if i != 0:
                    retraced_states.append(prev_state)
            retraced_states = {k: tf.stack([r[k][0] for r in retraced_states], axis=0) for k in retraced_states[0].keys()}
            retraced_states = {k: tf.transpose(v, [1, 0, 2]) for k, v in retraced_states.items()}
        return retraced_states

    @tf.function
    def imagine(self, action, state=None, num_latent_sample=None):
        if state is None:
            state = self.initial(tf.shape(action)[0])
        if num_latent_sample is None:
            num_latent_sample = self._num_latent_sample
        assert isinstance(state, dict), state
        action = tf.transpose(action, [1, 0, 2])
        prior = tools.static_scan(lambda prev, inputs: self.imagine_step(prev, inputs, num_latent_sample=num_latent_sample), action, state)
        prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
        return prior
    
    @tf.function
    def obs_k_step(self, prev_state, actions, embed, num_latent_sample):
        priors = self.imagine_k_step(prev_state, actions, num_latent_sample)
        last_prior = {k: v[:, -1, ...] for k, v in priors.items()}
        x = tf.concat([last_prior['internal'], embed], -1)
        x = self.get('obs1', tf.keras.layers.Dense, self._hidden_size, self._activation)(x)
        x = self.get('obs2', tf.keras.layers.Dense, 2*self._stoch_size, None)(x)
        mean, std = tf.split(x, 2, -1)
        std = tf.nn.softplus(std) + 0.1
        stoch = self.get_distribution({'mean': mean, 'std': std}).sample()
        post = {'mean': mean, 'std': std, 'stoch': stoch, 'internal': last_prior['internal']}
        return post, priors, last_prior
    
    @tf.function
    def obs_step(self, prev_state, prev_action, embed, num_latent_sample):
        # we do one-step update here as we have ground-truth external supervision
        prior = self.imagine_step(prev_state, prev_action, num_latent_sample)
        x = tf.concat([prior['internal'], embed], -1)
        x = self.get('obs1', tf.keras.layers.Dense, self._hidden_size, self._activation)(x)
        x = self.get('obs2', tf.keras.layers.Dense, self._stoch_size*2, None)(x)
        mean, std = tf.split(x, 2, -1)
        std = tf.nn.softplus(std) + 0.1
        stoch = self.get_distribution({'mean': mean, 'std': std}).sample()
        post = {'mean': mean, 'std': std, 'stoch': stoch, 'internal': prior['internal']}
        return post, prior

    @tf.function
    def imagine_step(self, prev_state, prev_action, num_latent_sample=None):
        dist = self.get_distribution(prev_state)
        prev_internal = [prev_state['internal']]
        mean, std = 0.0, 0.0
        internal = 0
        if num_latent_sample is None:
            num_latent_sample = self._num_latent_sample
        for i in range(num_latent_sample):
            x = tf.concat([dist.sample(), prev_action], -1)
            # x = tf.reshape(x, (1, -1))
            x = self.get('imagine1', tf.keras.layers.Dense, self._hidden_size, self._activation)(x)
            x, internal_temp = self._gru(x, prev_internal)
            internal_temp = internal_temp[0]
            internal += internal_temp
            x = self.get('imagine2', tf.keras.layers.Dense, self._hidden_size, self._activation)(x)
            x = self.get('imagine3', tf.keras.layers.Dense, self._stoch_size*2, None)(x)
            mean_temp, std_temp = tf.split(x, 2, -1)
            std_temp = tf.nn.softplus(std_temp) + 0.1
            mean += mean_temp
            std += std_temp
        mean = mean / num_latent_sample
        std = std / num_latent_sample
        internal = internal / num_latent_sample
        stoch = self.get_distribution({'mean': mean, 'std': std}).sample()
        prior = {'mean': mean, 'std': std, 'stoch': stoch, 'internal': internal}
        return prior
    
    @tf.function
    def imagine_k_step(self, prev_state, actions, num_latent_sample=None):
        K = tf.shape(actions)[0]
        # priors = tf.TensorArray(tf.float16, size=K+1)
        # priors = deque()
        d = {k: tf.TensorArray(tf.float16, K+1) for k in prev_state.keys()}
        prior = prev_state
        # priors = priors.write(0, prior)
        # priors.append(prior)
        d = self.cache_TensorArray(d, 0, prior)
        if num_latent_sample is None:
            num_latent_sample = self._num_latent_sample
        for i in range(K):
            prior = self.imagine_step(prior, actions[i], num_latent_sample)
            # priors = priors.write(i+1, prior)
            # priors.append(prior)
            d = self.cache_TensorArray(d, i+1, prior)
        priors = {k: v.stack() for k, v in d.items()}
        # print(priors['mean'].shape)
        # priors = {k: tf.stack([p[k] for p in d], axis=0) for k in d[0].keys()}
        priors = {k: tf.transpose(v, [1, 0, 2]) for k, v in priors.items()}
        return priors
    
    def cache_TensorArray(self, cached_d, i, prior):
        for k in prior.keys():
            cached_d[k] = cached_d[k].write(i, prior[k])
        return cached_d

    def get_distribution(self, state):
        return tfd.MultivariateNormalDiag(state['mean'], state['std'])

    def get_feature(self, state):
        return tf.concat([state['stoch'], state['internal']], -1)


class ConvEncoder(tools.Module):

  def __init__(self, depth=32, act=tf.nn.relu):
    self._act = act
    self._depth = depth

  def __call__(self, obs):
    kwargs = dict(strides=2, activation=self._act)
    x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-3:])) 
    x = self.get('h1', tf.keras.layers.Conv2D, 1 * self._depth, 4, **kwargs)(x) 
    x = self.get('h2', tf.keras.layers.Conv2D, 2 * self._depth, 4, **kwargs)(x)
    x = self.get('h3', tf.keras.layers.Conv2D, 4 * self._depth, 4, **kwargs)(x)
    x = self.get('h4', tf.keras.layers.Conv2D, 8 * self._depth, 4, **kwargs)(x)
    shape = tf.concat([tf.shape(obs['image'])[:-3], [32 * self._depth]], 0)
    return tf.reshape(x, shape)


class ConvDecoder(tools.Module):
    def __init__(self, depth=32, act=tf.nn.relu, shape=(64, 64, 3)):
        self._act = act
        self._depth = depth
        self._shape = shape

    def __call__(self, features):
        kwargs = dict(strides=2, activation=self._act)
        x = self.get('h1', tf.keras.layers.Dense, 32 * self._depth, None)(features)
        x = tf.reshape(x, [-1, 1, 1, 32 * self._depth])
        x = self.get('h2', tf.keras.layers.Conv2DTranspose, 4 * self._depth, 5, **kwargs)(x)
        x = self.get('h3', tf.keras.layers.Conv2DTranspose, 2 * self._depth, 5, **kwargs)(x)
        x = self.get('h4', tf.keras.layers.Conv2DTranspose, 1 * self._depth, 6, **kwargs)(x)
        x = self.get('h5', tf.keras.layers.Conv2DTranspose, self._shape[-1], 6, strides=2)(x)
        mean = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
        return tfd.Independent(tfd.Normal(mean, 1), len(self._shape))


class DenseDecoder(tools.Module):
    def __init__(self, shape, layers, units, dist='normal', act=tf.nn.elu):
        self._shape = shape
        self._layers = layers
        self._units = units
        self._dist = dist
        self._act = act

    def __call__(self, features):
        x = features
        for index in range(self._layers):
            x = self.get(f'h{index}', tf.keras.layers.Dense, self._units, self._act)(x)
        x = self.get(f'hout', tf.keras.layers.Dense, np.prod(self._shape))(x)
        x = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
        if self._dist == 'normal':
            return tfd.Independent(tfd.Normal(x, 1), len(self._shape)) 
        if self._dist == 'binary':
            return tfd.Independent(tfd.Bernoulli(x), len(self._shape))
        raise NotImplementedError(self._dist)


class ActionDecoder(tools.Module):
    def __init__(self, size, layers, units, dist='tanh_normal', act=tf.nn.elu, min_std=1e-4, init_std=5, mean_scale=5):
        self._size = size
        self._layers = layers
        self._units = units
        self._dist = dist
        self._act = act
        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

    def __call__(self, features):
        raw_init_std = np.log(np.exp(self._init_std) - 1) # sicne the init_std was computed via applying the softplus function
        x = features
        for index in range(self._layers):
            x = self.get(f'h{index}', tf.keras.layers.Dense, self._units, self._act)(x)
        if self._dist == 'tanh_normal':
            x = self.get(f'hout', tf.keras.layers.Dense, 2 * self._size)(x)
            mean, std = tf.split(x, 2, -1)
            mean = self._mean_scale * tf.tanh(mean / self._mean_scale)
            std = tf.nn.softplus(std + raw_init_std) + self._min_std
            dist = tfd.Normal(mean, std)
            dist = tfd.TransformedDistribution(dist, tools.TanhBijector())
            dist = tfd.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == 'onehot':
            x = self.get(f'hout', tf.keras.layers.Dense, self._size)(x)
            dist = tools.OneHotDist(x)
        else:
            raise NotImplementedError(dist)
        return dist
