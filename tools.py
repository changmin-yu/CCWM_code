import datetime
import io
import pathlib
import pickle
import re
import uuid
import os
import matplotlib.pyplot as plt

import gym
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_probability as tfp
from tensorflow.keras.mixed_precision import experimental as prec
from tensorflow_probability import distributions as tfd
from sklearn.manifold import TSNE

class AttrDict(dict):

    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


class Module(tf.Module):

    def save(self, filename):
        values = tf.nest.map_structure(lambda x: x.numpy(), self.variables)
        with pathlib.Path(filename).open('wb') as f:
            pickle.dump(values, f)

    def load(self, filename):
        with pathlib.Path(filename).open('rb') as f:
            values = pickle.load(f)
        tf.nest.map_structure(lambda x, y: x.assign(y), self.variables, values)

    def get(self, name, ctor, *args, **kwargs):
        if not hasattr(self, '_modules'):
            self._modules = {}
        if name not in self._modules:
            self._modules[name] = ctor(*args, **kwargs)
        return self._modules[name]


def nest_summary(structure):
    if isinstance(structure, dict):
        return {k: nest_summary(v) for k, v in structure.items()}
    if isinstance(structure, list):
        return [nest_summary(v) for v in structure]
    if hasattr(structure, 'shape'):
        return str(structure.shape).replace(', ', 'x').strip('(), ')
    return '?'


def graph_summary(writer, fn, *args):
    step = tf.summary.experimental.get_step()
    def inner(*args):
            tf.summary.experimental.set_step(step)
            with writer.as_default():
                fn(*args)
    return tf.numpy_function(inner, args, []) # apply inner to the args


def video_summary(name, video, step=None, fps=20):
    # name = name if isinstance(name, str) else name.decode('utf-8')
    # name = name if isinstance(name, str) else name.item()
    if isinstance(name, np.ndarray):
        name = name.item()
    else:
        name = name if isinstance(name, str) else name.decode('utf-8')
    if np.issubdtype(video.dtype, np.floating):
        video = np.clip(255 * video, 0, 255).astype(np.uint8)
    B, T, H, W, C = video.shape
    try:
        frames = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
        summary = tf1.Summary()
        image = tf1.Summary.Image(height=B * H, width=T * W, colorspace=C)
        image.encoded_image_string = encode_gif(frames, fps)
        summary.value.add(tag=name + '/gif', image=image)
        tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
    except (IOError, OSError) as e:
        print('GIF summaries require ffmpeg in $PATH.', e)
        frames = video.transpose((0, 2, 1, 3, 4)).reshape((1, B * H, T * W, C))
        tf.summary.image(name + '/grid', frames, step)


def encode_gif(frames, fps):
    from subprocess import Popen, PIPE
    h, w, c = frames[0].shape
    pxfmt = {1: 'gray', 3: 'rgb24'}[c]
    cmd = ' '.join([
        f'ffmpeg -y -f rawvideo -vcodec rawvideo',
        f'-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex',
        f'[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
        f'-r {fps:.02f} -f gif -'])
    proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in frames:
        proc.stdin.write(image.tostring())
    out, err = proc.communicate()
    if proc.returncode:
        raise IOError('\n'.join([' '.join(cmd), err.decode('utf8')]))
    del proc
    return out


def linear_annealing_scheme(curr_ratio, start, end=0.0):
    return max(end+(start-end)*curr_ratio, end)

def simulate(agent, envs, steps=0, episodes=0, state=None, truncation_annealing_period=None, done_truncation_init=0.0, truncation_max_episode=1000, 
             current_step=None):
    if state is None:
        step, episode = 0, 0
        done = np.ones(len(envs), np.bool)
        length = np.zeros(len(envs), np.int32)
        obs = [None] * len(envs)
        agent_state = None
    else:
        step, episode, done, length, obs, agent_state = state
    if truncation_annealing_period is not None:
        max_truncation_episode =  int(truncation_annealing_period * truncation_max_episode)
    truncation_ratio = done_truncation_init
    while (steps and step < steps) or (episodes and episode < episodes):
        if done.any():
            indices = [index for index, d in enumerate(done) if d]
            promises = [envs[i].reset(blocking=False) for i in indices] # at the steps at which the termination is reached, reset the env
            for index, promise in zip(indices, promises):
                obs[index] = promise()
        if truncation_annealing_period is not None:
            truncation_ratio = linear_annealing_scheme(tf.cast(1 - episode / max_truncation_episode, dtype=tf.float16), done_truncation_init)
        obs = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
        action, agent_state = agent(obs, done, agent_state, current_step=current_step)
        action = np.array(action)
        assert len(action) == len(envs)
        promises = [e.step(a, blocking=False, truncation_ratio=truncation_ratio) for e, a in zip(envs, action)]
        obs, _, done = zip(*[p()[:3] for p in promises])
        obs = list(obs)
        done = np.stack(done)
        episode += int(done.sum())
        length += 1
        step += (done * length).sum()
        length *= (1 - done)
    return (step - steps, episode - episodes, done, length, obs, agent_state)

def simulate_eval(agent, envs, steps=0, episodes=0, state=None):
    if state is None:
        step, episode = 0, 0
        done = np.ones(len(envs), np.bool)
        length = np.zeros(len(envs), np.int32)
        obs = [None] * len(envs)
        agent_state = None
    else:
        step, episode, done, length, obs, agent_state = state
    while (steps and step < steps) or (episodes and episode < episodes):
        if done.any():
            indices = [index for index, d in enumerate(done) if d]
            promises = [envs[i].reset(blocking=False) for i in indices]
            for index, promise in zip(indices, promises):
                obs[index] = promise()

        obs = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
        action, agent_state = agent(obs, done, agent_state)
        action = np.array(action)
        assert len(action) == len(envs)
        if step % 100 == 0:
            obs = envs[0].render(480, 480, camera_id=0)
            plt.figure()
            plt.imshow(obs)
            plt.savefig(f'./logdir_figs/dmc_walker_walk_{step}.pdf')
            plt.savefig(f'./logdir_figs/dmc_walker_walk_{step}.png')
        promises = [e.step(a, blocking=False) for e, a in zip(envs, action)]
        obs, _, done = zip(*[p()[:3] for p in promises])
        obs = list(obs)
        done = np.stack(done)
        episode += int(done.sum())
        length += 1
        step += (done * length).sum()
        length *= (1 - done)
    return (step - steps, episode - episodes, done, length, obs, agent_state)

def simulate_reverse_path(agent, envs, steps=0, episodes=0, state=None):
    agent_state_list = []
    prior_list = []
    if state is None:
        step, episode = 0, 0
        done = np.ones(len(envs), np.bool)
        length = np.zeros(len(envs), np.int32)
        obs = [None] * len(envs)
        agent_state = None
    else:
        step, episode, done, length, obs, agent_state = state
    while (steps and step < steps) or (episodes and episode < episodes):
        if done.any():
            indices = [index for index, d in enumerate(done) if d]
            promises = [envs[i].reset(blocking=False) for i in indices] # at the steps at which the termination is reached, reset the env
            for index, promise in zip(indices, promises):
                obs[index] = promise()
        obs = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
        action, agent_state, prior = agent(obs, done, agent_state)
        prior_list.append(prior)
        agent_state_list.append(agent_state[0])
        action = np.array(action)
        assert len(action) == len(envs)
        promises = [e.step(a, blocking=False) for e, a in zip(envs, action)]
        obs, _, done = zip(*[p()[:3] for p in promises])
        obs = list(obs)
        done = np.stack(done)
        episode += int(done.sum())
        length += 1
        step += (done * length).sum()
        length *= (1 - done)
    return (step - steps, episode - episodes, done, length, obs, agent_state, agent_state_list, prior_list)


def count_episodes(directory):
    filenames = directory.glob('*.npz') 
    lengths = [int(n.stem.rsplit('-', 1)[-1]) - 1 for n in filenames]
    episodes, steps = len(lengths), sum(lengths)
    return episodes, steps


def save_episodes(directory, episodes):
    directory = pathlib.Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    for episode in episodes:
        identifier = str(uuid.uuid4().hex) # generates random identifier
        length = len(episode['reward'])
        filename = directory / f'{timestamp}-{identifier}-{length}.npz'
        with io.BytesIO() as f1:
            np.savez_compressed(f1, **episode)
            f1.seek(0)
            with filename.open('wb') as f2:
                f2.write(f1.read())

def sortTimeKey(s):
    return datetime.datetime.strptime(str(os.path.basename(s))[:15], '%Y%m%dT%H%M%S')

def save_episodes_fifo(directory, episodes, replay_buffer_capacity):
    directory = pathlib.Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    filenames = list(directory.glob('*.npz'))
    num_files = len(filenames)
    if num_files > replay_buffer_capacity:
        filenames.sort(key=sortTimeKey)
        num_episodes = len(episodes)
        num_delete = num_files - (replay_buffer_capacity-num_episodes)
        if num_delete > 0:
            for i in range(num_delete):
                os.remove(str(filenames[i]))
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    for episode in episodes:
        identifier = str(uuid.uuid4().hex) # generates random identifier
        length = len(episode['reward'])
        filename = directory / f'{timestamp}-{identifier}-{length}.npz'
        with io.BytesIO() as f1:
            np.savez_compressed(f1, **episode)
            f1.seek(0)
            with filename.open('wb') as f2:
                f2.write(f1.read())


def load_episodes(directory, rescan, length=None, balance=False, seed=0):# , drop_rate=0.0):
    directory = pathlib.Path(directory).expanduser()
    random = np.random.RandomState(seed)
    cache = {}
    while True:
        for filename in directory.glob('*.npz'):
            if filename not in cache:
                try:
                    with filename.open('rb') as f:
                        episode = np.load(f)
                        episode = {k: episode[k] for k in episode.keys()}
                except Exception as e:
                    print(f'Could not load episode: {e}')
                    continue
                cache[filename] = episode
        
        keys = list(cache.keys())
        for index in random.choice(len(keys), rescan):
            episode = cache[keys[index]]
            if length:
                total = len(next(iter(episode.values())))
                available = total - length
                # available -= int(drop_rate * total)
                if available < 1:
                    print(f'Skipped short episode of length {available}')
                    continue
                if balance:
                    index = min(random.randint(0, total), available)
                else:
                    index = int(random.randint(0, available))
                episode = {k: v[index: index + length] for k, v in episode.items()}
            yield episode


class DummyEnv:
    def __init__(self):
        self._random = np.random.RandomState(seed=0)
        self._step = None

    @property
    def observation_space(self):
        low = np.zeros([64, 64, 3], dtype=np.uint8)
        high = 255 * np.ones([64, 64, 3], dtype=np.uint8)
        spaces = {'image': gym.spaces.Box(low, high)}
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        low = -np.ones([5], dtype=np.float32)
        high = np.ones([5], dtype=np.float32)
        return gym.spaces.Box(low, high)

    def reset(self):
        self._step = 0
        obs = self.observation_space.sample()
        return obs

    def step(self, action):
        obs = self.observation_space.sample()
        reward = self._random.uniform(0, 1)
        self._step += 1
        done = self._step >= 1000
        info = {}
        return obs, reward, done, info


class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample(self._samples)
        return tf.reduce_mean(samples, 0)
  
    def stddev(self):
        samples = self._dist.sample(self._samples)
        return tf.math.reduce_std(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        return tf.gather(sample, tf.argmax(logprob))[0]

    def entropy(self):
        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -tf.reduce_mean(logprob, 0)


class OneHotDist:
    def __init__(self, logits=None, probs=None):
        self._dist = tfd.Categorical(logits=logits, probs=probs) # categorical distribution over the actions
        self._num_classes = self.mean().shape[-1]
        self._dtype = prec.global_policy().compute_dtype

    @property
    def name(self):
        return 'OneHotDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def prob(self, events):
        indices = tf.argmax(events, axis=-1)
        return self._dist.prob(indices)

    def log_prob(self, events):
        indices = tf.argmax(events, axis=-1)
        return self._dist.log_prob(indices)

    def mean(self):
        return self._dist.probs_parameter()

    def mode(self):
        return self._one_hot(self._dist.mode())

    def sample(self, amount=None):
        amount = [amount] if amount else []
        indices = self._dist.sample(*amount)
        sample = self._one_hot(indices)
        probs = self._dist.probs_parameter()
        sample += tf.cast(probs - tf.stop_gradient(probs), self._dtype) # what is this line useful for?
        return sample

    def _one_hot(self, indices):
        return tf.one_hot(indices, self._num_classes, dtype=self._dtype)


class TanhBijector(tfp.bijectors.Bijector):

    def __init__(self, validate_args=False, name='tanh'):
        super().__init__(forward_min_event_ndims=0, validate_args=validate_args, name=name)

    def _forward(self, x):
        return tf.nn.tanh(x)

    def _inverse(self, y):
        dtype = y.dtype
        y = tf.cast(y, tf.float32)
        y = tf.where(tf.less_equal(tf.abs(y), 1.), tf.clip_by_value(y, -0.99999997, 0.99999997), y)
        y = tf.atanh(y)
        y = tf.cast(y, dtype)
        return y

    def _forward_log_det_jacobian(self, x):
        log2 = tf.math.log(tf.constant(2.0, dtype=x.dtype))
        return 2.0 * (log2 - x - tf.nn.softplus(-2.0 * x))


def lambda_return(reward, value, pcont, bootstrap, lambda_, axis):
    assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * tf.ones_like(reward) 
    dims = list(range(reward.shape.ndims))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:] 
    if axis != 0:
        reward = tf.transpose(reward, dims)
        value = tf.transpose(value, dims)
        pcont = tf.transpose(pcont, dims)
    if bootstrap is None:
        bootstrap = tf.zeros_like(value[-1])
    next_values = tf.concat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    returns = static_scan(
        lambda agg, cur: cur[0] + cur[1] * lambda_ * agg, (inputs, pcont), bootstrap, reverse=True) 
    if axis != 0:
        returns = tf.transpose(returns, dims)
    return returns


class Adam(tf.Module):
    def __init__(self, name, modules, lr, clip=None, wd=None, wdpattern=r'.*'):
        self._name = name
        self._modules = modules
        self._clip = clip
        self._wd = wd
        self._wdpattern = wdpattern
        self._opt = tf.optimizers.Adam(lr)
        self._opt = prec.LossScaleOptimizer(self._opt, 'dynamic') # An optimizer that applies loss scaling to prevent numeric underflow.
        self._variables = None

    @property
    def variables(self):
        return self._opt.variables()

    def __call__(self, tape, loss):
        if self._variables is None:
            variables = [module.variables for module in self._modules]
            self._variables = tf.nest.flatten(variables)
            count = sum(np.prod(x.shape) for x in self._variables)
            print(f'Found {count} {self._name} parameters.')
        assert len(loss.shape) == 0, loss.shape
        with tape:
            loss = self._opt.get_scaled_loss(loss)
        grads = tape.gradient(loss, self._variables)
        grads = self._opt.get_unscaled_gradients(grads)
        norm = tf.linalg.global_norm(grads)
        if self._clip:
            grads, _ = tf.clip_by_global_norm(grads, self._clip, norm)
        if self._wd:
            context = tf.distribute.get_replica_context()
            context.merge_call(self._apply_weight_decay)
        self._opt.apply_gradients(zip(grads, self._variables))
        return norm

    def _apply_weight_decay(self, strategy):
        print('Applied weight decay to variables:')
        for var in self._variables:
            if re.search(self._wdpattern, self._name + '/' + var.name):
                print('- ' + self._name + '/' + var.name)
                strategy.extended.update(var, lambda var: self._wd * var)


def args_type(default):
    if isinstance(default, bool):
        return lambda x: bool(['False', 'True'].index(x))
    if isinstance(default, int):
        return lambda x: float(x) if ('e' in x or '.' in x) else int(x)
    if isinstance(default, pathlib.Path):
        return lambda x: pathlib.Path(x).expanduser()
    return type(default)


def static_scan(fn, inputs, start, reverse=False):
    last = start
    outputs = [[] for _ in tf.nest.flatten(start)] # create an empty list for each element in the input argument start
    indices = range(len(tf.nest.flatten(inputs)[0]))
    if reverse:
        indices = reversed(indices)
    for index in indices:
        inp = tf.nest.map_structure(lambda x: x[index], inputs) # getting input
        last = fn(last, inp) # apply the fn to the input start argument and the computed input (i.e., inputs[i])
        [o.append(l) for o, l in zip(outputs, tf.nest.flatten(last))]
    if reverse:
        outputs = [list(reversed(x)) for x in outputs] # if reverse, reverse back the reversed logged values in the ouputs list
    outputs = [tf.stack(x, 0) for x in outputs] # stack the list of values in the outputs list along the 0-th axis
    return tf.nest.pack_sequence_as(start, outputs)


def _mnd_sample(self, sample_shape=(), seed=None, name='sample'):
    return tf.random.normal(
        tuple(sample_shape) + tuple(self.event_shape),
        self.mean(), self.stddev(), self.dtype, seed, name)


tfd.MultivariateNormalDiag.sample = _mnd_sample


def _cat_sample(self, sample_shape=(), seed=None, name='sample'):
    assert len(sample_shape) in (0, 1), sample_shape
    assert len(self.logits_parameter().shape) == 2
    indices = tf.random.categorical(
        self.logits_parameter(), sample_shape[0] if sample_shape else 1,
        self.dtype, seed, name)
    if not sample_shape:
        indices = indices[..., 0]
    return indices


tfd.Categorical.sample = _cat_sample


class Every:

    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
        if self._last is None:
        # self._last will only be come not None once it is called (not instantiated), so in the initial step, always return True
            self._last = step
            return True
        if step >= self._last + self._every:
        # self._last records the last step we execute something (e.g,, train every 5 steps)
        # if the current input step count is larger than self._last by a margin larger than self._every (frequency for executing, e.g., training)
        # incremet self._last by self._every, and return True (i.e., confirm executing, e.g., training)
            self._last += self._every
            return True
    # otherwise, return False
        return False


class Once:
  # Once object: initialise self._once as True, once called, it will set self._once to False, but return True, after which if you call the class, it will return
  # False, and remains that way
    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False


def k_steps_action(action, k, batch_length):
    out = []
    for i in range(batch_length - k + 1):
        x = action[i:(i+k), ...]
        out.append(x)
    return out



def compute_tSNE_3D(forward_state_list, backward_state_list, config):
    assert len(forward_state_list) == len(backward_state_list), f'forward length {len(forward_state_list)}, backward length{len(backward_state_list)}.'
    forward_stochs = tf.stack([f['stoch'] for f in forward_state_list], axis=1)
    backward_stochs = tf.stack([b['stoch'] for b in backward_state_list], axis=1)
    print(tf.shape(forward_stochs))
    print(tf.shape(backward_stochs))
    out = []
    plt.figure()
    forward_embed = TSNE(n_components=2).fit_transform(forward_stochs[0])
    backward_embed = TSNE(n_components=2).fit_transform(backward_stochs[0])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(tf.shape(forward_stochs)[1]):
        ax.scatter(i, forward_embed[i, 0], forward_embed[i, 1], c='blue')
        ax.scatter(i, backward_embed[i, 0], backward_embed[i, 1], c='green')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('y')

    plt.savefig(pathlib.Path(config.save_tsne_dir) / 'tsne.png')
    plt.savefig(pathlib.Path(config.save_tsne_dir) / 'tsne.pdf')
    return out

def compute_tSNE(forward_state_list, backward_state_list, config):
    assert len(forward_state_list) == len(backward_state_list), f'forward length {len(forward_state_list)}, backward length{len(backward_state_list)}.'
    forward_stochs = tf.stack([tf.concat([f['mean'], f['internal']], axis=-1) for f in forward_state_list], axis=0)
    backward_stochs = tf.stack([tf.concat([b['mean'], b['internal']], axis=-1) for b in backward_state_list], axis=0)
    out = []

    forward_embed = TSNE(n_components=2).fit_transform(forward_stochs[0])
    backward_embed = TSNE(n_components=2).fit_transform(backward_stochs[0])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(forward_embed[:, 0], forward_embed[:, 1], s=3, c='blue')
    ax.scatter(backward_embed[:, 0], backward_embed[:, 1], s=3, c='green')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    np.savetxt(pathlib.Path(config.save_tsne_dir)/'forward_mean_2.txt', forward_embed)
    np.savetxt(pathlib.Path(config.save_tsne_dir)/'backward_mean_2.txt', backward_embed)
    plt.savefig(pathlib.Path(config.save_tsne_dir) / 'tsne_2D_2.png')
    plt.savefig(pathlib.Path(config.save_tsne_dir) / 'tsne_2D_2.pdf')
    return out

def bisimulation_metric(state, pred_state, c=1, pcont=None, mask=None):
    # if pcont is None:
    #     pcont = 1.
    # pcont = tf.cast(pcont, dtype=state['stoch'].dtype)
    # if len(tf.shape(state['stoch'])) > len(tf.shape(pcont)):
    #     pcont_tile = tf.expand_dims(pcont, axis=-1)
    mask = tf.cast(mask, dtype=tf.float16)
    z1, z2 = state['stoch']*mask, pred_state['stoch']*mask
    rdist_1, rdist_2 = state['reward_dist'], pred_state['reward_dist']
    mean_1, mean_2 = state['dynamics_mean']*mask, pred_state['dynamics_mean']*mask
    std_1, std_2 = state['dynamics_std']*mask, pred_state['dynamics_std']*mask
    W2_dist = tf.reduce_mean(tf.reduce_sum(tf.math.square(mean_1-mean_2), axis=-1)) + tf.reduce_mean(tf.reduce_sum(tf.math.square(std_1-std_2), axis=-1))
    bisim_metric = tf.square(tf.reduce_mean(tf.reduce_sum(tf.abs(z1-z2), axis=-1)) - tf.reduce_mean(tfd.kl_divergence(rdist_1, rdist_2)*mask[:, :, 0]) - 0.99*W2_dist)
    return bisim_metric

def sliding_window(tensor, window_size):
    n, l = tensor.shape[:2]
    # here we always assume that the window size is odd
    mean_tensor = [[tf.reduce_mean(tensor[j, max(i-window_size//2, 0):min(i+window_size//2, l)]) for i in range(l)] for j in range(n)]
    return mean_tensor# [0:len(mean_tensor)][:-1] # TODO: check the dimension 


def pcont_truncation_mask(pcont_pred, window_size, pcont_threshold):
    mean_tensor = sliding_window(pcont_pred, window_size)
    return (mean_tensor > tf.cast(pcont_threshold, dtype=tf.float16)) * 1.
