import tensorflow as tf
from tensorflow_probability import distributions as tfd
import tensorflow.keras.mixed_precision.experimental as prec
import numpy as np

import os
import sys
import time
import pathlib
import collections
import argparse
import functools
import json

import models
import tools
import wrappers

tf.get_logger().setLevel('ERROR')
# tf.config.run_functions_eagerly(True) # comment out this line to enable earger execution for debugging 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class retracePlanner(tools.Module):
    def __init__(self, config, datadir, actspace, writer):
        self._c = config 
        self._actspace = actspace 
        self._actdim = actspace.n if hasattr(actspace, 'n') else actspace.shape[0] 
        self._writer = writer
        self._random = np.random.RandomState(config.seed)
        with tf.device('cpu:0'):
            self._step = tf.Variable(count_steps(datadir, config), dtype=tf.int64)
        self._should_pretrain = tools.Once() 
        self._should_train = tools.Every(config.train_every) 
        self._should_log = tools.Every(config.log_every) 
        self._last_log = None 
        self._last_time = time.time()
        self._metrics = collections.defaultdict(tf.metrics.Mean) 
        self._metrics['expl_amount'] 
        self._float = prec.global_policy().compute_dtype 
        self._strategy = tf.distribute.MirroredStrategy() 
        self._latent_prediction_steps = self._c.latent_prediction_steps
        self._adaptive_truncation_mask = tf.ones((config.batch_size, config.batch_length-1, 1))
        with self._strategy.scope():
            self._dataset = iter(self._strategy.experimental_distribute_dataset(load_dataset(datadir, self._c)))
            self._build_model()

    def __call__(self, obs, reset, state=None, training=True, current_step=None):
        step = self._step.numpy().item()
        tf.summary.experimental.set_step(step)
        if state is not None and reset.any():
            mask = tf.cast(1 - reset, self._float)[:, None]
            state = tf.nest.map_structure(lambda x: x * mask, state)
        if self._should_train(step):
            log = self._should_log(step)
            n = self._c.pretrain if self._should_pretrain() else self._c.train_steps
            print(f'Training for {n} steps.')
            with self._strategy.scope():
                for train_step in range(n):
                    log_images = self._c.log_images and log and train_step == 0
                    self.train(next(self._dataset), log_images, current_step)
            if log:
                self._write_summaries()
        action, state = self.policy(obs, state, training)
        if training:
            self._step.assign_add(len(reset) * self._c.action_repeat)
        return action, state

    @tf.function
    def policy(self, obs, state, training):
        if state is None:
            latent = self._dynamics.initial(len(obs['image']))
            action = tf.zeros((len(obs['image']), self._actdim), self._float) 
        else:
            latent, action = state
        embed = self._encode(preprocess(obs, self._c))
        latent, _ = self._dynamics.obs_step(latent, action, embed, num_latent_sample=1) 
        feat = self._dynamics.get_feature(latent) 
        if training:
            action = self._actor(feat).sample() 
        else:
            action = self._actor(feat).mode()
        action = self._exploration(action, training) 
        state = (latent, action)
        return action, state

    def load(self, filename):
        super().load(filename)
        self._should_pretrain()

    @tf.function()
    def train(self, data, log_images=False, current_step=None):
        self._strategy.run(self._train, args=(data, log_images, current_step))

    def _train(self, data, log_images, current_step=None):
        with tf.GradientTape() as model_tape:
            embed = self._encode(data)
            post, prior, last_prior = self._dynamics.observe(embed, data['action'], k=self._latent_prediction_steps)

            feat = self._dynamics.get_feature(post)
            image_pred = self._decode(feat)
            reward_pred = self._reward(feat)
            likes = tools.AttrDict()

            likes.image = tf.reduce_mean(image_pred.log_prob(data['image'][:, (self._latent_prediction_steps-1):, ...]))
            likes.reward = tf.reduce_mean(reward_pred.log_prob(data['reward'][:, (self._latent_prediction_steps-1):, ...]))
            if self._c.pcont:
                pcont_pred = self._pcont(feat) 
                pcont_target = self._c.discount * data['discount']
                likes.pcont = tf.reduce_mean(pcont_pred.log_prob(pcont_target[:, (self._latent_prediction_steps-1):]))
                likes.pcont *= self._c.pcont_scale
            if self._latent_prediction_steps == 1:
                prior_dist = self._dynamics.get_distribution(prior)
            else:
                prior_dist = self._dynamics.get_distribution(last_prior)
            post_dist = self._dynamics.get_distribution(post)
            div = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist))
            div = tf.maximum(div, self._c.free_nats) 
            model_loss = self._c.kl_scale * div - sum(likes.values())

            if self._c.cycle_consistency:
                if self._c.adaptive_truncation and current_step is not None and current_step > self._c.no_truncation_period:
                    values = tf.stop_gradient(self._value(self._dynamics.get_feature(post)))
                    values_bar = tf.stop_gradient(tf.constant([tf.reduce_mean(values[:, i:(i+self._c.adaptive_truncation_window_size)], axis=1) for i in range(self._c.batch_length-self._c.adaptive_truncation_window_size)]))
                    diff = tf.stop_gradient(tf.constant([tf.abs(tf.abs(values_bar[:, i+1]-values_bar[:, i])/values_bar[:, i])>self._c.adaptive_truncation_threshold for i in range(self._c.batch_length-self._c.adaptive_truncation_window_size-1)]))
                    truncation_mask = tf.ones((self.batch_size, self.batch_length, 1))
                    for i in range(1, self._c.batch_length-self._c.adaptive_truncation_window_size-1):
                        truncation_mask[:, i:(i+self._c.adaptive_truncation_window_size)] *= diff[:, i]
                    truncation_mask = truncation_mask[:, 1:, :]
                    truncation_percentage = tf.stop_gradient(tf.reduce_mean(tf.reduce_sum(diff, axis=1)/self._c.batch_length))
                else:
                    truncation_mask = self._adaptive_truncation_mask
                    truncation_percentage = 0.0
                
                retraced_priors = self._dynamics.retrace(reverse_actor=self._reverse_actor, priors=prior, posts=post, latent_prediction_steps=self._latent_prediction_steps)
                
                if self._c.retrace_loss == 'bisimulation':
                    original_features = self._dynamics.get_feature(post)[:, :-self._latent_prediction_steps, ...]
                    original_sampled_action = tf.stop_gradient(self._actor(original_features).sample())
                    original_dynamics = self._dynamics.imagine_step({k: v[:, :-self._latent_prediction_steps, :] for k, v in post.items()}, original_sampled_action)
                    original_state = {'stoch': post['mean'][:, :-self._latent_prediction_steps, :], 'reward_dist': self._reward(original_features), 'dynamics_mean': original_dynamics['mean'], 'dynamics_std': original_dynamics['std']}
                    retraced_features = self._dynamics.get_feature(retraced_priors)[:, (self._latent_prediction_steps-1):, ...]
                    retraced_sampled_action = tf.stop_gradient(self._actor(retraced_features).sample())
                    retraced_dynamics = self._dynamics.imagine_step({k: v[:, (self._latent_prediction_steps-1):, ...] for k, v in retraced_priors.items()}, retraced_sampled_action)
                    retraced_state = {'stoch': retraced_priors['mean'][:, (self._latent_prediction_steps-1):, ...], 'reward_dist': self._reward(retraced_features), 'dynamics_mean': retraced_dynamics['mean'], 'dynamics_std': retraced_dynamics['std']}
                    pcont_mask = None
                    if self._c.pcont:
                        pcont_sample = pcont_pred.sample()
                        pcont = tools.sliding_window(pcont_sample, self._c.truncation_window_size)
                        pcont_mask = tf.cast(tf.math.greater(tf.cast(pcont, tf.float16)[:, :-1], tf.cast(0.7, tf.float16)), tf.float16)
                    retrace_error = tools.bisimulation_metric(original_state, retraced_state, pcont=pcont_mask, mask=truncation_mask)
                    retrace_error = tf.maximum(retrace_error, self._c.retrace_minimum)
                    model_loss += self._c.retrace_scale * retrace_error
                elif self._c.retrace_loss == 'L2':
                    retraced_stoch = retraced_priors['stoch']
                    original_stoch = post['stoch'][:, 1:, :]
                    retrace_error = tf.math.sqrt(tf.reduce_sum(tf.math.square(retraced_stoch - original_stoch), axis=-1))
                    retrace_error = tf.reduce_mean(retrace_error)
                    model_loss += self._c.retrace_scale * retrace_error
                elif self._c.retrace_loss == 'reconstruction':
                    retraced_feat = self._dynamics.get_feature(retraced_priors)
                    retraced_image_pred = self._decode(retraced_feat)
                    retraced_decoding_ll = tf.reduce_mean(retraced_image_pred.log_prob(data['image'][:, 1:, ...]))
                    retrace_error = retraced_decoding_ll
                    model_loss += self._c.retrace_scale * retrace_error
                else:
                    raise NotImplementedError('Retrace loss function not recognised.')
            model_loss /= float(self._strategy.num_replicas_in_sync)

        with tf.GradientTape() as actor_tape:
            imag_feat = self._imagine_ahead(post)
            reward = self._reward(imag_feat).mode() 
            if self._c.pcont:
                pcont = self._pcont(imag_feat).mean() 
            else:
                pcont = self._c.discount * tf.ones_like(reward) 
            value = self._value(imag_feat).mode()
            returns = tools.lambda_return(
                reward[:-1], value[:-1], pcont[:-1],
                bootstrap=value[-1], lambda_=self._c.disclam, axis=0)
            discount = tf.stop_gradient(tf.math.cumprod(tf.concat(
                [tf.ones_like(pcont[:1]), pcont[:-2]], 0), 0))
            actor_loss = -tf.reduce_mean(discount * returns)
            actor_loss /= float(self._strategy.num_replicas_in_sync)

        with tf.GradientTape() as value_tape:
            value_pred = self._value(imag_feat)[:-1]
            target = tf.stop_gradient(returns) 
            value_loss = -tf.reduce_mean(discount * value_pred.log_prob(target))
            value_loss /= float(self._strategy.num_replicas_in_sync)

        model_norm = self._model_opt(model_tape, model_loss)
        actor_norm = self._actor_opt(actor_tape, actor_loss)
        value_norm = self._value_opt(value_tape, value_loss)
        
        if tf.distribute.get_replica_context().replica_id_in_sync_group == 0:
            if not self._c.cycle_consistency:
                retrace_error = 0
            if self._c.log_scalars:
                self._scalar_summaries(
                    data, feat, prior_dist, post_dist, likes, div,
                    model_loss, value_loss, actor_loss, model_norm, value_norm,
                    actor_norm, retrace_error, truncation_percentage)
            if tf.equal(log_images, True):
                self._image_summaries(data, embed, image_pred)

    def _build_model(self):
        acts = dict(
            elu=tf.nn.elu, relu=tf.nn.relu, swish=tf.nn.swish,
            leaky_relu=tf.nn.leaky_relu)
        cnn_act = acts[self._c.cnn_act]
        act = acts[self._c.dense_act]
        self._encode = models.ConvEncoder(self._c.cnn_depth, cnn_act) 
        self._dynamics = models.latentRSSM(self._c.stoch_size, self._c.deter_size, self._c.deter_size, num_latent_sample=self._c.num_latent_sample)
        self._decode = models.ConvDecoder(self._c.cnn_depth, cnn_act)
        self._reward = models.DenseDecoder((), 2, self._c.num_units, act=act)
        if self._c.cycle_consistency:
            self._reverse_actor = models.reverseAction(4, self._c.num_units, self._c.joint_dim, self._actdim, dist=self._c.reverse_dist, act=act)
        if self._c.pcont:
            self._pcont = models.DenseDecoder((), 3, self._c.num_units, 'binary', act=act) 
        self._value = models.DenseDecoder((), 3, self._c.num_units, act=act)
        self._actor = models.ActionDecoder(self._actdim, 4, self._c.num_units, self._c.action_dist, init_std=self._c.action_init_std, act=act)
        model_modules = [self._encode, self._dynamics, self._decode, self._reward]
        if self._c.pcont:
            model_modules.append(self._pcont)
        if self._c.cycle_consistency:
            model_modules.append(self._reverse_actor)
        Optimizer = functools.partial(
            tools.Adam, wd=self._c.weight_decay, clip=self._c.grad_clip,
            wdpattern=self._c.weight_decay_pattern)
        self._model_opt = Optimizer('model', model_modules, self._c.model_lr)
        self._value_opt = Optimizer('value', [self._value], self._c.value_lr)
        self._actor_opt = Optimizer('actor', [self._actor], self._c.actor_lr)
        self.train(next(self._dataset))

    def _exploration(self, action, training):
        if training:
            amount = self._c.expl_amount 
            if self._c.expl_decay:
                amount *= 0.5 ** (tf.cast(self._step, tf.float32) / self._c.expl_decay)
            if self._c.expl_min:
                amount = tf.maximum(self._c.expl_min, amount) 
            self._metrics['expl_amount'].update_state(amount)
        elif self._c.eval_noise:
            amount = self._c.eval_noise
        else:
            return action
        if self._c.expl == 'additive_gaussian':
            return tf.clip_by_value(tfd.Normal(action, amount).sample(), -1, 1)
        if self._c.expl == 'completely_random':
            return tf.random.uniform(action.shape, -1, 1)
        if self._c.expl == 'epsilon_greedy':
            indices = tfd.Categorical(0 * action).sample()
            return tf.where(
                tf.random.uniform(action.shape[:1], 0, 1) < amount,
                tf.one_hot(indices, action.shape[-1], dtype=self._float),
                action)
        raise NotImplementedError(self._c.expl)

    def _imagine_ahead(self, post):
        if self._c.pcont:  
            post = {k: v[:, :-1] for k, v in post.items()}
        flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in post.items()}
        policy = lambda state: self._actor(tf.stop_gradient(self._dynamics.get_feature(state))).sample() 
        states = tools.static_scan(
            lambda prev, _: self._dynamics.imagine_step(prev, policy(prev), num_latent_sample=1),
            tf.range(self._c.horizon), start)
        imag_feat = self._dynamics.get_feature(states)
        return imag_feat

    def _scalar_summaries(
        self, data, feat, prior_dist, post_dist, likes, div,
        model_loss, value_loss, actor_loss, model_norm, value_norm,
        actor_norm, retrace_loss, truncation_percentage):
        self._metrics['model_grad_norm'].update_state(model_norm)
        self._metrics['value_grad_norm'].update_state(value_norm)
        self._metrics['actor_grad_norm'].update_state(actor_norm)
        self._metrics['prior_ent'].update_state(prior_dist.entropy()) 
        self._metrics['post_ent'].update_state(post_dist.entropy())
        for name, logprob in likes.items():
            self._metrics[name + '_loss'].update_state(-logprob)
        self._metrics['div'].update_state(div)
        self._metrics['model_loss'].update_state(model_loss)
        self._metrics['value_loss'].update_state(value_loss)
        self._metrics['actor_loss'].update_state(actor_loss)
        self._metrics['retrace_loss'].update_state(retrace_loss)
        self._metrics['action_ent'].update_state(self._actor(feat).entropy())
        self._metrics['truncation_percentage'].update_state(truncation_percentage)

    def _image_summaries(self, data, embed, image_pred):
        truth = data['image'][:6] + 0.5
        recon = image_pred.mode()[:6]
        _, _, init = self._dynamics.observe(embed[:6, :5], data['action'][:6, :5], num_latent_sample=1)
        init = {k: v[:, -1] for k, v in init.items()}
        prior = self._dynamics.imagine(data['action'][:6, 5:], init, num_latent_sample=1)
        openl = self._decode(self._dynamics.get_feature(prior)).mode()
        model = tf.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        openl = tf.concat([truth, model, error], 2)
        tools.graph_summary(self._writer, tools.video_summary, 'agent/openl', openl)

    def _write_summaries(self):
        step = int(self._step.numpy())
        metrics = [(k, float(v.result())) for k, v in self._metrics.items()]
        if self._last_log is not None:
            duration = time.time() - self._last_time
            self._last_time += duration
            metrics.append(('fps', (step - self._last_log) / duration))
        self._last_log = step
        [m.reset_states() for m in self._metrics.values()]
        with (self._c.logdir / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps({'step': step, **dict(metrics)}) + '\n')
        [tf.summary.scalar('agent/' + k, m) for k, m in metrics]
        print(f'[{step}]', ' / '.join(f'{k} {v:.1f}' for k, v in metrics))
        self._writer.flush()


def preprocess(obs, config):
    dtype = prec.global_policy().compute_dtype
    obs = obs.copy()
    with tf.device('cpu:0'):
        obs['image'] = tf.cast(obs['image'], dtype)/255.0 - 0.5
        clip_rewards = dict(none=lambda x: x, tanh=tf.tanh)[config.clip_rewards]
        obs['reward'] = clip_rewards(obs['reward'])
    return obs

def count_steps(datadir, config):
    return tools.count_episodes(datadir)[1] * config.action_repeat

def load_dataset(directory, config):
    episode = next(tools.load_episodes(directory, 1))
    types = {k: v.dtype for k, v in episode.items()}
    shapes = {k: (None, ) + v.shape[1:] for k, v in episode.items()}
    generator = lambda: tools.load_episodes(directory, config.train_steps, config.batch_length, config.dataset_balance)
    dataset = tf.data.Dataset.from_generator(generator, types, shapes)
    dataset = dataset.batch(config.batch_size, drop_remainder=True)
    dataset = dataset.map(functools.partial(preprocess, config=config))
    dataset = dataset.prefetch(10)
    return dataset

def summarize_episode(episode, config, datadir, writer, prefix):
    n_episodes, steps = tools.count_episodes(datadir)
    length = (len(episode['reward']) - 1) * config.action_repeat
    ret = episode['reward'].sum()
    print(f'{prefix.title()} episode of length {length} with return {ret:.1f}.')
    metrics = [
        (f'{prefix}/return', float(episode['reward'].sum())), 
        (f'{prefix}/length', len(episode['reward'])-1), 
        (f'episodes', n_episodes)
    ]
    step = count_steps(datadir, config)
    with (config.logdir/'metrics.jsonl').open('a') as f:
        f.write(json.dumps(dict([('step', step)] + metrics)) + '\n')
    with writer.as_default():
        tf.summary.experimental.set_step(step)
        [tf.summary.scalar('sim/' + k, v) for k, v in metrics]
        if prefix == 'test':
            tools.video_summary(f'sim/{prefix}/video', episode['image'][None])

def make_env(config, writer, prefix, datadir, store):
    suite, task = config.task.split('_', 1)
    if suite == 'dmc':
        env = wrappers.DeepMindControl(task)
        env = wrappers.ActionRepeat(env, config.action_repeat)
        env = wrappers.NormalizeActions(env)
    elif suite == 'atari':
        env = wrappers.Atari(task, config.action_repeat, (64, 64), grayscale=False, 
                            life_done=True, sticky_actions=True)
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit/config.action_repeat)
    callbacks = []
    if store:
        callbacks.append(lambda ep: tools.save_episodes(datadir, [ep]))
    callbacks.append(lambda ep: summarize_episode(ep, config, datadir, writer, prefix))
    env = wrappers.Collect(env, callbacks, config.precision)
    env = wrappers.RewardObs(env)
    env = wrappers.TruncationDone(env)
    return env

def define_config():
    config = tools.AttrDict()
    # General.
    config.logdir = pathlib.Path('./logdir')
    config.seed = 0
    config.steps = 5e6
    config.eval_every = 1e4
    config.log_every = 1e3
    config.log_scalars = True
    config.log_images = True
    config.gpu_growth = True
    config.precision = 16
    # Environment.
    config.task = 'dmc_walker_walk'
    config.envs = 1
    config.parallel = 'none'
    config.action_repeat = 2
    config.time_limit = 1000
    config.prefill = 5000
    config.eval_noise = 0.0
    config.clip_rewards = 'none'
    # Model.
    config.deter_size = 256
    config.stoch_size = 32
    config.num_units = 512
    config.joint_dim = 256
    config.reverse_dist = None
    config.dense_act = 'elu'
    config.cnn_act = 'relu'
    config.cnn_depth = 32
    config.pcont = False
    config.free_nats = 3.0
    config.kl_scale = 1.0
    config.pcont_scale = 10.0
    config.weight_decay = 0.0
    config.weight_decay_pattern = r'.*'
    config.cycle_consistency = True
    config.latent_prediction_steps = 1
    config.retrace_scale = 1.0 
    config.retrace_minimum = 0.0
    config.retrace_loss = 'bisimulation'
    config.early_truncation_ratio = 0.0
    config.pcont_threshold = 0.7
    config.truncation_window_size = 5
    config.num_latent_sample = 1
    config.truncation_annealing_period = 0.0
    config.truncation_max_episode = 1000
    # general
    config.batch_size = 50
    config.batch_length = 50
    config.train_every = 1000
    config.train_steps = 100
    config.pretrain = 100
    config.model_lr = 6e-4
    config.value_lr = 8e-5
    config.actor_lr = 8e-5
    config.grad_clip = 100.0
    config.dataset_balance = False 
    # Behavior.
    config.discount = 0.99
    config.disclam = 0.95 
    config.horizon = 15
    config.action_dist = 'tanh_normal'
    config.action_init_std = 5.0
    config.expl = 'additive_gaussian'
    config.expl_amount = 0.3
    config.expl_decay = 0.0
    config.expl_min = 0.0
    # truncation
    config.adaptive_truncation = False
    config.adaptive_truncation_window_size = 10
    config.no_truncation_period = 50000
    config.adaptive_truncation_threshold = 0.5
    return config


def main(config):
    if config.gpu_growth:
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        prec.set_policy(prec.Policy('mixed_float16'))
    config.steps = int(config.steps)
    # config.logdir.mkdir(parents=True, exist_ok=True)
    print('Logdir', config.logdir)

    datadir = config.logdir / 'episodes'
    writer = tf.summary.create_file_writer(str(config.logdir), max_queue=1000, flush_millis=20000)
    writer.set_as_default()
    train_envs = [wrappers.Async(lambda: make_env(config, writer, 'train', datadir, store=True), config.parallel) for _ in range(config.envs)]
    test_envs =[wrappers.Async(lambda: make_env(config, writer, 'test', datadir, store=False), config.parallel) for _ in range(config.envs)]
    actspace = train_envs[0].action_space
    
    step = count_steps(datadir, config)
    prefill = max(0, config.prefill - step)
    print(f'Prefill dataset with {prefill} steps. ')
    random_agent = lambda o, d, _, current_step: ([actspace.sample() for _ in d], None)
    tools.simulate(random_agent, train_envs, prefill/config.action_repeat, truncation_annealing_period=None, done_truncation_init=0.0)
    writer.flush()

    step = count_steps(datadir, config)
    print(f'Simulating agent for {config.steps-step} steps.')
    agent = retracePlanner(config, datadir, actspace, writer)
    if (config.logdir/'variables.pkl').exists():
        print('Load checkpoint.')
        agent.load(config.logdir / 'variables.pkl')
    state = None
    if config.truncation_annealing_period == 0.0:
        truncation_annealing_period = None
    else:
        truncation_annealing_period = config.truncation_annealing_period
    while step < config.steps:
        print('Start evaluation.')
        tools.simulate(functools.partial(agent, training=False), test_envs, episodes=5, truncation_annealing_period=None, done_truncation_init=0.0)
        writer.flush()
        print('Start collection.')
        steps = config.eval_every // config.action_repeat
        state = tools.simulate(agent, train_envs, steps, state=state, truncation_annealing_period=truncation_annealing_period, 
                               done_truncation_init=config.early_truncation_ratio, truncation_max_episode=config.truncation_max_episode)
        step = count_steps(datadir, config)
        agent.save(config.logdir / 'variables.pkl')
    for env in train_envs + test_envs:
        env.close()

if __name__ == '__main__':
    try:
        import colored_traceback
        colored_traceback.add_hook()
    except ImportError:
        pass
    parser = argparse.ArgumentParser()
    for key, value in define_config().items():
        parser.add_argument('--{}'.format(key), type=tools.args_type(value), default=value)
    main(parser.parse_args())
