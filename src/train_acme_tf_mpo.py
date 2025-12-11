from typing import (
    List,
    Optional,
    Dict,
    Tuple,
    Union,
    Sequence,
    Callable,
    Any,
    NamedTuple,
    Mapping,
)
import time
import collections
import threading
import argparse
import numpy as np
import sonnet as snt  # type: ignore
import wandb
import tree  # type: ignore
import operator
import os


import dm_env  # type: ignore
from dm_env import specs  # type: ignore
from dm_control import suite  # type: ignore


import tensorflow as tf  # type: ignore
import tensorflow_probability as tfp  # type: ignore
import trfl  # type: ignore


_MPO_FLOAT_EPSILON = 1e-8


def add_batch_dim(nest):
    """Adds a batch dimension to each leaf of a nested structure of Tensors."""
    return tree.map_structure(lambda x: tf.expand_dims(x, axis=0), nest)


def squeeze_batch_dim(nest):
    """Squeezes out a batch dimension from each leaf of a nested structure."""
    return tree.map_structure(lambda x: tf.squeeze(x, axis=0), nest)


def batch_concat(inputs) -> tf.Tensor:
    """Concatenate a collection of Tensors while preserving the batch dimension.

    This takes a potentially nested collection of tensors, flattens everything
    but the batch (first) dimension, and concatenates along the resulting data
    (second) dimension.

    Args:
      inputs: a tensor or nested collection of tensors.

    Returns:
      A concatenated tensor which maintains the batch dimension but concatenates
      all other data along the flattened second dimension.
    """
    flat_leaves = tree.map_structure(snt.Flatten(), inputs)
    return tf.concat(tree.flatten(flat_leaves), axis=-1)


def tile_tensor(tensor: tf.Tensor, multiple: int) -> tf.Tensor:
    """Tiles `multiple` copies of `tensor` along a new leading axis."""
    rank = len(tensor.shape)
    multiples = tf.constant([multiple] + [1] * rank, dtype=tf.int32)
    expanded_tensor = tf.expand_dims(tensor, axis=0)
    return tf.tile(expanded_tensor, multiples)


def create_variables(
    network: snt.Module,
    input_spec: List[Any],
) -> Optional[tf.TensorSpec]:
    """Builds the network with dummy inputs to create the necessary variables.

    Args:
      network: Sonnet Module whose variables are to be created.
      input_spec: list of input specs to the network. The length of this list
        should match the number of arguments expected by `network`.

    Returns:
      output_spec: only returns an output spec if the output is a tf.Tensor, else
          it doesn't return anything (None); e.g. if the output is a
          tfp.distributions.Distribution.
    """
    # Create a dummy observation with no batch dimension.
    dummy_input = zeros_like(input_spec)

    # If we have an RNNCore the hidden state will be an additional input.
    if isinstance(network, snt.RNNCore):
        initial_state = squeeze_batch_dim(network.initial_state(1))
        dummy_input += [initial_state]

    # Forward pass of the network which will create variables as a side effect.
    dummy_output = network(*add_batch_dim(dummy_input))

    # Evaluate the input signature by converting the dummy input into a
    # TensorSpec. We then save the signature as a property of the network. This is
    # done so that we can later use it when creating snapshots. We do this here
    # because the snapshot code may not have access to the precise form of the
    # inputs.
    input_signature = tree.map_structure(
        lambda t: tf.TensorSpec((None,) + t.shape, t.dtype), dummy_input
    )
    network._input_signature = input_signature  # pylint: disable=protected-access

    def spec(output):
        # If the output is not a Tensor, return None as spec is ill-defined.
        if not isinstance(output, tf.Tensor):
            return None
        # If this is not a scalar Tensor, make sure to squeeze out the batch dim.
        if tf.rank(output) > 0:
            output = squeeze_batch_dim(output)
        return tf.TensorSpec(output.shape, output.dtype)

    return tree.map_structure(spec, dummy_output)


class TransformationWrapper(snt.Module):
    """Helper class for to_sonnet_module.

    This wraps arbitrary Tensor-valued callables as a Sonnet module.
    A use case for this is in agent code that could take either a trainable
    sonnet module or a hard-coded function as its policy. By wrapping a hard-coded
    policy with this class, the agent can then treat it as if it were a Sonnet
    module. This removes the need for "if is_hard_coded:..." branches, which you'd
    otherwise need if e.g. calling get_variables() on the policy.
    """

    def __init__(self, transformation, name: Optional[str] = None):
        super().__init__(name=name)
        self._transformation = transformation

    def __call__(self, *args, **kwargs):
        return self._transformation(*args, **kwargs)


def to_sonnet_module(transformation) -> snt.Module:
    """Convert a tensor transformation to a Sonnet Module.

    Args:
      transformation: A Callable that takes one or more (nested) Tensors, and
        returns one or more (nested) Tensors.

    Returns:
      A Sonnet Module that wraps the transformation.
    """

    if isinstance(transformation, snt.Module):
        return transformation

    module = TransformationWrapper(transformation)

    # Wrap the module to allow it to return an empty variable tuple.
    return snt.allow_empty_variables(module)


def to_numpy(nest):
    """Converts a nest of Tensors to a nest of numpy arrays."""
    return tree.map_structure(lambda x: x.numpy(), nest)


def to_numpy_squeeze(nest, axis=0):
    """Converts a nest of Tensors to a nest of numpy arrays and squeeze axis."""
    return tree.map_structure(lambda x: tf.squeeze(x, axis=axis).numpy(), nest)


def zeros_like(nest):
    """Given a nest of array-like objects, returns similarly nested tf.zeros."""
    return tree.map_structure(lambda x: tf.zeros(x.shape, x.dtype), nest)


class EnvironmentSpec(NamedTuple):
    """Full specification of the domains used by a given environment."""

    # TODO(b/144758674): Use NestedSpec type here.
    observations: Any
    actions: Any
    rewards: Any
    discounts: Any


# ---------------------------------------------------------------------------
# In-memory replay buffer + N-step adder (replaces Reverb for single-actor setup)
# ---------------------------------------------------------------------------


class Transition(NamedTuple):
    observation: Any
    action: Any
    reward: Any
    discount: Any
    next_observation: Any
    extras: Any = ()


class SimpleReplayBuffer:
    """A minimal in-memory replay buffer supporting uniform sampling."""

    def __init__(self, max_size: int = 1000000, seed: Optional[int] = None):
        self._max_size = int(max_size)
        self._buffer = collections.deque(maxlen=self._max_size)
        self._rng = np.random.RandomState(seed)

    def add(self, transition: Transition):
        self._buffer.append(transition)

    def size(self) -> int:
        return len(self._buffer)

    def sample(self, batch_size: int) -> Transition:
        """Return a batch of transitions as numpy arrays stacked along axis 0."""
        if len(self._buffer) == 0:
            raise RuntimeError("Replay buffer is empty.")
        # allow sampling with replacement if not enough items
        idx = self._rng.randint(0, len(self._buffer), size=batch_size)
        sampled = [self._buffer[i] for i in idx]
        # stack fields (support nested structures)
        obs = tree.map_structure(
            lambda *xs: np.stack(xs, axis=0), *[t.observation for t in sampled]
        )
        act = tree.map_structure(
            lambda *xs: np.stack(xs, axis=0), *[t.action for t in sampled]
        )
        rew = tree.map_structure(
            lambda *xs: np.stack(xs, axis=0), *[t.reward for t in sampled]
        )
        disc = tree.map_structure(
            lambda *xs: np.stack(xs, axis=0), *[t.discount for t in sampled]
        )
        next_obs = tree.map_structure(
            lambda *xs: np.stack(xs, axis=0), *[t.next_observation for t in sampled]
        )
        extras = (
            tree.map_structure(
                lambda *xs: np.stack(xs, axis=0), *[t.extras for t in sampled]
            )
            if sampled[0].extras
            else ()
        )
        return Transition(
            observation=obs,
            action=act,
            reward=rew,
            discount=disc,
            next_observation=next_obs,
            extras=extras,
        )

    def all(self):
        return list(self._buffer)


class NStepTransitionAdder:
    """A tiny N-step adder for a single actor writing into the in-memory buffer."""

    def __init__(self, replay: SimpleReplayBuffer, n_step: int, discount: float):
        self._replay = replay
        self.n_step = max(1, int(n_step))
        self._discount = float(discount)
        # buffers
        self._observations = []
        self._actions = []
        self._rewards = []
        self._discounts = []
        self._extras = []
        self._first_idx = 0
        self._add_first_called = False

    def add_first(self, timestep):
        # timestep.observation is the initial observation
        self._observations = [np.asarray(timestep.observation)]
        self._actions = []
        self._rewards = []
        self._discounts = []
        self._extras = []
        self._first_idx = 0
        self._add_first_called = True

    def add(self, action, next_timestep, extras=()):
        if not self._add_first_called:
            raise ValueError("add_first must be called before add()")
        action = np.asarray(action)
        reward = np.asarray(next_timestep.reward)
        discount = np.asarray(next_timestep.discount)
        next_obs = np.asarray(next_timestep.observation)
        self._actions.append(action)
        self._rewards.append(reward)
        self._discounts.append(discount)
        self._extras.append(extras)
        # append next observation
        self._observations.append(next_obs)

        # After each add, potentially write one transition (the earliest outstanding).
        last_idx = len(self._actions)
        # Effective available length for the earliest index
        while self._first_idx < last_idx and (
            last_idx - self._first_idx >= self.n_step or next_timestep.last()
        ):
            # compute n-step return/window length
            window_len = min(self.n_step, last_idx - self._first_idx)
            # compute n-step return and total discount
            g = self._discount
            total_discount = 1.0
            R = 0.0
            prod_env = 1.0
            for k in range(window_len):
                r = self._rewards[self._first_idx + k]
                # product of environment discounts up to k-1
                R += (g**k) * prod_env * r
                # update prod_env by environment discount at this step
                prod_env *= float(self._discounts[self._first_idx + k])
            if window_len > 0:
                total_discount = (g ** (window_len - 1)) * prod_env
            else:
                total_discount = 1.0

            # build transition for start index
            s = self._observations[self._first_idx]
            a = self._actions[self._first_idx]
            next_s = self._observations[self._first_idx + window_len]
            extras = self._extras[self._first_idx] if self._extras else ()
            transition = Transition(
                observation=s,
                action=a,
                reward=R,
                discount=total_discount,
                next_observation=next_s,
                extras=extras,
            )
            self._replay.add(transition)
            self._first_idx += 1

        # If episode ended, reset internal buffers for next episode.
        if next_timestep.last():
            # Leftover transitions, already handled by the while loop above.
            self._observations = []
            self._actions = []
            self._rewards = []
            self._discounts = []
            self._extras = []
            self._first_idx = 0
            self._add_first_called = False


# ---------------------------------------------------------------------------
# Dataset factory that uses the in-memory replay buffer.
# ---------------------------------------------------------------------------


def make_in_memory_dataset(replay: SimpleReplayBuffer):
    """Create a tf.data.Dataset that samples single transitions from the replay.
    The dataset yields elements identical to the previous `TrajectoryDataset`'s
    `.data` object (i.e., a Transition-like mapping). The caller typically will
    then .batch(...) as before.
    """

    def generator():
        # Infinite generator that waits until the buffer has at least one item.
        while True:
            # Busy-wait until there is at least one transition.
            while replay.size() == 0:
                time.sleep(0.01)
            # sample a single transition (batch size 1)
            t = replay.sample(1)
            # For consistency with previous code, yield a mapping (not batched)
            # Remove the leading batch axis (size 1) so batching later matches old pipeline.
            # All fields are numpy arrays with leading dim = batch_size.
            # We squeeze the leading dimension so that dataset.batch will produce the intended batch.
            yield tree.map_structure(lambda x: np.squeeze(x, axis=0), t)

    # The output signature is constructed lazily: sample one item to infer structure.
    # Wait until buffer has at least one item (actors will populate it).
    while replay.size() == 0:
        time.sleep(0.01)
    sample_item = replay.sample(1)
    sample_item_no_batch = tree.map_structure(
        lambda x: np.squeeze(x, axis=0), sample_item
    )

    def tf_signature_from_numpy(numpy_obj):
        if isinstance(numpy_obj, np.ndarray):
            return tf.TensorSpec(shape=numpy_obj.shape, dtype=numpy_obj.dtype)
        else:
            # fallback for nested structures
            return tf.TensorSpec(shape=(), dtype=type(numpy_obj))

    # Create nested signature
    output_signature = tree.map_structure(tf_signature_from_numpy, sample_item_no_batch)

    # Return a dataset of transitions (not yet batched).
    return tf.data.Dataset.from_generator(generator, output_signature=output_signature)


# ...existing code...


# ---------------------------------------------------------------------------
# Modify MPOLearner and MPO wiring to use the in-memory replay/dataset.
# ---------------------------------------------------------------------------


class MPOLearner:

    def __init__(
        self,
        policy_network: snt.Module,
        critic_network: snt.Module,
        target_policy_network: snt.Module,
        target_critic_network: snt.Module,
        discount: float,
        num_samples: int,
        target_policy_update_period: int,
        target_critic_update_period: int,
        policy_loss_module: snt.Module,
        dataset,  # dataset is now a tf.data.Dataset that yields transitions directly
        observation_network=tf.identity,
        target_observation_network=tf.identity,
        policy_optimizer: Optional[snt.Optimizer] = None,
        critic_optimizer: Optional[snt.Optimizer] = None,
        dual_optimizer: Optional[snt.Optimizer] = None,
        clipping: bool = True,
        log_every: int = 1,
    ):
        self._discount = discount
        self._num_samples = num_samples
        self._clipping = clipping

        # Necessary to track when to update target networks.
        self._num_steps = tf.Variable(0, dtype=tf.int32)
        self._target_policy_update_period = target_policy_update_period
        self._target_critic_update_period = target_critic_update_period
        self._log_every = log_every

        # Batch dataset and create iterator.
        # TODO(b/155086959): Fix type stubs and remove.
        self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types

        # Store online and target networks.
        self._policy_network = policy_network
        self._critic_network = critic_network
        self._target_policy_network = target_policy_network
        self._target_critic_network = target_critic_network

        # Make sure observation networks are snt.Module's so they have variables.
        self._observation_network = to_sonnet_module(observation_network)
        self._target_observation_network = to_sonnet_module(target_observation_network)

        self._policy_loss_module = policy_loss_module

        # Create the optimizers.
        self._critic_optimizer = critic_optimizer or snt.optimizers.Adam(1e-4)
        self._policy_optimizer = policy_optimizer or snt.optimizers.Adam(1e-4)
        self._dual_optimizer = dual_optimizer or snt.optimizers.Adam(1e-2)

        # Expose the variables.
        policy_network_to_expose = snt.Sequential(
            [self._target_observation_network, self._target_policy_network]
        )
        self._variables = {
            "critic": self._target_critic_network.variables,
            "policy": policy_network_to_expose.variables,
        }

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp = None
        self._total_steps = 0

    def run(self, num_steps: int):
        iterator = range(num_steps)
        for _ in iterator:
            self.step()

    @tf.function
    def _step(self):
        # Update target network.
        online_policy_variables = self._policy_network.variables
        target_policy_variables = self._target_policy_network.variables
        online_critic_variables = (
            *self._observation_network.variables,
            *self._critic_network.variables,
        )
        target_critic_variables = (
            *self._target_observation_network.variables,
            *self._target_critic_network.variables,
        )

        # Make online policy -> target policy network update ops.
        if tf.math.mod(self._num_steps, self._target_policy_update_period) == 0:
            for src, dest in zip(online_policy_variables, target_policy_variables):
                dest.assign(src)
        # Make online critic -> target critic network update ops.
        if tf.math.mod(self._num_steps, self._target_critic_update_period) == 0:
            for src, dest in zip(online_critic_variables, target_critic_variables):
                dest.assign(src)

        # Increment number of learner steps for periodic update bookkeeping.
        self._num_steps.assign_add(1)

        # Get next batch of data.
        inputs = next(self._iterator)

        # Get data from replay (dropping extras if any). Note there is no
        # extra data here because we do not insert any into Reverb.
        transitions = inputs  # dataset yields transitions directly

        # Cast the additional discount to match the environment discount dtype.
        discount = tf.cast(self._discount, dtype=transitions.discount.dtype)

        with tf.GradientTape(persistent=True) as tape:
            # Maybe transform the observation before feeding into policy and critic.
            # Transforming the observations this way at the start of the learning
            # step effectively means that the policy and critic share observation
            # network weights.
            o_tm1 = self._observation_network(transitions.observation)
            # This stop_gradient prevents gradients to propagate into the target
            # observation network. In addition, since the online policy network is
            # evaluated at o_t, this also means the policy loss does not influence
            # the observation network training.
            o_t = tf.stop_gradient(
                self._target_observation_network(transitions.next_observation)
            )

            # Get action distributions from policy networks.
            online_action_distribution = self._policy_network(o_t)
            target_action_distribution = self._target_policy_network(o_t)

            # Get sampled actions to evaluate policy; of size [N, B, ...].
            sampled_actions = target_action_distribution.sample(self._num_samples)
            tiled_o_t = tile_tensor(o_t, self._num_samples)  # [N, B, ...]

            # Compute the target critic's Q-value of the sampled actions in state o_t.
            sampled_q_t = self._target_critic_network(
                # Merge batch dimensions; to shape [N*B, ...].
                snt.merge_leading_dims(tiled_o_t, num_dims=2),
                snt.merge_leading_dims(sampled_actions, num_dims=2),
            )

            # Reshape Q-value samples back to original batch dimensions and average
            # them to compute the TD-learning bootstrap target.
            sampled_q_t = tf.reshape(sampled_q_t, (self._num_samples, -1))  # [N, B]
            q_t = tf.reduce_mean(sampled_q_t, axis=0)  # [B]

            # Compute online critic value of a_tm1 in state o_tm1.
            q_tm1 = self._critic_network(o_tm1, transitions.action)  # [B, 1]
            q_tm1 = tf.squeeze(q_tm1, axis=-1)  # [B]; necessary for trfl.td_learning.

            # Critic loss.
            critic_loss = trfl.td_learning(
                q_tm1, transitions.reward, discount * transitions.discount, q_t
            ).loss
            critic_loss = tf.reduce_mean(critic_loss)

            # Actor learning.
            policy_loss, policy_stats = self._policy_loss_module(
                online_action_distribution=online_action_distribution,
                target_action_distribution=target_action_distribution,
                actions=sampled_actions,
                q_values=sampled_q_t,
            )

        # For clarity, explicitly define which variables are trained by which loss.
        critic_trainable_variables = (
            # In this agent, the critic loss trains the observation network.
            self._observation_network.trainable_variables
            + self._critic_network.trainable_variables
        )
        policy_trainable_variables = self._policy_network.trainable_variables
        # The following are the MPO dual variables, stored in the loss module.
        dual_trainable_variables = self._policy_loss_module.trainable_variables

        # Compute gradients.
        critic_gradients = tape.gradient(critic_loss, critic_trainable_variables)
        policy_gradients, dual_gradients = tape.gradient(
            policy_loss, (policy_trainable_variables, dual_trainable_variables)
        )

        # Delete the tape manually because of the persistent=True flag.
        del tape

        # Maybe clip gradients.
        if self._clipping:
            policy_gradients = tuple(tf.clip_by_global_norm(policy_gradients, 40.0)[0])
            critic_gradients = tuple(tf.clip_by_global_norm(critic_gradients, 40.0)[0])

        # Apply gradients.
        self._critic_optimizer.apply(critic_gradients, critic_trainable_variables)
        self._policy_optimizer.apply(policy_gradients, policy_trainable_variables)
        self._dual_optimizer.apply(dual_gradients, dual_trainable_variables)

        # Losses to track.
        fetches = {
            "critic_loss": critic_loss,
            "policy_loss": policy_loss,
        }
        fetches.update(policy_stats)  # Log MPO stats.

        return fetches

    def step(self):
        self._total_steps = self._total_steps + 1
        fetches = self._step()

        # Compute elapsed time.
        timestamp = time.time()
        self._timestamp = timestamp

        # Update our counts and record it.
        fetches.update({"total_steps": self._total_steps})

        if self._total_steps % self._log_every == 0:
            wandb.log({"learner/" + k: v for k, v in fetches.items()})

    def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
        return [to_numpy(self._variables[name]) for name in names]


class EvaluatorActor:

    def __init__(
        self, policy_network: snt.Module, learner: MPOLearner, update_period: int
    ):
        self._learner = learner
        self._policy_network = policy_network
        self._update_period = update_period
        self.label = "evaluator"

    @tf.function
    def _policy(self, observation):
        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = add_batch_dim(observation)

        # Compute the policy, conditioned on the observation.
        policy = self._policy_network(batched_observation)

        # Sample from the policy if it is stochastic.
        action = (
            policy.sample()
            if isinstance(policy, tfp.distributions.Distribution)
            else policy
        )

        return action

    def select_action(self, observation):
        # Pass the observation through the policy network.
        action = self._policy(observation)

        # Return a numpy array with squeezed out batch dimension.
        return to_numpy_squeeze(action)

    def observe_first(self, _: dm_env.TimeStep):
        pass

    def observe(self, _, __: dm_env.TimeStep):
        pass

    def update(self, total_steps: int):
        if total_steps % self._update_period == 0:
            new_variables = tree.flatten(self._learner.get_variables(["policy"]))
            old_variables = tree.flatten(
                list({"policy": self._policy_network.variables}.values())
            )
            for new, old in zip(new_variables, old_variables):
                old.assign(new)


class FeedForwardActor:
    """A feed-forward actor.

    An actor based on a feed-forward policy which takes non-batched observations
    and outputs non-batched actions. It also allows adding experiences to replay
    and updating the weights from the policy on the learner.
    """

    def __init__(
        self,
        policy_network: snt.Module,
        adder: Any,
        learner: MPOLearner,
        update_period: int,
    ):
        self._adder = adder
        self._learner = learner
        self._policy_network = policy_network
        self._update_period = update_period
        self.label = "actor"

    @tf.function
    def _policy(self, observation):
        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = add_batch_dim(observation)

        # Compute the policy, conditioned on the observation.
        policy = self._policy_network(batched_observation)

        # Sample from the policy if it is stochastic.
        action = (
            policy.sample()
            if isinstance(policy, tfp.distributions.Distribution)
            else policy
        )

        return action

    def select_action(self, observation):
        # Pass the observation through the policy network.
        action = self._policy(observation)

        # Return a numpy array with squeezed out batch dimension.
        return to_numpy_squeeze(action)

    def observe_first(self, timestep: dm_env.TimeStep):
        self._adder.add_first(timestep)

    def observe(self, action, next_timestep: dm_env.TimeStep):
        self._adder.add(action, next_timestep)

    def update(self, total_steps: int):
        if total_steps % self._update_period == 0:
            new_variables = tree.flatten(self._learner.get_variables(["policy"]))
            old_variables = tree.flatten(
                list({"policy": self._policy_network.variables}.values())
            )
            for new, old in zip(new_variables, old_variables):
                old.assign(new)


class EnvironmentLoop:
    def __init__(
        self,
        environment: dm_env.Environment,
        actor,
    ):
        self._environment = environment
        self._actor = actor
        self._total_steps = 0

    def run_episode(self) -> Mapping[str, Any]:
        """Run one episode.

        Each episode is a loop which interacts first with the environment to get an
        observation and then give that observation to the agent in order to retrieve
        an action.
        """
        # Reset any counts and start the environment.
        episode_start_time = time.time()
        select_action_durations: List[float] = []
        env_step_durations: List[float] = []
        episode_steps: int = 0

        # For evaluation, this keeps track of the total undiscounted reward
        # accumulated during the episode.
        episode_return = tree.map_structure(
            _generate_zeros_from_spec, self._environment.reward_spec()
        )
        env_reset_start = time.time()
        timestep = self._environment.reset()
        env_reset_duration = time.time() - env_reset_start
        # Make the first observation.
        self._actor.observe_first(timestep)

        # Run an episode.
        while not timestep.last():
            # Book-keeping.
            episode_steps += 1
            self._total_steps += 1

            # Generate an action from the agent's policy.
            select_action_start = time.time()
            action = self._actor.select_action(timestep.observation)
            select_action_durations.append(time.time() - select_action_start)

            # Step the environment with the agent's selected action.
            env_step_start = time.time()
            timestep = self._environment.step(action)
            env_step_durations.append(time.time() - env_step_start)

            # Have the agent and observers observe the timestep.
            self._actor.observe(action, timestep)

            # Give the actor the opportunity to update itself.
            self._actor.update(self._total_steps)

            # Equivalent to: episode_return += timestep.reward
            # We capture the return value because if timestep.reward is a JAX
            # DeviceArray, episode_return will not be mutated in-place. (In all other
            # cases, the returned episode_return will be the same object as the
            # argument episode_return.)
            episode_return = tree.map_structure(
                operator.iadd, episode_return, timestep.reward
            )

        # Collect the results and combine with counts.
        steps_per_second = episode_steps / (time.time() - episode_start_time)
        result = {
            "episode_length": episode_steps,
            "episode_return": episode_return,
            "steps_per_second": steps_per_second,
            "env_reset_duration_sec": env_reset_duration,
            "select_action_duration_sec": np.mean(select_action_durations),
            "env_step_duration_sec": np.mean(env_step_durations),
            "total_steps": self._total_steps,
        }
        return result

    def run(
        self,
        num_episodes: Optional[int] = None,
        num_steps: Optional[int] = None,
    ) -> int:
        if not (num_episodes is None or num_steps is None):
            raise ValueError('Either "num_episodes" or "num_steps" should be None.')

        def should_terminate(episode_count: int, step_count: int) -> bool:
            return (num_episodes is not None and episode_count >= num_episodes) or (
                num_steps is not None and step_count >= num_steps
            )

        episode_count: int = 0
        while not should_terminate(episode_count, self._total_steps):
            episode_start = time.time()
            result = self.run_episode()
            result = {**result, **{"episode_duration": time.time() - episode_start}}
            episode_count += 1
            wandb.log(
                {self._actor.label + "/" + k: v for k, v in result.items()},
            )

        return self._total_steps


def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
    return np.zeros(spec.shape, spec.dtype)


class StochasticMeanHead(snt.Module):
    """Simple sonnet module to produce the mean of a tfp.Distribution."""

    def __call__(self, distribution: tfp.distributions.Distribution):
        return distribution.mean()


class MPO:

    def __init__(
        self,
        network_factory: Callable[[specs.BoundedArray], Dict[str, snt.Module]],
        n_step: int,
        max_actor_steps: int,
        batch_size: int = 256,
        prefetch_size: int = 4,
        min_replay_size: int = 1000,
        max_replay_size: int = 1000000,
        samples_per_insert=32.0,
        num_samples: int = 20,
        additional_discount: float = 0.99,
        target_policy_update_period: int = 100,
        target_critic_update_period: int = 100,
        variable_update_period: int = 1000,
        domain_name: str = "cartpole",
        task_name: str = "balance",
    ):

        self._domain_name = domain_name
        self._task_name = task_name

        environment = _make_environment(
            domain_name=self._domain_name,
            task_name=self._task_name,
        )
        environment_spec = EnvironmentSpec(
            observations=environment.observation_spec(),
            actions=environment.action_spec(),
            rewards=environment.reward_spec(),
            discounts=environment.discount_spec(),
        )
        del environment

        self._network_factory = network_factory
        self._environment_spec = environment_spec
        self._batch_size = batch_size
        self._prefetch_size = prefetch_size
        self._min_replay_size = min_replay_size
        self._max_replay_size = max_replay_size
        self._samples_per_insert = samples_per_insert
        self._n_step = n_step
        self._additional_discount = additional_discount
        self._num_samples = num_samples
        self._target_policy_update_period = target_policy_update_period
        self._target_critic_update_period = target_critic_update_period
        self._variable_update_period = variable_update_period
        self._max_actor_steps = max_actor_steps

    def replay(self):
        # This function no longer creates reverb tables. The MPO.run will create
        # an in-memory replay buffer and pass it to actors/learners.
        # Keep API for backward compatibility: return a placeholder.
        return None

    def learner(
        self,
        replay,
    ):
        """The Learning part of the agent."""

        act_spec = self._environment_spec.actions
        obs_spec = self._environment_spec.observations

        # Create online and target networks.
        online_networks = self._network_factory(act_spec)
        target_networks = self._network_factory(act_spec)

        # Make sure observation networks are Sonnet Modules.
        observation_network = online_networks.get("observation", tf.identity)
        observation_network = to_sonnet_module(observation_network)
        online_networks["observation"] = observation_network
        target_observation_network = target_networks.get("observation", tf.identity)
        target_observation_network = to_sonnet_module(target_observation_network)
        target_networks["observation"] = target_observation_network

        # Get embedding spec and create observation network variables.
        emb_spec = create_variables(observation_network, [obs_spec])

        create_variables(online_networks["policy"], [emb_spec])
        create_variables(online_networks["critic"], [emb_spec, act_spec])
        create_variables(target_networks["observation"], [obs_spec])
        create_variables(target_networks["policy"], [emb_spec])
        create_variables(target_networks["critic"], [emb_spec, act_spec])

        # The dataset object to learn from: use in-memory dataset created from the
        # provided replay buffer. The dataset yields 1-transition elements; batch
        # and prefetch as before.
        dataset = make_in_memory_dataset(replay)
        dataset = dataset.batch(self._batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self._prefetch_size)

        # Return the learning agent.
        return MPOLearner(
            policy_network=online_networks["policy"],
            critic_network=online_networks["critic"],
            observation_network=observation_network,
            target_policy_network=target_networks["policy"],
            target_critic_network=target_networks["critic"],
            target_observation_network=target_observation_network,
            discount=self._additional_discount,
            num_samples=self._num_samples,
            target_policy_update_period=self._target_policy_update_period,
            target_critic_update_period=self._target_critic_update_period,
            policy_loss_module=MPOLoss(
                epsilon=1e-1,
                epsilon_penalty=1e-3,
                epsilon_mean=2.5e-3,
                epsilon_stddev=1e-6,
                init_log_temperature=10.0,
                init_log_alpha_mean=10.0,
                init_log_alpha_stddev=1000.0,
            ),
            dataset=dataset,
        )

    def actor(
        self,
        replay: SimpleReplayBuffer,
        learner: MPOLearner,
    ) -> EnvironmentLoop:
        """The actor process."""

        action_spec = self._environment_spec.actions
        observation_spec = self._environment_spec.observations

        # Create environment and target networks to act with.
        environment = _make_environment(
            domain_name=self._domain_name, task_name=self._task_name
        )
        agent_networks = self._network_factory(action_spec)

        # Create a stochastic behavior policy.
        behavior_modules = [
            agent_networks.get("observation", tf.identity),
            agent_networks.get("policy"),
            StochasticSamplingHead(),
        ]
        behavior_network = snt.Sequential(behavior_modules)

        # Ensure network variables are created.
        create_variables(behavior_network, [observation_spec])

        # Component to add things into replay: use in-memory adder.
        adder = NStepTransitionAdder(
            replay=replay, n_step=self._n_step, discount=self._additional_discount
        )

        actor = FeedForwardActor(
            policy_network=behavior_network,
            adder=adder,
            learner=learner,
            update_period=self._variable_update_period,
        )

        # Create the run loop and return it.
        return EnvironmentLoop(environment=environment, actor=actor)

    def evaluator(
        self,
        learner: MPOLearner,
    ):
        action_spec = self._environment_spec.actions
        observation_spec = self._environment_spec.observations

        # Create environment and target networks to act with.
        environment = _make_environment(
            domain_name=self._domain_name, task_name=self._task_name
        )
        agent_networks = self._network_factory(action_spec)

        # Create a stochastic behavior policy.
        evaluator_modules = [
            agent_networks.get("observation", tf.identity),
            agent_networks.get("policy"),
            StochasticMeanHead(),
        ]

        if isinstance(action_spec, specs.BoundedArray):
            evaluator_modules += [ClipToSpec(action_spec)]
        evaluator_network = snt.Sequential(evaluator_modules)

        # Ensure network variables are created.
        create_variables(evaluator_network, [observation_spec])

        evaluator = EvaluatorActor(
            policy_network=evaluator_network,
            learner=learner,
            update_period=self._variable_update_period,
        )

        return EnvironmentLoop(environment=environment, actor=evaluator)

    def run(self):
        # Create in-memory replay buffer instead of reverb server/client.
        replay = SimpleReplayBuffer(max_size=self._max_replay_size)
        # Create learner/actor/evaluator.
        learner = self.learner(replay)
        actor = self.actor(replay, learner)
        evaluator = self.evaluator(learner)
        threads = [
            threading.Thread(
                target=learner.run,
                kwargs={"num_steps": self._max_actor_steps},
                daemon=True,
            ),
            threading.Thread(
                target=actor.run,
                kwargs={"num_steps": self._max_actor_steps},
                daemon=True,
            ),
            threading.Thread(target=evaluator.run, daemon=True),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()


"""Implements the MPO losses.

The MPO loss is implemented as a Sonnet module rather than a function so that it
can hold its own dual variables, as instances of `tf.Variable`, which it creates
the first time the module is called.

Tensor shapes are annotated, where helpful, as follow:
  B: batch size,
  N: number of sampled actions, see MPO paper for more details,
  D: dimensionality of the action space.
"""


class MPOLoss(snt.Module):
    """MPO loss with decoupled KL constraints as in (Abdolmaleki et al., 2018).

    This implementation of the MPO loss includes the following features, as
    options:
    - Satisfying the KL-constraint on a per-dimension basis (on by default);
    - Penalizing actions that fall outside of [-1, 1] (on by default) as a
        special case of multi-objective MPO (MO-MPO; Abdolmaleki et al., 2020).
    For best results on the control suite, keep both of these on.

    (Abdolmaleki et al., 2018): https://arxiv.org/pdf/1812.02256.pdf
    (Abdolmaleki et al., 2020): https://arxiv.org/pdf/2005.07513.pdf
    """

    def __init__(
        self,
        epsilon: float,
        epsilon_mean: float,
        epsilon_stddev: float,
        init_log_temperature: float,
        init_log_alpha_mean: float,
        init_log_alpha_stddev: float,
        per_dim_constraining: bool = True,
        action_penalization: bool = True,
        epsilon_penalty: float = 0.001,
        name: str = "MPO",
    ):
        """Initialize and configure the MPO loss.

        Args:
          epsilon: KL constraint on the non-parametric auxiliary policy, the one
            associated with the dual variable called temperature.
          epsilon_mean: KL constraint on the mean of the Gaussian policy, the one
            associated with the dual variable called alpha_mean.
          epsilon_stddev: KL constraint on the stddev of the Gaussian policy, the
            one associated with the dual variable called alpha_mean.
          init_log_temperature: initial value for the temperature in log-space, note
            a softplus (rather than an exp) will be used to transform this.
          init_log_alpha_mean: initial value for the alpha_mean in log-space, note
            a softplus (rather than an exp) will be used to transform this.
          init_log_alpha_stddev: initial value for the alpha_stddev in log-space,
            note a softplus (rather than an exp) will be used to transform this.
          per_dim_constraining: whether to enforce the KL constraint on each
            dimension independently; this is the default. Otherwise the overall KL
            is constrained, which allows some dimensions to change more at the
            expense of others staying put.
          action_penalization: whether to use a KL constraint to penalize actions
            via the MO-MPO algorithm.
          epsilon_penalty: KL constraint on the probability of violating the action
            constraint.
          name: a name for the module, passed directly to snt.Module.

        """
        super().__init__(name=name)

        # MPO constrain thresholds.
        self._epsilon = tf.constant(epsilon)
        self._epsilon_mean = tf.constant(epsilon_mean)
        self._epsilon_stddev = tf.constant(epsilon_stddev)

        # Initial values for the constraints' dual variables.
        self._init_log_temperature = init_log_temperature
        self._init_log_alpha_mean = init_log_alpha_mean
        self._init_log_alpha_stddev = init_log_alpha_stddev

        # Whether to penalize out-of-bound actions via MO-MPO and its corresponding
        # constraint threshold.
        self._action_penalization = action_penalization
        self._epsilon_penalty = tf.constant(epsilon_penalty)

        # Whether to ensure per-dimension KL constraint satisfication.
        self._per_dim_constraining = per_dim_constraining

    @snt.once
    def create_dual_variables_once(self, shape: tf.TensorShape, dtype: tf.DType):
        """Creates the dual variables the first time the loss module is called."""

        # Create the dual variables.
        self._log_temperature = tf.Variable(
            initial_value=[self._init_log_temperature],
            dtype=dtype,
            name="log_temperature",
            shape=(1,),
        )
        self._log_alpha_mean = tf.Variable(
            initial_value=tf.fill(shape, self._init_log_alpha_mean),
            dtype=dtype,
            name="log_alpha_mean",
            shape=shape,
        )
        self._log_alpha_stddev = tf.Variable(
            initial_value=tf.fill(shape, self._init_log_alpha_stddev),
            dtype=dtype,
            name="log_alpha_stddev",
            shape=shape,
        )

        # Cast constraint thresholds to the expected dtype.
        self._epsilon = tf.cast(self._epsilon, dtype)
        self._epsilon_mean = tf.cast(self._epsilon_mean, dtype)
        self._epsilon_stddev = tf.cast(self._epsilon_stddev, dtype)

        # Maybe create the action penalization dual variable.
        if self._action_penalization:
            self._epsilon_penalty = tf.cast(self._epsilon_penalty, dtype)
            self._log_penalty_temperature = tf.Variable(
                initial_value=[self._init_log_temperature],
                dtype=dtype,
                name="log_penalty_temperature",
                shape=(1,),
            )

    def __call__(
        self,
        online_action_distribution: Union[
            tfp.distributions.MultivariateNormalDiag, tfp.distributions.Independent
        ],
        target_action_distribution: Union[
            tfp.distributions.MultivariateNormalDiag, tfp.distributions.Independent
        ],
        actions: tf.Tensor,  # Shape [N, B, D].
        q_values: tf.Tensor,  # Shape [N, B].
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Computes the decoupled MPO loss.

        Args:
          online_action_distribution: online distribution returned by the online
            policy network; expects batch_dims of [B] and event_dims of [D].
          target_action_distribution: target distribution returned by the target
            policy network; expects same shapes as online distribution.
          actions: actions sampled from the target policy; expects shape [N, B, D].
          q_values: Q-values associated with each action; expects shape [N, B].

        Returns:
          Loss, combining the policy loss, KL penalty, and dual losses required to
            adapt the dual variables.
          Stats, for diagnostics and tracking performance.
        """

        # Cast `MultivariateNormalDiag`s to Independent Normals.
        # The latter allows us to satisfy KL constraints per-dimension.
        if isinstance(
            target_action_distribution, tfp.distributions.MultivariateNormalDiag
        ):
            target_action_distribution = tfp.distributions.Independent(
                tfp.distributions.Normal(
                    target_action_distribution.mean(),
                    target_action_distribution.stddev(),
                )
            )
            online_action_distribution = tfp.distributions.Independent(
                tfp.distributions.Normal(
                    online_action_distribution.mean(),
                    online_action_distribution.stddev(),
                )
            )

        # Infer the shape and dtype of dual variables.
        scalar_dtype = q_values.dtype
        if self._per_dim_constraining:
            dual_variable_shape = target_action_distribution.distribution.kl_divergence(
                online_action_distribution.distribution
            ).shape[
                1:
            ]  # Should be [D].
        else:
            dual_variable_shape = target_action_distribution.kl_divergence(
                online_action_distribution
            ).shape[
                1:
            ]  # Should be [1].

        # Create dual variables for the KL constraints; only happens the first call.
        self.create_dual_variables_once(dual_variable_shape, scalar_dtype)

        # Project dual variables to ensure they stay positive.
        min_log_temperature = tf.constant(-18.0, scalar_dtype)
        min_log_alpha = tf.constant(-18.0, scalar_dtype)
        self._log_temperature.assign(
            tf.maximum(min_log_temperature, self._log_temperature)
        )
        self._log_alpha_mean.assign(tf.maximum(min_log_alpha, self._log_alpha_mean))
        self._log_alpha_stddev.assign(tf.maximum(min_log_alpha, self._log_alpha_stddev))

        # Transform dual variables from log-space.
        # Note: using softplus instead of exponential for numerical stability.
        temperature = tf.math.softplus(self._log_temperature) + _MPO_FLOAT_EPSILON
        alpha_mean = tf.math.softplus(self._log_alpha_mean) + _MPO_FLOAT_EPSILON
        alpha_stddev = tf.math.softplus(self._log_alpha_stddev) + _MPO_FLOAT_EPSILON

        # Get online and target means and stddevs in preparation for decomposition.
        online_mean = online_action_distribution.distribution.mean()
        online_scale = online_action_distribution.distribution.stddev()
        target_mean = target_action_distribution.distribution.mean()
        target_scale = target_action_distribution.distribution.stddev()

        # Compute normalized importance weights, used to compute expectations with
        # respect to the non-parametric policy; and the temperature loss, used to
        # adapt the tempering of Q-values.
        normalized_weights, loss_temperature = compute_weights_and_temperature_loss(
            q_values, self._epsilon, temperature
        )

        # Only needed for diagnostics: Compute estimated actualized KL between the
        # non-parametric and current target policies.
        kl_nonparametric = compute_nonparametric_kl_from_normalized_weights(
            normalized_weights
        )
        penalty_kl_nonparametric = None
        if self._action_penalization:
            # Project and transform action penalization temperature.
            self._log_penalty_temperature.assign(
                tf.maximum(min_log_temperature, self._log_penalty_temperature)
            )
            penalty_temperature = (
                tf.math.softplus(self._log_penalty_temperature) + _MPO_FLOAT_EPSILON
            )

            # Compute action penalization cost.
            # Note: the cost is zero in [-1, 1] and quadratic beyond.
            diff_out_of_bound = actions - tf.clip_by_value(actions, -1.0, 1.0)
            cost_out_of_bound = -tf.norm(diff_out_of_bound, axis=-1)

            penalty_normalized_weights, loss_penalty_temperature = (
                compute_weights_and_temperature_loss(
                    cost_out_of_bound, self._epsilon_penalty, penalty_temperature
                )
            )

            # Only needed for diagnostics: Compute estimated actualized KL between the
            # non-parametric and current target policies.
            penalty_kl_nonparametric = compute_nonparametric_kl_from_normalized_weights(
                penalty_normalized_weights
            )

            # Combine normalized weights.
            normalized_weights += penalty_normalized_weights
            loss_temperature += loss_penalty_temperature
        # Decompose the online policy into fixed-mean & fixed-stddev distributions.
        # This has been documented as having better performance in bandit settings,
        # see e.g. https://arxiv.org/pdf/1812.02256.pdf.
        fixed_stddev_distribution = tfp.distributions.Independent(
            tfp.distributions.Normal(loc=online_mean, scale=target_scale)
        )
        fixed_mean_distribution = tfp.distributions.Independent(
            tfp.distributions.Normal(loc=target_mean, scale=online_scale)
        )

        # Compute the decomposed policy losses.
        loss_policy_mean = compute_cross_entropy_loss(
            actions, normalized_weights, fixed_stddev_distribution
        )
        loss_policy_stddev = compute_cross_entropy_loss(
            actions, normalized_weights, fixed_mean_distribution
        )

        # Compute the decomposed KL between the target and online policies.
        if self._per_dim_constraining:
            kl_mean = target_action_distribution.distribution.kl_divergence(
                fixed_stddev_distribution.distribution
            )  # Shape [B, D].
            kl_stddev = target_action_distribution.distribution.kl_divergence(
                fixed_mean_distribution.distribution
            )  # Shape [B, D].
        else:
            kl_mean = target_action_distribution.kl_divergence(
                fixed_stddev_distribution
            )  # Shape [B].
            kl_stddev = target_action_distribution.kl_divergence(
                fixed_mean_distribution
            )  # Shape [B].

        # Compute the alpha-weighted KL-penalty and dual losses to adapt the alphas.
        loss_kl_mean, loss_alpha_mean = compute_parametric_kl_penalty_and_dual_loss(
            kl_mean, alpha_mean, self._epsilon_mean
        )
        loss_kl_stddev, loss_alpha_stddev = compute_parametric_kl_penalty_and_dual_loss(
            kl_stddev, alpha_stddev, self._epsilon_stddev
        )

        # Combine losses.
        loss_policy = loss_policy_mean + loss_policy_stddev
        loss_kl_penalty = loss_kl_mean + loss_kl_stddev
        loss_dual = loss_alpha_mean + loss_alpha_stddev + loss_temperature
        loss = loss_policy + loss_kl_penalty + loss_dual

        stats = {}
        # Dual Variables.
        stats["dual_alpha_mean"] = tf.reduce_mean(alpha_mean)
        stats["dual_alpha_stddev"] = tf.reduce_mean(alpha_stddev)
        stats["dual_temperature"] = tf.reduce_mean(temperature)
        # Losses.
        stats["loss_policy"] = tf.reduce_mean(loss)
        stats["loss_alpha"] = tf.reduce_mean(loss_alpha_mean + loss_alpha_stddev)
        stats["loss_temperature"] = tf.reduce_mean(loss_temperature)
        # KL measurements.
        stats["kl_q_rel"] = tf.reduce_mean(kl_nonparametric) / self._epsilon

        if self._action_penalization:
            stats["penalty_kl_q_rel"] = (
                tf.reduce_mean(penalty_kl_nonparametric) / self._epsilon_penalty
            )

        stats["kl_mean_rel"] = tf.reduce_mean(kl_mean) / self._epsilon_mean
        stats["kl_stddev_rel"] = tf.reduce_mean(kl_stddev) / self._epsilon_stddev
        if self._per_dim_constraining:
            # When KL is constrained per-dimension, we also log per-dimension min and
            # max of mean/std of the realized KL costs.
            stats["kl_mean_rel_min"] = (
                tf.reduce_min(tf.reduce_mean(kl_mean, axis=0)) / self._epsilon_mean
            )
            stats["kl_mean_rel_max"] = (
                tf.reduce_max(tf.reduce_mean(kl_mean, axis=0)) / self._epsilon_mean
            )
            stats["kl_stddev_rel_min"] = (
                tf.reduce_min(tf.reduce_mean(kl_stddev, axis=0)) / self._epsilon_stddev
            )
            stats["kl_stddev_rel_max"] = (
                tf.reduce_max(tf.reduce_mean(kl_stddev, axis=0)) / self._epsilon_stddev
            )
        # Q measurements.
        stats["q_min"] = tf.reduce_mean(tf.reduce_min(q_values, axis=0))
        stats["q_max"] = tf.reduce_mean(tf.reduce_max(q_values, axis=0))
        # If the policy has standard deviation, log summary stats for this as well.
        pi_stddev = online_action_distribution.distribution.stddev()
        stats["pi_stddev_min"] = tf.reduce_mean(tf.reduce_min(pi_stddev, axis=-1))
        stats["pi_stddev_max"] = tf.reduce_mean(tf.reduce_max(pi_stddev, axis=-1))
        # Condition number of the diagonal covariance (actually, stddev) matrix.
        stats["pi_stddev_cond"] = tf.reduce_mean(
            tf.reduce_max(pi_stddev, axis=-1) / tf.reduce_min(pi_stddev, axis=-1)
        )

        return loss, stats


def compute_weights_and_temperature_loss(
    q_values: tf.Tensor,
    epsilon: float,
    temperature: tf.Variable,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Computes normalized importance weights for the policy optimization.

    Args:
      q_values: Q-values associated with the actions sampled from the target
        policy; expected shape [N, B].
      epsilon: Desired constraint on the KL between the target and non-parametric
        policies.
      temperature: Scalar used to temper the Q-values before computing normalized
        importance weights from them. This is really the Lagrange dual variable
        in the constrained optimization problem, the solution of which is the
        non-parametric policy targeted by the policy loss.

    Returns:
      Normalized importance weights, used for policy optimization.
      Temperature loss, used to adapt the temperature.
    """

    # Temper the given Q-values using the current temperature.
    tempered_q_values = tf.stop_gradient(q_values) / temperature

    # Compute the normalized importance weights used to compute expectations with
    # respect to the non-parametric policy.
    normalized_weights = tf.nn.softmax(tempered_q_values, axis=0)
    normalized_weights = tf.stop_gradient(normalized_weights)

    # Compute the temperature loss (dual of the E-step optimization problem).
    q_logsumexp = tf.reduce_logsumexp(tempered_q_values, axis=0)
    log_num_actions = tf.math.log(tf.cast(q_values.shape[0], tf.float32))
    loss_temperature = epsilon + tf.reduce_mean(q_logsumexp) - log_num_actions
    loss_temperature = temperature * loss_temperature

    return normalized_weights, loss_temperature


def compute_nonparametric_kl_from_normalized_weights(
    normalized_weights: tf.Tensor,
) -> tf.Tensor:
    """Estimate the actualized KL between the non-parametric and target policies."""

    # Compute integrand.
    num_action_samples = tf.cast(normalized_weights.shape[0], tf.float32)
    integrand = tf.math.log(num_action_samples * normalized_weights + 1e-8)

    # Return the expectation with respect to the non-parametric policy.
    return tf.reduce_sum(normalized_weights * integrand, axis=0)


def compute_cross_entropy_loss(
    sampled_actions: tf.Tensor,
    normalized_weights: tf.Tensor,
    online_action_distribution: tfp.distributions.Distribution,
) -> tf.Tensor:
    """Compute cross-entropy online and the reweighted target policy.

    Args:
      sampled_actions: samples used in the Monte Carlo integration in the policy
        loss. Expected shape is [N, B, ...], where N is the number of sampled
        actions and B is the number of sampled states.
      normalized_weights: target policy multiplied by the exponentiated Q values
        and normalized; expected shape is [N, B].
      online_action_distribution: policy to be optimized.

    Returns:
      loss_policy_gradient: the cross-entropy loss that, when differentiated,
        produces the policy gradient.
    """

    # Compute the M-step loss.
    log_prob = online_action_distribution.log_prob(sampled_actions)

    # Compute the weighted average log-prob using the normalized weights.
    loss_policy_gradient = -tf.reduce_sum(log_prob * normalized_weights, axis=0)

    # Return the mean loss over the batch of states.
    return tf.reduce_mean(loss_policy_gradient, axis=0)


def compute_parametric_kl_penalty_and_dual_loss(
    kl: tf.Tensor,
    alpha: tf.Variable,
    epsilon: float,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Computes the KL cost to be added to the Lagragian and its dual loss.

    The KL cost is simply the alpha-weighted KL divergence and it is added as a
    regularizer to the policy loss. The dual variable alpha itself has a loss that
    can be minimized to adapt the strength of the regularizer to keep the KL
    between consecutive updates at the desired target value of epsilon.

    Args:
      kl: KL divergence between the target and online policies.
      alpha: Lagrange multipliers (dual variables) for the KL constraints.
      epsilon: Desired value for the KL.

    Returns:
      loss_kl: alpha-weighted KL regularization to be added to the policy loss.
      loss_alpha: The Lagrange dual loss minimized to adapt alpha.
    """

    # Compute the mean KL over the batch.
    mean_kl = tf.reduce_mean(kl, axis=0)

    # Compute the regularization.
    loss_kl = tf.reduce_sum(tf.stop_gradient(alpha) * mean_kl)

    # Compute the dual loss.
    loss_alpha = tf.reduce_sum(alpha * (epsilon - tf.stop_gradient(mean_kl)))

    return loss_kl, loss_alpha


class StochasticSamplingHead(snt.Module):
    """Simple sonnet module to sample from a tfp.Distribution."""

    def __call__(self, distribution: tfp.distributions.Distribution):
        return distribution.sample()


def _uniform_initializer():
    return tf.initializers.VarianceScaling(
        distribution="uniform", mode="fan_out", scale=0.333
    )


class LayerNormMLP(snt.Module):
    """Simple feedforward MLP torso with initial layer-norm.

    This module is an MLP which uses LayerNorm (with a tanh normalizer) on the
    first layer and non-linearities (elu) on all but the last remaining layers.
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        w_init: Optional[snt.initializers.Initializer] = None,
        activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.elu,
        activate_final: bool = False,
    ):
        """Construct the MLP.

        Args:
          layer_sizes: a sequence of ints specifying the size of each layer.
          w_init: initializer for Linear weights.
          activation: activation function to apply between linear layers. Defaults
            to ELU. Note! This is different from snt.nets.MLP's default.
          activate_final: whether or not to use the activation function on the final
            layer of the neural network.
        """
        super().__init__(name="feedforward_mlp_torso")

        self._network = snt.Sequential(
            [
                snt.Linear(layer_sizes[0], w_init=w_init or _uniform_initializer()),
                snt.LayerNorm(
                    axis=slice(1, None), create_scale=True, create_offset=True
                ),
                tf.nn.tanh,
                snt.nets.MLP(
                    layer_sizes[1:],
                    w_init=w_init or _uniform_initializer(),
                    activation=activation,
                    activate_final=activate_final,
                ),
            ]
        )

    def __call__(self, observations) -> tf.Tensor:
        """Forwards the policy network."""
        return self._network(batch_concat(observations))


class MultivariateNormalDiagHead(snt.Module):
    """Module that produces a multivariate normal distribution using tfp.distributions.Independent or tfp.distributions.MultivariateNormalDiag."""

    def __init__(
        self,
        num_dimensions: int,
        init_scale: float = 0.3,
        min_scale: float = 1e-6,
        tanh_mean: bool = False,
        w_init=tf.initializers.VarianceScaling(1e-4),
        b_init=tf.initializers.Zeros(),
    ):
        """Initialization.

        Args:
          num_dimensions: Number of dimensions of MVN distribution.
          init_scale: Initial standard deviation.
          min_scale: Minimum standard deviation.
          tanh_mean: Whether to transform the mean (via tanh) before passing it to
            the distribution.
          w_init: Initialization for linear layer weights.
          b_init: Initialization for linear layer biases.
        """
        super().__init__(name="MultivariateNormalDiagHead")
        self._init_scale = init_scale
        self._min_scale = min_scale
        self._tanh_mean = tanh_mean
        self._mean_layer = snt.Linear(num_dimensions, w_init=w_init, b_init=b_init)
        self._scale_layer = snt.Linear(num_dimensions, w_init=w_init, b_init=b_init)

    def __call__(self, inputs: tf.Tensor):
        zero = tf.constant(0, dtype=inputs.dtype)
        mean = self._mean_layer(inputs)

        scale = tf.nn.softplus(self._scale_layer(inputs))
        scale *= self._init_scale / tf.nn.softplus(zero)
        scale += self._min_scale

        # Maybe transform the mean.
        if self._tanh_mean:
            mean = tf.tanh(mean)

        dist = tfp.distributions.Independent(
            tfp.distributions.Normal(loc=mean, scale=scale)
        )

        return dist


class NearZeroInitializedLinear(snt.Linear):
    """Simple linear layer, initialized at near zero weights and zero biases."""

    def __init__(self, output_size: int, scale: float = 1e-4):
        super().__init__(output_size, w_init=tf.initializers.VarianceScaling(scale))


class CriticMultiplexer(snt.Module):
    """Module connecting a critic torso to (transformed) observations/actions.

    This takes as input a `critic_network`, an `observation_network`, and an
    `action_network` and returns another network whose outputs are given by
    `critic_network(observation_network(o), action_network(a))`.

    The observations and actions passed to this module are assumed to have a batch
    dimension that match.

    Notes:
    - Either the `observation_` or `action_network` can be `None`, in which case
      the observation or action, resp., are passed to the critic network as is.
    - If all `critic_`, `observation_` and `action_network` are `None`, this
      module reduces to a simple `tf2_utils.batch_concat()`.
    """

    def __init__(
        self,
        critic_network: Optional[Any] = None,
        observation_network: Optional[Any] = None,
        action_network: Optional[Any] = None,
    ):
        self._critic_network = critic_network
        self._observation_network = observation_network
        self._action_network = action_network
        super().__init__(name="critic_multiplexer")

    def __call__(self, observation, action) -> tf.Tensor:

        # Maybe transform observations and actions before feeding them on.
        if self._observation_network:
            observation = self._observation_network(observation)
        if self._action_network:
            action = self._action_network(action)

        if hasattr(observation, "dtype") and hasattr(action, "dtype"):
            if observation.dtype != action.dtype:
                # Observation and action must be the same type for concat to work
                action = tf.cast(action, observation.dtype)

        # Concat observations and actions, with one batch dimension.
        outputs = batch_concat([observation, action])

        # Maybe transform output before returning.
        if self._critic_network:
            outputs = self._critic_network(outputs)

        return outputs


class ClipToSpec(snt.Module):
    """Sonnet module clipping inputs to within a BoundedArraySpec."""

    def __init__(self, spec: specs.BoundedArray, name: str = "clip_to_spec"):
        super().__init__(name=name)
        self._min = spec.minimum
        self._max = spec.maximum

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.clip_by_value(inputs, self._min, self._max)


def make_networks(
    action_spec: specs.BoundedArray,
    policy_layer_sizes: Sequence[int] = (256, 256, 256),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
) -> Dict[str, Any]:
    """Creates networks used by the agent."""

    num_dimensions = np.prod(action_spec.shape, dtype=int)

    policy_network = snt.Sequential(
        [
            LayerNormMLP(policy_layer_sizes, activate_final=True),
            MultivariateNormalDiagHead(num_dimensions, init_scale=0.7),
        ]
    )

    # The multiplexer concatenates the (maybe transformed) observations/actions.
    multiplexer = CriticMultiplexer(action_network=ClipToSpec(action_spec))
    critic_network = snt.Sequential(
        [
            multiplexer,
            LayerNormMLP(critic_layer_sizes, activate_final=True),
            NearZeroInitializedLinear(1),
        ]
    )

    return {
        "policy": policy_network,
        "critic": critic_network,
        "observation": batch_concat,
    }


class CanonicalSinglePrecisionWrapper(dm_env.Environment):
    def __init__(self, environment: dm_env.Environment, clip: bool = False):
        self._environment = environment
        self._clip = clip
        self._raw_action_spec = environment.action_spec()

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self._environment, name)

    @property
    def environment(self) -> dm_env.Environment:
        return self._environment

    def step(self, action) -> dm_env.TimeStep:
        scaled_action = _scale_nested_action(action, self._raw_action_spec, self._clip)
        timestep = self._environment.step(scaled_action)
        return self._convert_timestep(timestep)

    def reset(self) -> dm_env.TimeStep:
        return self._convert_timestep(self._environment.reset())

    def action_spec(self):
        canonical_spec = tree.map_structure(
            self._canonicalize_bounded_spec, self._raw_action_spec
        )
        return self._convert_spec(canonical_spec)

    def discount_spec(self):
        return self._convert_spec(self._environment.discount_spec())

    def observation_spec(self):
        return self._convert_spec(self._environment.observation_spec())

    def reward_spec(self):
        return self._convert_spec(self._environment.reward_spec())

    def close(self):
        return self._environment.close()

    def _canonicalize_bounded_spec(self, spec: specs.Array) -> specs.Array:
        if isinstance(spec, specs.BoundedArray):
            return spec.replace(
                minimum=-np.ones(spec.shape), maximum=np.ones(spec.shape)
            )
        return spec

    def _convert_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
        return timestep._replace(
            reward=self._convert_value(timestep.reward),
            discount=self._convert_value(timestep.discount),
            observation=self._convert_value(timestep.observation),
        )

    def _convert_spec(self, nested_spec):
        def _convert_single_spec(spec: specs.Array):
            if spec.dtype == "O":
                return spec
            if np.issubdtype(spec.dtype, np.float64):
                dtype = np.float32
            elif np.issubdtype(spec.dtype, np.int64):
                dtype = np.int32
            else:
                dtype = spec.dtype
            return spec.replace(dtype=dtype)

        return tree.map_structure(_convert_single_spec, nested_spec)

    def _convert_value(self, nested_value):
        def _convert_single_value(value):
            if value is not None:
                value = np.asarray(value)
                if np.issubdtype(value.dtype, np.float64):
                    value = value.astype(np.float32)
                elif np.issubdtype(value.dtype, np.int64):
                    value = value.astype(np.int32)
            return value

        return tree.map_structure(_convert_single_value, nested_value)


def _scale_nested_action(
    nested_action,
    nested_spec,
    clip: bool,
):
    """Converts a canonical nested action back to the given nested action spec."""

    def _scale_action(action: np.ndarray, spec: specs.Array):
        """Converts a single canonical action back to the given action spec."""
        if isinstance(spec, specs.BoundedArray):
            # Get scale and offset of output action spec.
            scale = spec.maximum - spec.minimum
            offset = spec.minimum

            # Maybe clip the action.
            if clip:
                action = np.clip(action, -1.0, 1.0)

            # Map action to [0, 1].
            action = 0.5 * (action + 1.0)

            # Map action to [spec.minimum, spec.maximum].
            action *= scale
            action += offset

        return action

    return tree.map_structure(_scale_action, nested_action, nested_spec)


def _make_environment(
    domain_name: str = "cartpole",
    task_name: str = "balance",
) -> dm_env.Environment:
    environment = suite.load(domain_name, task_name)
    environment = CanonicalSinglePrecisionWrapper(environment, clip=True)
    return environment


def parse_args():
    parser = argparse.ArgumentParser(description="Train ACME MPO agent.")
    parser.add_argument("--domain", default="cartpole")
    parser.add_argument("--task", default="balance")
    parser.add_argument("--max_actor_steps", type=int, default=None)
    parser.add_argument("--n_step", type=int, default=5)
    return parser.parse_args()


def main(args):
    wandb.init(
        entity="adrian-research",
        group=f"{args.domain}_{args.task}",
        project="lp_mpo_single_file",
        config={
            "domain": args.domain,
            "task": args.task,
            "max_actor_steps": args.max_actor_steps,
        },
    )
    program_builder = MPO(
        network_factory=make_networks,
        target_policy_update_period=25,
        max_actor_steps=args.max_actor_steps,
        domain_name=args.domain,
        task_name=args.task,
        n_step=args.n_step,
    )
    program_builder.run()


if __name__ == "__main__":
    main(parse_args())
