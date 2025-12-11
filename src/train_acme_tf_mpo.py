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
)
import time
import collections
import threading
from absl import app
from absl import flags
import numpy as np
import sonnet as snt
import reverb
import wandb
import tree
import operator
import itertools
import os

from acme import types
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
from acme.adders import reverb as adders
from acme import wrappers

import dm_env
from dm_env import specs
from dm_control import suite


import tensorflow as tf
import tensorflow_probability as tfp
import trfl


tfd = tfp.distributions
TensorTransformation = Union[snt.Module, Callable[[types.NestedTensor], tf.Tensor]]

FLAGS = flags.FLAGS
_MAX_ACTOR_STEPS = flags.DEFINE_integer(
    "max_actor_steps",
    None,
    "Number of actor steps to run; defaults to None for an endless loop.",
)
_DOMAIN = flags.DEFINE_string("domain", "cartpole", "Control suite domain name.")
_TASK = flags.DEFINE_string("task", "balance", "Control suite task name.")
_MPO_FLOAT_EPSILON = 1e-8


def make_reverb_dataset(
    server_address: str,
    batch_size: Optional[int] = None,
    prefetch_size: Optional[int] = None,
    table: str = "priority_table",
    num_parallel_calls: Optional[int] = 12,
    max_in_flight_samples_per_worker: Optional[int] = None,
) -> tf.data.Dataset:
    # This is the default that used to be set by reverb.TFClient.dataset().
    if max_in_flight_samples_per_worker is None and batch_size is None:
        max_in_flight_samples_per_worker = 100
    elif max_in_flight_samples_per_worker is None:
        max_in_flight_samples_per_worker = 2 * batch_size

    # Create mapping from tables to non-zero weights.
    if isinstance(table, str):
        tables = collections.OrderedDict([(table, 1.0)])
    else:
        tables = collections.OrderedDict(
            [(name, weight) for name, weight in table.items() if weight > 0.0]
        )
        if len(tables) <= 0:
            raise ValueError(f"No positive weights in input tables {tables}")

    # Normalize weights.
    total_weight = sum(tables.values())
    tables = collections.OrderedDict(
        [(name, weight / total_weight) for name, weight in tables.items()]
    )

    def _make_dataset(unused_idx: tf.Tensor) -> tf.data.Dataset:
        datasets = ()
        for table_name, weight in tables.items():
            max_in_flight_samples = max(
                1, int(max_in_flight_samples_per_worker * weight)
            )
            dataset = reverb.TrajectoryDataset.from_table_signature(
                server_address=server_address,
                table=table_name,
                max_in_flight_samples_per_worker=max_in_flight_samples,
            )
            datasets += (dataset,)
        if len(datasets) > 1:
            dataset = tf.data.Dataset.sample_from_datasets(
                datasets, weights=tables.values()
            )
        else:
            dataset = datasets[0]

        if batch_size:
            dataset = dataset.batch(batch_size, drop_remainder=True)

        return dataset

    if num_parallel_calls is not None:
        # Create a datasets and interleaves it to create `num_parallel_calls`
        # `TrajectoryDataset`s.
        num_datasets_to_interleave = (
            os.cpu_count()
            if num_parallel_calls == tf.data.AUTOTUNE
            else num_parallel_calls
        )
        dataset = tf.data.Dataset.range(num_datasets_to_interleave).interleave(
            map_func=_make_dataset,
            cycle_length=num_parallel_calls,
            num_parallel_calls=num_parallel_calls,
            deterministic=False,
        )
    else:
        dataset = _make_dataset(tf.constant(0))

    if prefetch_size:
        dataset = dataset.prefetch(prefetch_size)

    return dataset


class EnvironmentSpec(NamedTuple):
    """Full specification of the domains used by a given environment."""

    # TODO(b/144758674): Use NestedSpec type here.
    observations: Any
    actions: Any
    rewards: Any
    discounts: Any


def make_environment_spec(environment: dm_env.Environment) -> EnvironmentSpec:
    """Returns an `EnvironmentSpec` describing values used by an environment."""
    return EnvironmentSpec(
        observations=environment.observation_spec(),
        actions=environment.action_spec(),
        rewards=environment.reward_spec(),
        discounts=environment.discount_spec(),
    )


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
        dataset: tf.data.Dataset,
        counter: counting.Counter,
        observation_network: types.TensorTransformation = tf.identity,
        target_observation_network: types.TensorTransformation = tf.identity,
        policy_loss_module: Optional[snt.Module] = None,
        policy_optimizer: Optional[snt.Optimizer] = None,
        critic_optimizer: Optional[snt.Optimizer] = None,
        dual_optimizer: Optional[snt.Optimizer] = None,
        clipping: bool = True,
        logger: Optional[loggers.Logger] = None,
    ):

        self._counter = counter
        self._logger = logger or loggers.make_default_logger("learner")
        self._discount = discount
        self._num_samples = num_samples
        self._clipping = clipping

        # Necessary to track when to update target networks.
        self._num_steps = tf.Variable(0, dtype=tf.int32)
        self._target_policy_update_period = target_policy_update_period
        self._target_critic_update_period = target_critic_update_period

        # Batch dataset and create iterator.
        # TODO(b/155086959): Fix type stubs and remove.
        self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types

        # Store online and target networks.
        self._policy_network = policy_network
        self._critic_network = critic_network
        self._target_policy_network = target_policy_network
        self._target_critic_network = target_critic_network

        # Make sure observation networks are snt.Module's so they have variables.
        self._observation_network = tf2_utils.to_sonnet_module(observation_network)
        self._target_observation_network = tf2_utils.to_sonnet_module(
            target_observation_network
        )

        self._policy_loss_module = policy_loss_module or MPOLoss(
            epsilon=1e-1,
            epsilon_penalty=1e-3,
            epsilon_mean=2.5e-3,
            epsilon_stddev=1e-6,
            init_log_temperature=10.0,
            init_log_alpha_mean=10.0,
            init_log_alpha_stddev=1000.0,
        )

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

    def run(self, num_steps: Optional[int] = None) -> None:

        iterator = range(num_steps) if num_steps is not None else itertools.count()

        for _ in iterator:
            self.step()

    @tf.function
    def _step(self) -> types.Nest:
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
        transitions: types.Transition = inputs.data

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
            tiled_o_t = tf2_utils.tile_tensor(o_t, self._num_samples)  # [N, B, ...]

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
        # Run the learning step.
        fetches = self._step()

        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp

        # Update our counts and record it.
        counts = self._counter.increment(steps=1, walltime=elapsed_time)
        fetches.update(counts)

        wandb.log(fetches)

        self._logger.write(fetches)

    def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
        return [tf2_utils.to_numpy(self._variables[name]) for name in names]


class EvaluatorActor:

    def __init__(
        self, policy_network: snt.Module, learner: MPOLearner, update_period: int
    ):
        self._learner = learner
        self._policy_network = policy_network
        self._update_period = update_period

    @tf.function
    def _policy(self, observation: types.NestedTensor) -> types.NestedTensor:
        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = tf2_utils.add_batch_dim(observation)

        # Compute the policy, conditioned on the observation.
        policy = self._policy_network(batched_observation)

        # Sample from the policy if it is stochastic.
        action = policy.sample() if isinstance(policy, tfd.Distribution) else policy

        return action

    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        # Pass the observation through the policy network.
        action = self._policy(observation)

        # Return a numpy array with squeezed out batch dimension.
        return tf2_utils.to_numpy_squeeze(action)

    def observe_first(self, _: dm_env.TimeStep):
        pass

    def observe(self, _: types.NestedArray, __: dm_env.TimeStep):
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
        adder: adders.NStepTransitionAdder,
        learner: MPOLearner,
        update_period: int,
    ):
        self._adder = adder
        self._learner = learner
        self._policy_network = policy_network
        self._update_period = update_period

    @tf.function
    def _policy(self, observation: types.NestedTensor) -> types.NestedTensor:
        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = tf2_utils.add_batch_dim(observation)

        # Compute the policy, conditioned on the observation.
        policy = self._policy_network(batched_observation)

        # Sample from the policy if it is stochastic.
        action = policy.sample() if isinstance(policy, tfd.Distribution) else policy

        return action

    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        # Pass the observation through the policy network.
        action = self._policy(observation)

        # Return a numpy array with squeezed out batch dimension.
        return tf2_utils.to_numpy_squeeze(action)

    def observe_first(self, timestep: dm_env.TimeStep):
        self._adder.add_first(timestep)

    def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
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
    """A simple RL environment loop.

    This takes `Environment` and `Actor` instances and coordinates their
    interaction. This can be used as:

      loop = EnvironmentLoop(environment, actor)
      loop.run(num_episodes)

    A `Counter` instance can optionally be given in order to maintain counts
    between different Acme components. If not given a local Counter will be
    created to maintain counts between calls to the `run` method.

    A `Logger` instance can also be passed in order to control the output of the
    loop. If not given a platform-specific default logger will be used as defined
    by utils.loggers.make_default_logger. A string `label` can be passed to easily
    change the label associated with the default logger; this is ignored if a
    `Logger` instance is given.

    A list of 'Observer' instances can be specified to generate additional metrics
    to be logged by the logger. They have access to the 'Environment' instance,
    the current timestep datastruct and the current action.
    """

    def __init__(
        self,
        environment: dm_env.Environment,
        actor,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        label: str = "environment_loop",
    ):
        # Internalize agent and environment.
        self._environment = environment
        self._actor = actor
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger(
            label, steps_key=self._counter.get_steps_key()
        )
        self._total_steps = 0

    def run_episode(self) -> loggers.LoggingData:
        """Run one episode.

        Each episode is a loop which interacts first with the environment to get an
        observation and then give that observation to the agent in order to retrieve
        an action.

        Returns:
          An instance of `loggers.LoggingData`.
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

        # Record counts.
        counts = self._counter.increment(episodes=1, steps=episode_steps)

        # Collect the results and combine with counts.
        steps_per_second = episode_steps / (time.time() - episode_start_time)
        result = {
            "episode_length": episode_steps,
            "episode_return": episode_return,
            "steps_per_second": steps_per_second,
            "env_reset_duration_sec": env_reset_duration,
            "select_action_duration_sec": np.mean(select_action_durations),
            "env_step_duration_sec": np.mean(env_step_durations),
        }
        result.update(counts)
        return result

    def run(
        self,
        num_episodes: Optional[int] = None,
        num_steps: Optional[int] = None,
    ) -> int:
        """Perform the run loop.

        Run the environment loop either for `num_episodes` episodes or for at
        least `num_steps` steps (the last episode is always run until completion,
        so the total number of steps may be slightly more than `num_steps`).
        At least one of these two arguments has to be None.

        Upon termination of an episode a new episode will be started. If the number
        of episodes and the number of steps are not given then this will interact
        with the environment infinitely.

        Args:
          num_episodes: number of episodes to run the loop for.
          num_steps: minimal number of steps to run the loop for.

        Returns:
          Actual number of steps the loop executed.

        Raises:
          ValueError: If both 'num_episodes' and 'num_steps' are not None.
        """

        if not (num_episodes is None or num_steps is None):
            raise ValueError('Either "num_episodes" or "num_steps" should be None.')

        def should_terminate(episode_count: int, step_count: int) -> bool:
            return (num_episodes is not None and episode_count >= num_episodes) or (
                num_steps is not None and step_count >= num_steps
            )

        episode_count: int = 0
        step_count: int = 0
        while not should_terminate(episode_count, step_count):
            episode_start = time.time()
            result = self.run_episode()
            result = {**result, **{"episode_duration": time.time() - episode_start}}
            episode_count += 1
            step_count += int(result["episode_length"])
            # Log the given episode results.
            self._logger.write(result)

        return step_count


def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
    return np.zeros(spec.shape, spec.dtype)


class StochasticMeanHead(snt.Module):
    """Simple sonnet module to produce the mean of a tfp.Distribution."""

    def __call__(self, distribution: tfd.Distribution):
        return distribution.mean()


class MPO:

    def __init__(
        self,
        network_factory: Callable[[specs.BoundedArray], Dict[str, snt.Module]],
        environment_spec: Optional[EnvironmentSpec] = None,
        batch_size: int = 256,
        prefetch_size: int = 4,
        min_replay_size: int = 1000,
        max_replay_size: int = 1000000,
        samples_per_insert=32.0,
        n_step: int = 5,
        num_samples: int = 20,
        additional_discount: float = 0.99,
        target_policy_update_period: int = 100,
        target_critic_update_period: int = 100,
        variable_update_period: int = 1000,
        policy_loss_factory: Optional[Callable[[], snt.Module]] = None,
        max_actor_steps: Optional[int] = None,
        log_every: float = 10.0,
    ):

        if environment_spec is None:
            environment_spec = make_environment_spec(
                _make_environment(
                    domain_name=_DOMAIN.value,
                    task_name=_TASK.value,
                )
            )

        self._network_factory = network_factory
        self._policy_loss_factory = policy_loss_factory
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
        self._log_every = log_every

    def replay(self):
        # Create enough of an error buffer to give a 10% tolerance in rate.
        samples_per_insert_tolerance = 0.1 * self._samples_per_insert
        error_buffer = self._min_replay_size * samples_per_insert_tolerance

        limiter = reverb.rate_limiters.SampleToInsertRatio(
            min_size_to_sample=self._min_replay_size,
            samples_per_insert=self._samples_per_insert,
            error_buffer=error_buffer,
        )
        replay_table = reverb.Table(
            name=adders.DEFAULT_PRIORITY_TABLE,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=self._max_replay_size,
            rate_limiter=limiter,
            signature=adders.NStepTransitionAdder.signature(self._environment_spec),
        )
        return [replay_table]

    def learner(
        self,
        replay: reverb.Client,
        counter: counting.Counter,
    ):
        """The Learning part of the agent."""

        act_spec = self._environment_spec.actions
        obs_spec = self._environment_spec.observations

        # Create online and target networks.
        online_networks = self._network_factory(act_spec)
        target_networks = self._network_factory(act_spec)

        # Make sure observation networks are Sonnet Modules.
        observation_network = online_networks.get("observation", tf.identity)
        observation_network = tf2_utils.to_sonnet_module(observation_network)
        online_networks["observation"] = observation_network
        target_observation_network = target_networks.get("observation", tf.identity)
        target_observation_network = tf2_utils.to_sonnet_module(
            target_observation_network
        )
        target_networks["observation"] = target_observation_network

        # Get embedding spec and create observation network variables.
        emb_spec = tf2_utils.create_variables(observation_network, [obs_spec])

        tf2_utils.create_variables(online_networks["policy"], [emb_spec])
        tf2_utils.create_variables(online_networks["critic"], [emb_spec, act_spec])
        tf2_utils.create_variables(target_networks["observation"], [obs_spec])
        tf2_utils.create_variables(target_networks["policy"], [emb_spec])
        tf2_utils.create_variables(target_networks["critic"], [emb_spec, act_spec])

        # The dataset object to learn from.
        dataset = make_reverb_dataset(server_address=replay.server_address)
        dataset = dataset.batch(self._batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self._prefetch_size)

        # Create a counter and logger for bookkeeping steps and performance.
        counter = counting.Counter(counter, "learner")
        logger = loggers.make_default_logger(
            "learner", time_delta=self._log_every, steps_key="learner_steps"
        )

        # Create policy loss module if a factory is passed.
        if self._policy_loss_factory:
            policy_loss_module = self._policy_loss_factory()
        else:
            policy_loss_module = None

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
            policy_loss_module=policy_loss_module,
            dataset=dataset,
            counter=counter,
            logger=logger,
        )

    def actor(
        self,
        replay: reverb.Client,
        learner: MPOLearner,
        counter: counting.Counter,
    ) -> EnvironmentLoop:
        """The actor process."""

        action_spec = self._environment_spec.actions
        observation_spec = self._environment_spec.observations

        # Create environment and target networks to act with.
        environment = _make_environment(
            domain_name=_DOMAIN.value, task_name=_TASK.value
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
        tf2_utils.create_variables(behavior_network, [observation_spec])

        # Component to add things into replay.
        adder = adders.NStepTransitionAdder(
            client=replay, n_step=self._n_step, discount=self._additional_discount
        )

        # Create logger and counter; actors will not spam bigtable.
        counter = counting.Counter(counter, "actor")
        logger = loggers.make_default_logger(
            "actor",
            save_data=False,
            time_delta=self._log_every,
            steps_key="actor_steps",
        )

        actor = FeedForwardActor(
            policy_network=behavior_network,
            adder=adder,
            learner=learner,
            update_period=self._variable_update_period,
        )

        # Create the run loop and return it.
        return EnvironmentLoop(environment, actor, counter, logger)

    def evaluator(
        self,
        learner: MPOLearner,
        counter: counting.Counter,
    ):
        action_spec = self._environment_spec.actions
        observation_spec = self._environment_spec.observations

        # Create environment and target networks to act with.
        environment = _make_environment(
            domain_name=_DOMAIN.value, task_name=_TASK.value
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
        tf2_utils.create_variables(evaluator_network, [observation_spec])

        # Create logger and counter.
        counter = counting.Counter(counter, "evaluator")
        logger = loggers.make_default_logger(
            "evaluator", time_delta=self._log_every, steps_key="evaluator_steps"
        )

        evaluator = EvaluatorActor(
            policy_network=evaluator_network,
            learner=learner,
            update_period=self._variable_update_period,
        )

        return EnvironmentLoop(environment, evaluator, counter, logger)

    def run(self):
        counter = counting.Counter()
        server = reverb.Server(tables=self.replay())
        client = reverb.Client(f"localhost:{server.port}")
        learner = self.learner(client, counter)
        actor = self.actor(client, learner, counter)
        evaluator = self.evaluator(learner, counter)
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
        online_action_distribution: Union[tfd.MultivariateNormalDiag, tfd.Independent],
        target_action_distribution: Union[tfd.MultivariateNormalDiag, tfd.Independent],
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
        if isinstance(target_action_distribution, tfd.MultivariateNormalDiag):
            target_action_distribution = tfd.Independent(
                tfd.Normal(
                    target_action_distribution.mean(),
                    target_action_distribution.stddev(),
                )
            )
            online_action_distribution = tfd.Independent(
                tfd.Normal(
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
        fixed_stddev_distribution = tfd.Independent(
            tfd.Normal(loc=online_mean, scale=target_scale)
        )
        fixed_mean_distribution = tfd.Independent(
            tfd.Normal(loc=target_mean, scale=online_scale)
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

    def __call__(self, distribution: tfd.Distribution):
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

    def __call__(self, observations: types.Nest) -> tf.Tensor:
        """Forwards the policy network."""
        return self._network(tf2_utils.batch_concat(observations))


class MultivariateNormalDiagHead(snt.Module):
    """Module that produces a multivariate normal distribution using tfd.Independent or tfd.MultivariateNormalDiag."""

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

        dist = tfd.Independent(tfd.Normal(loc=mean, scale=scale))

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
        critic_network: Optional[TensorTransformation] = None,
        observation_network: Optional[TensorTransformation] = None,
        action_network: Optional[TensorTransformation] = None,
    ):
        self._critic_network = critic_network
        self._observation_network = observation_network
        self._action_network = action_network
        super().__init__(name="critic_multiplexer")

    def __call__(
        self, observation: types.NestedTensor, action: types.NestedTensor
    ) -> tf.Tensor:

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
        outputs = tf2_utils.batch_concat([observation, action])

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
) -> Dict[str, types.TensorTransformation]:
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
        "observation": tf2_utils.batch_concat,
    }


def _make_environment(
    domain_name: str = "cartpole",
    task_name: str = "balance",
) -> dm_env.Environment:
    environment = suite.load(domain_name, task_name)
    environment = wrappers.CanonicalSpecWrapper(environment, clip=True)
    environment = wrappers.SinglePrecisionWrapper(environment)
    return environment


def main(_):
    wandb.init(
        entity="adrian-research",
        group=_DOMAIN.value + "_" + _TASK.value,
        project="lp_mpo_single_file",
        config={
            "domain": _DOMAIN.value,
            "task": _TASK.value,
            "max_actor_steps": _MAX_ACTOR_STEPS.value,
        },
    )

    program_builder = MPO(
        network_factory=make_networks,
        target_policy_update_period=25,
        max_actor_steps=_MAX_ACTOR_STEPS.value,
    )

    program_builder.run()


if __name__ == "__main__":
    app.run(main)
