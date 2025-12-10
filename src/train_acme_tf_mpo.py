"""Launch MPO agent on the control suite via Launchpad."""

import functools
from typing import List, Optional, Dict, Tuple, Union, Sequence, Callable
import time
import copy

import tree
import dm_env
import reverb
import operator

from absl import app
from absl import flags

from dm_control import suite
import threading
import numpy as np
import sonnet as snt

import tensorflow as tf
import trfl

import acme
from acme.utils import counting
from acme.utils import loggers
from acme import datasets
from acme.tf import variable_utils as tf2_variable_utils
from acme.utils import lp_utils
import tensorflow_probability as tfp
from acme import wrappers
from acme import specs
from acme import types
from acme import core
from acme import adders as acme_adders
from acme.tf import utils as tf2_utils
from acme.utils import observers as observers_lib
from acme.utils import signals
from acme.adders.reverb import base
from acme.adders.reverb import utils
from acme.utils import tree_utils


tfd = tfp.distributions
snt_init = snt.initializers


FLAGS = flags.FLAGS
_MAX_ACTOR_STEPS = flags.DEFINE_integer(
    "max_actor_steps",
    None,
    "Number of actor steps to run; defaults to None for an endless loop.",
)
_DOMAIN = flags.DEFINE_string("domain", "cartpole", "Control suite domain name.")
_TASK = flags.DEFINE_string("task", "balance", "Control suite task name.")
_DEFAULT_PRIORITY_TABLE = "priority_table"


class Runner(core.Worker):
    """Wrap an object and expose a run method which checkpoints periodically."""

    def __init__(
        self,
        wrapped,
        key: str = "wrapped",
    ):
        self._wrapped = wrapped

    # Handle preemption signal. Note that this must happen in the main thread.
    def _signal_handler(self):
        pass

    def step(self):
        if isinstance(self._wrapped, core.Learner):
            # Learners have a step() method, so alternate between that and ckpt call.
            self._wrapped.step()

    def run(self):
        """Runs the checkpointer."""
        with signals.runtime_terminator(self._signal_handler):
            while True:
                self.step()

    def get_directory(self) -> str:
        return ""

    def __dir__(self):
        return dir(self._wrapped) + ["get_directory"]

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


class EnvironmentLoop(core.Worker):
    """A simple RL environment loop.

    This takes `Environment` and `Actor` instances and coordinates their
    interaction. Agent is updated if `should_update=True`. This can be used as:

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
        actor: core.Actor,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        should_update: bool = True,
        label: str = "environment_loop",
        observers: Sequence[observers_lib.EnvLoopObserver] = (),
    ):
        # Internalize agent and environment.
        self._environment = environment
        self._actor = actor
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger(
            label, steps_key=self._counter.get_steps_key()
        )
        self._should_update = should_update
        self._observers = observers

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
        for observer in self._observers:
            # Initialize the observer with the current state of the env after reset
            # and the initial timestep.
            observer.observe_first(self._environment, timestep)

        # Run an episode.
        while not timestep.last():
            # Book-keeping.
            episode_steps += 1

            # Generate an action from the agent's policy.
            select_action_start = time.time()
            action = self._actor.select_action(timestep.observation)
            select_action_durations.append(time.time() - select_action_start)

            # Step the environment with the agent's selected action.
            env_step_start = time.time()
            timestep = self._environment.step(action)
            env_step_durations.append(time.time() - env_step_start)

            # Have the agent and observers observe the timestep.
            self._actor.observe(action, next_timestep=timestep)
            for observer in self._observers:
                # One environment step was completed. Observe the current state of the
                # environment, the current timestep and the action.
                observer.observe(self._environment, timestep, action)

            # Give the actor the opportunity to update itself.
            if self._should_update:
                self._actor.update()

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
        for observer in self._observers:
            result.update(observer.get_metrics())
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
        with signals.runtime_terminator():
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


class NStepTransitionAdder(base.ReverbAdder):
    """An N-step transition adder.

    This will buffer a sequence of N timesteps in order to form a single N-step
    transition which is added to reverb for future retrieval.

    For N=1 the data added to replay will be a standard one-step transition which
    takes the form:

          (s_t, a_t, r_t, d_t, s_{t+1}, e_t)

    where:

      s_t = state observation at time t
      a_t = the action taken from s_t
      r_t = reward ensuing from action a_t
      d_t = environment discount ensuing from action a_t. This discount is
          applied to future rewards after r_t.
      e_t [Optional] = extra data that the agent persists in replay.

    For N greater than 1, transitions are of the form:

          (s_t, a_t, R_{t:t+n}, D_{t:t+n}, s_{t+N}, e_t),

    where:

      s_t = State (observation) at time t.
      a_t = Action taken from state s_t.
      g = the additional discount, used by the agent to discount future returns.
      R_{t:t+n} = N-step discounted return, i.e. accumulated over N rewards:
            R_{t:t+n} := r_t + g * d_t * r_{t+1} + ...
                             + g^{n-1} * d_t * ... * d_{t+n-2} * r_{t+n-1}.
      D_{t:t+n}: N-step product of agent discounts g_i and environment
        "discounts" d_i.
            D_{t:t+n} := g^{n-1} * d_{t} * ... * d_{t+n-1},
        For most environments d_i is 1 for all steps except the last,
        i.e. it is the episode termination signal.
      s_{t+n}: The "arrival" state, i.e. the state at time t+n.
      e_t [Optional]: A nested structure of any 'extras' the user wishes to add.

    Notes:
      - At the beginning and end of episodes, shorter transitions are added.
        That is, at the beginning of the episode, it will add:
              (s_0 -> s_1), (s_0 -> s_2), ..., (s_0 -> s_n), (s_1 -> s_{n+1})

        And at the end of the episode, it will add:
              (s_{T-n+1} -> s_T), (s_{T-n+2} -> s_T), ... (s_{T-1} -> s_T).
      - We add the *first* `extra` of each transition, not the *last*, i.e.
          if extras are provided, we get e_t, not e_{t+n}.
    """

    def __init__(
        self,
        client: reverb.Client,
        n_step: int,
        discount: float,
        *,
        priority_fns: Optional[base.PriorityFnMapping] = None,
        max_in_flight_items: int = 5,
    ):
        """Creates an N-step transition adder.

        Args:
          client: A `reverb.Client` to send the data to replay through.
          n_step: The "N" in N-step transition. See the class docstring for the
            precise definition of what an N-step transition is. `n_step` must be at
            least 1, in which case we use the standard one-step transition, i.e.
            (s_t, a_t, r_t, d_t, s_t+1, e_t).
          discount: Discount factor to apply. This corresponds to the agent's
            discount in the class docstring.
          priority_fns: See docstring for BaseAdder.
          max_in_flight_items: The maximum number of items allowed to be "in flight"
            at the same time. See `block_until_num_items` in
            `reverb.TrajectoryWriter.flush` for more info.

        Raises:
          ValueError: If n_step is less than 1.
        """
        # Makes the additional discount a float32, which means that it will be
        # upcast if rewards/discounts are float64 and left alone otherwise.
        self.n_step = n_step
        self._discount = tree.map_structure(np.float32, discount)
        self._first_idx = 0
        self._last_idx = 0

        super().__init__(
            client=client,
            max_sequence_length=n_step + 1,
            priority_fns=priority_fns,
            max_in_flight_items=max_in_flight_items,
        )

    def add(self, *args, **kwargs):
        # Increment the indices for the start and end of the window for computing
        # n-step returns.
        if self._writer.episode_steps >= self.n_step:
            self._first_idx += 1
        self._last_idx += 1

        super().add(*args, **kwargs)

    def reset(
        self,
    ):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
        super().reset()
        self._first_idx = 0
        self._last_idx = 0

    @property
    def _n_step(self) -> int:
        """Effective n-step, which may vary at starts and ends of episodes."""
        return self._last_idx - self._first_idx

    def _write(self):
        # Convenient getters for use in tree operations.
        get_first = lambda x: x[self._first_idx]
        get_last = lambda x: x[self._last_idx]
        # Note: this getter is meant to be used on a TrajectoryWriter.history to
        # obtain its numpy values.
        get_all_np = lambda x: x[self._first_idx : self._last_idx].numpy()

        # Get the state, action, next_state, as well as possibly extras for the
        # transition that is about to be written.
        history = self._writer.history
        s, a = tree.map_structure(
            get_first, (history["observation"], history["action"])
        )
        s_ = tree.map_structure(get_last, history["observation"])

        # Maybe get extras to add to the transition later.
        if "extras" in history:
            extras = tree.map_structure(get_first, history["extras"])

        # Note: at the beginning of an episode we will add the initial N-1
        # transitions (of size 1, 2, ...) and at the end of an episode (when
        # called from write_last) we will write the final transitions of size (N,
        # N-1, ...). See the Note in the docstring.
        # Get numpy view of the steps to be fed into the priority functions.
        reward, discount = tree.map_structure(
            get_all_np, (history["reward"], history["discount"])
        )

        # Compute discounted return and geometric discount over n steps.
        n_step_return, total_discount = self._compute_cumulative_quantities(
            reward, discount
        )

        # Append the computed n-step return and total discount.
        # Note: if this call to _write() is within a call to _write_last(), then
        # this is the only data being appended and so it is not a partial append.
        self._writer.append(
            dict(n_step_return=n_step_return, total_discount=total_discount),
            partial_step=self._writer.episode_steps <= self._last_idx,
        )
        # This should be done immediately after self._writer.append so the history
        # includes the recently appended data.
        history = self._writer.history

        # Form the n-step transition by using the following:
        # the first observation and action in the buffer, along with the cumulative
        # reward and discount computed above.
        n_step_return, total_discount = tree.map_structure(
            lambda x: x[-1], (history["n_step_return"], history["total_discount"])
        )
        transition = types.Transition(
            observation=s,
            action=a,
            reward=n_step_return,
            discount=total_discount,
            next_observation=s_,
            extras=(extras if "extras" in history else ()),
        )

        # Calculate the priority for this transition.
        table_priorities = utils.calculate_priorities(self._priority_fns, transition)

        # Insert the transition into replay along with its priority.
        for table, priority in table_priorities.items():
            self._writer.create_item(
                table=table, priority=priority, trajectory=transition
            )
            self._writer.flush(self._max_in_flight_items)

    def _write_last(self):
        # Write the remaining shorter transitions by alternating writing and
        # incrementingfirst_idx. Note that last_idx will no longer be incremented
        # once we're in this method's scope.
        self._first_idx += 1
        while self._first_idx < self._last_idx:
            self._write()
            self._first_idx += 1

    def _compute_cumulative_quantities(
        self, rewards: types.NestedArray, discounts: types.NestedArray
    ) -> Tuple[types.NestedArray, types.NestedArray]:

        # Give the same tree structure to the n-step return accumulator,
        # n-step discount accumulator, and self.discount, so that they can be
        # iterated in parallel using tree.map_structure.
        rewards, discounts, self_discount = tree_utils.broadcast_structures(
            rewards, discounts, self._discount
        )
        flat_rewards = tree.flatten(rewards)
        flat_discounts = tree.flatten(discounts)
        flat_self_discount = tree.flatten(self_discount)

        # Copy total_discount as it is otherwise read-only.
        total_discount = [np.copy(a[0]) for a in flat_discounts]

        # Broadcast n_step_return to have the broadcasted shape of
        # reward * discount.
        n_step_return = [
            np.copy(np.broadcast_to(r[0], np.broadcast(r[0], d).shape))
            for r, d in zip(flat_rewards, total_discount)
        ]

        # NOTE: total_discount will have one less self_discount applied to it than
        # the value of self._n_step. This is so that when the learner/update uses
        # an additional discount we don't apply it twice. Inside the following loop
        # we will apply this right before summing up the n_step_return.
        for i in range(1, self._n_step):
            for nsr, td, r, d, sd in zip(
                n_step_return,
                total_discount,
                flat_rewards,
                flat_discounts,
                flat_self_discount,
            ):
                # Equivalent to: `total_discount *= self._discount`.
                td *= sd
                # Equivalent to: `n_step_return += reward[i] * total_discount`.
                nsr += r[i] * td
                # Equivalent to: `total_discount *= discount[i]`.
                td *= d[i]

        n_step_return = tree.unflatten_as(rewards, n_step_return)
        total_discount = tree.unflatten_as(rewards, total_discount)
        return n_step_return, total_discount

    # TODO(bshahr): make this into a standalone method. Class methods should be
    # used as alternative constructors or when modifying some global state,
    # neither of which is done here.
    @classmethod
    def signature(
        cls, environment_spec: specs.EnvironmentSpec, extras_spec: types.NestedSpec = ()
    ):

        # This function currently assumes that self._discount is a scalar.
        # If it ever becomes a nested structure and/or a np.ndarray, this method
        # will need to know its structure / shape. This is because the signature
        # discount shape is the environment's discount shape and this adder's
        # discount shape broadcasted together. Also, the reward shape is this
        # signature discount shape broadcasted together with the environment
        # reward shape. As long as self._discount is a scalar, it will not affect
        # either the signature discount shape nor the signature reward shape, so we
        # can ignore it.

        rewards_spec, step_discounts_spec = tree_utils.broadcast_structures(
            environment_spec.rewards, environment_spec.discounts
        )
        rewards_spec = tree.map_structure(
            _broadcast_specs, rewards_spec, step_discounts_spec
        )
        step_discounts_spec = tree.map_structure(copy.deepcopy, step_discounts_spec)

        transition_spec = types.Transition(
            environment_spec.observations,
            environment_spec.actions,
            rewards_spec,
            step_discounts_spec,
            environment_spec.observations,  # next_observation
            extras_spec,
        )

        return tree.map_structure_with_path(
            base.spec_like_to_tensor_spec, transition_spec
        )


def _broadcast_specs(*args: specs.Array) -> specs.Array:
    """Like np.broadcast, but for specs.Array.

    Args:
      *args: one or more specs.Array instances.

    Returns:
      A specs.Array with the broadcasted shape and dtype of the specs in *args.
    """
    bc_info = np.broadcast(*tuple(a.generate_value() for a in args))
    dtype = np.result_type(*tuple(a.dtype for a in args))
    return specs.Array(shape=bc_info.shape, dtype=dtype)


class FeedForwardActor(core.Actor):
    """A feed-forward actor.

    An actor based on a feed-forward policy which takes non-batched observations
    and outputs non-batched actions. It also allows adding experiences to replay
    and updating the weights from the policy on the learner.
    """

    def __init__(
        self,
        policy_network: snt.Module,
        adder: Optional[acme_adders.Adder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
    ):
        """Initializes the actor.

        Args:
          policy_network: the policy to run.
          adder: the adder object to which allows to add experiences to a
            dataset/replay buffer.
          variable_client: object which allows to copy weights from the learner copy
            of the policy to the actor copy (in case they are separate).
        """

        # Store these for later use.
        self._adder = adder
        self._variable_client = variable_client
        self._policy_network = policy_network

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
        if self._adder:
            self._adder.add_first(timestep)

    def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
        if self._adder:
            self._adder.add(action, next_timestep)

    def update(self, wait: bool = False):
        if self._variable_client:
            self._variable_client.update(wait)


class StochasticMeanHead(snt.Module):
    """Simple sonnet module to produce the mean of a tfp.Distribution."""

    def __call__(self, distribution: tfd.Distribution):
        return distribution.mean()


class AcmeMPO:
    """Program definition for MPO."""

    def __init__(
        self,
        environment_factory: Callable[[], dm_env.Environment],
        network_factory: Callable[[specs.BoundedArray], Dict[str, snt.Module]],
        environment_spec: Optional[specs.EnvironmentSpec] = None,
        batch_size: int = 256,
        prefetch_size: int = 4,
        min_replay_size: int = 1000,
        max_replay_size: int = 1000000,
        samples_per_insert: Optional[float] = 32.0,
        n_step: int = 5,
        num_samples: int = 20,
        additional_discount: float = 0.99,
        target_policy_update_period: int = 100,
        target_critic_update_period: int = 100,
        variable_update_period: int = 1000,
        max_actor_steps: Optional[int] = None,
        log_every: float = 10.0,
    ):

        if environment_spec is None:
            environment_spec = specs.make_environment_spec(environment_factory())

        self._environment_factory = environment_factory
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
        self._log_every = log_every

    def replay(self):
        """The replay storage."""
        if self._samples_per_insert is not None:
            # Create enough of an error buffer to give a 10% tolerance in rate.
            samples_per_insert_tolerance = 0.1 * self._samples_per_insert
            error_buffer = self._min_replay_size * samples_per_insert_tolerance

            limiter = reverb.rate_limiters.SampleToInsertRatio(
                min_size_to_sample=self._min_replay_size,
                samples_per_insert=self._samples_per_insert,
                error_buffer=error_buffer,
            )
        else:
            limiter = reverb.rate_limiters.MinSize(
                min_size_to_sample=self._min_replay_size
            )
        replay_table = reverb.Table(
            name=_DEFAULT_PRIORITY_TABLE,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=self._max_replay_size,
            rate_limiter=limiter,
            signature=NStepTransitionAdder.signature(self._environment_spec),
        )
        return [replay_table]

    def counter(self):
        return Runner(counting.Counter())

    def coordinator(self, counter: counting.Counter, max_actor_steps: int):
        return lp_utils.StepsLimiter(counter, max_actor_steps)

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
        dataset = datasets.make_reverb_dataset(server_address=replay.server_address)
        dataset = dataset.batch(self._batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self._prefetch_size)

        # Create a counter and logger for bookkeeping steps and performance.
        counter = counting.Counter(counter, "learner")
        logger = loggers.make_default_logger(
            "learner", time_delta=self._log_every, steps_key="learner_steps"
        )

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
            dataset=dataset,
            counter=counter,
            logger=logger,
        )

    def actor(
        self,
        replay: reverb.Client,
        variable_source: acme.VariableSource,
        counter: counting.Counter,
    ) -> EnvironmentLoop:
        """The actor process."""

        action_spec = self._environment_spec.actions
        observation_spec = self._environment_spec.observations

        # Create environment and target networks to act with.
        environment = self._environment_factory()
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
        policy_variables = {"policy": behavior_network.variables}

        # Create the variable client responsible for keeping the actor up-to-date.
        variable_client = tf2_variable_utils.VariableClient(
            variable_source,
            policy_variables,
            update_period=self._variable_update_period,
        )

        # Make sure not to use a random policy after checkpoint restoration by
        # assigning variables before running the environment loop.
        variable_client.update_and_wait()

        adder = NStepTransitionAdder(
            client=replay, n_step=self._n_step, discount=self._additional_discount
        )

        actor = FeedForwardActor(
            policy_network=behavior_network,
            adder=adder,
            variable_client=variable_client,
        )

        counter = counting.Counter(counter, "actor")
        logger = loggers.make_default_logger(
            "actor",
            save_data=False,
            time_delta=self._log_every,
            steps_key="actor_steps",
        )

        return EnvironmentLoop(environment, actor, counter, logger)

    def evaluator(
        self,
        variable_source: acme.VariableSource,
        counter: counting.Counter,
    ):
        """The evaluation process."""

        action_spec = self._environment_spec.actions
        observation_spec = self._environment_spec.observations

        # Create environment and target networks to act with.
        environment = self._environment_factory()
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
        policy_variables = {"policy": evaluator_network.variables}

        # Create the variable client responsible for keeping the actor up-to-date.
        variable_client = tf2_variable_utils.VariableClient(
            variable_source,
            policy_variables,
            update_period=self._variable_update_period,
        )

        # Make sure not to evaluate a random actor by assigning variables before
        # running the environment loop.
        variable_client.update_and_wait()

        # Create the agent.
        evaluator = FeedForwardActor(
            policy_network=evaluator_network, variable_client=variable_client
        )

        # Create logger and counter.
        counter = counting.Counter(counter, "evaluator")
        logger = loggers.make_default_logger(
            "evaluator", time_delta=self._log_every, steps_key="evaluator_steps"
        )

        # Create the run loop and return it.
        return EnvironmentLoop(environment, evaluator, counter, logger)

    def build(self, name="mpo"):
        raise NotImplementedError(
            "build() (Launchpad) has been removed; use run_local()"
        )

    def run_local(
        self,
        max_actor_steps: Optional[int] = None,
        run_learner_in_thread: bool = True,
        run_evaluator: bool = False,
        learner_steps_per_episode: int = 1,
    ):
        """Run the agent topology locally without Launchpad.

        This starts an in-process Reverb server, creates a client, instantiates
        the learner/actor/evaluator loops and runs the actor loop in the main
        thread. Optionally runs the learner (and evaluator) in background
        daemon threads.
        """
        # Create tables and start an in-process Reverb server.
        tables = self.replay()
        server = reverb.Server(tables)
        client = reverb.Client(server_address=server.server_address)

        # Shared counter for bookkeeping.
        counter = counting.Counter()

        # Instantiate learner, evaluator and actor loops.
        learner = self.learner(client, counter)
        evaluator_loop = None
        if run_evaluator:
            evaluator_loop = self.evaluator(learner, counter)

        actor_loop = self.actor(client, learner, counter)

        # If requested, run the learner in a background thread.
        if run_learner_in_thread:

            def _learner_loop():
                try:
                    while True:
                        learner.step()
                except Exception:
                    return

            t = threading.Thread(target=_learner_loop, daemon=True)
            t.start()

            # Optionally run evaluator in a background thread.
            if run_evaluator:

                def _eval_loop():
                    try:
                        evaluator_loop.run()
                    except Exception:
                        return

                t_eval = threading.Thread(target=_eval_loop, daemon=True)
                t_eval.start()

            # Run actor loop in the foreground (blocks until completion).
            if max_actor_steps:
                actor_loop.run(num_steps=max_actor_steps)
            else:
                actor_loop.run()
        else:
            # Single-threaded interleaved mode: run one episode at a time and
            # execute a configurable number of learner steps between episodes.
            num_episodes = None
            num_steps = max_actor_steps

            def _should_terminate(episode_count: int, step_count: int) -> bool:
                return (num_episodes is not None and episode_count >= num_episodes) or (
                    num_steps is not None and step_count >= num_steps
                )

            episode_count = 0
            step_count = 0
            with signals.runtime_terminator():
                while not _should_terminate(episode_count, step_count):
                    episode_result = actor_loop.run_episode()
                    # Add episode duration and other bookkeeping similar to
                    # EnvironmentLoop.run
                    episode_count += 1
                    step_count += int(episode_result.get("episode_length", 0))
                    # Write actor logs for the episode.
                    try:
                        actor_loop._logger.write(
                            {**episode_result, **{"episode_duration": 0}}
                        )
                    except Exception:
                        pass

                    # Run learner steps after the episode.
                    for _ in range(max(1, learner_steps_per_episode)):
                        try:
                            learner.step()
                        except Exception:
                            # If learner fails, continue with acting.
                            pass

        # Cleanly stop server when done.
        try:
            server.stop()
        except Exception:
            pass


"""Implements the MPO losses.

The MPO loss is implemented as a Sonnet module rather than a function so that it
can hold its own dual variables, as instances of `tf.Variable`, which it creates
the first time the module is called.

Tensor shapes are annotated, where helpful, as follow:
  B: batch size,
  N: number of sampled actions, see MPO paper for more details,
  D: dimensionality of the action space.
"""


_MPO_FLOAT_EPSILON = 1e-8


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


class MPOLearner(acme.Learner):
    """MPO learner."""

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
        observation_network: types.TensorTransformation = tf.identity,
        target_observation_network: types.TensorTransformation = tf.identity,
        policy_optimizer: Optional[snt.Optimizer] = None,
        critic_optimizer: Optional[snt.Optimizer] = None,
        dual_optimizer: Optional[snt.Optimizer] = None,
        clipping: bool = True,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
    ):

        self._counter = counter or counting.Counter()
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

        self._policy_loss_module = MPOLoss(
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
        self._logger.write(fetches)

    def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
        return [tf2_utils.to_numpy(self._variables[name]) for name in names]


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
        fixed_scale: bool = False,
        use_tfd_independent: bool = False,
        w_init: snt_init.Initializer = tf.initializers.VarianceScaling(1e-4),
        b_init: snt_init.Initializer = tf.initializers.Zeros(),
    ):
        """Initialization.

        Args:
          num_dimensions: Number of dimensions of MVN distribution.
          init_scale: Initial standard deviation.
          min_scale: Minimum standard deviation.
          tanh_mean: Whether to transform the mean (via tanh) before passing it to
            the distribution.
          fixed_scale: Whether to use a fixed variance.
          use_tfd_independent: Whether to use tfd.Independent or
            tfd.MultivariateNormalDiag class
          w_init: Initialization for linear layer weights.
          b_init: Initialization for linear layer biases.
        """
        super().__init__(name="MultivariateNormalDiagHead")
        self._init_scale = init_scale
        self._min_scale = min_scale
        self._tanh_mean = tanh_mean
        self._mean_layer = snt.Linear(num_dimensions, w_init=w_init, b_init=b_init)
        self._fixed_scale = fixed_scale

        if not fixed_scale:
            self._scale_layer = snt.Linear(num_dimensions, w_init=w_init, b_init=b_init)
        self._use_tfd_independent = use_tfd_independent

    def __call__(self, inputs: tf.Tensor) -> tfd.Distribution:
        zero = tf.constant(0, dtype=inputs.dtype)
        mean = self._mean_layer(inputs)

        if self._fixed_scale:
            scale = tf.ones_like(mean) * self._init_scale
        else:
            scale = tf.nn.softplus(self._scale_layer(inputs))
            scale *= self._init_scale / tf.nn.softplus(zero)
            scale += self._min_scale

        # Maybe transform the mean.
        if self._tanh_mean:
            mean = tf.tanh(mean)

        if self._use_tfd_independent:
            dist = tfd.Independent(tfd.Normal(loc=mean, scale=scale))
        else:
            dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=scale)

        return dist


TensorTransformation = Union[snt.Module, Callable[[types.NestedTensor], tf.Tensor]]


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
            MultivariateNormalDiagHead(
                num_dimensions, init_scale=0.7, use_tfd_independent=True
            ),
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
    evaluation: bool = False,
    domain_name: str = "cartpole",
    task_name: str = "balance",
    num_action_repeats: Optional[int] = None,
) -> dm_env.Environment:
    """Implements a control suite environment factory."""
    # Load raw control suite environment.
    environment = suite.load(domain_name, task_name)

    environment = wrappers.CanonicalSpecWrapper(environment, clip=True)

    if num_action_repeats:
        environment = wrappers.ActionRepeatWrapper(
            environment, num_repeats=num_action_repeats
        )
    environment = wrappers.SinglePrecisionWrapper(environment)

    return environment


def main(_):
    make_environment = functools.partial(
        _make_environment, domain_name=_DOMAIN.value, task_name=_TASK.value
    )

    program_builder = AcmeMPO(
        make_environment,
        make_networks,
        target_policy_update_period=25,
        max_actor_steps=_MAX_ACTOR_STEPS.value,
    )

    # Run locally without Launchpad using an in-process Reverb server.
    program_builder.run_local(max_actor_steps=_MAX_ACTOR_STEPS.value)


if __name__ == "__main__":
    app.run(main)
