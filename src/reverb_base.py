import abc
import time
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    NamedTuple,
    Optional,
    Sized,
    Union,
    Tuple,
)

from absl import logging
import dm_env
import numpy as np
import reverb
import tensorflow as tf
import tree
from dm_env import specs

Array = specs.Array
BoundedArray = specs.BoundedArray
DiscreteArray = specs.DiscreteArray


DEFAULT_PRIORITY_TABLE = "priority_table"
_MIN_WRITER_LIFESPAN_SECONDS = 60
StartOfEpisodeType = Union[bool, Any, tf.Tensor, tf.TensorSpec, Tuple[()]]


class Step(NamedTuple):
    """Step class used internally for reverb adders."""

    observation: Any
    action: Any
    reward: Any
    discount: Any
    start_of_episode: StartOfEpisodeType
    extras: Any = ()


Trajectory = Step


class PriorityFnInput(NamedTuple):
    """The input to a priority function consisting of stacked steps."""

    observations: Any
    actions: Any
    rewards: Any
    discounts: Any
    start_of_episode: Any
    extras: Any


# Define the type of a priority function and the mapping from table to function.
PriorityFn = Callable[["PriorityFnInput"], float]
PriorityFnMapping = Mapping[str, Optional[PriorityFn]]


def spec_like_to_tensor_spec(paths: Iterable[str], spec: specs.Array):
    return tf.TensorSpec.from_spec(spec, name="/".join(str(p) for p in paths))


class Adder(abc.ABC):
    """The Adder interface.

    An adder packs together data to send to the replay buffer, and potentially
    performs some reduction/transformation to this data in the process.

    All adders will use this API. Below is an illustrative example of how they
    are intended to be used in a typical RL run-loop. We assume that the
    environment conforms to the dm_env environment API.

    ```python
    # Reset the environment and add the first observation.
    timestep = env.reset()
    adder.add_first(timestep.observation)

    while not timestep.last():
      # Generate an action from the policy and step the environment.
      action = my_policy(timestep)
      timestep = env.step(action)

      # Add the action and the resulting timestep.
      adder.add(action, next_timestep=timestep)
    ```

    Note that for all adders, the `add()` method expects an action taken and the
    *resulting* timestep observed after taking this action. Note that this
    timestep is named `next_timestep` precisely to emphasize this point.
    """

    @abc.abstractmethod
    def add_first(self, timestep: dm_env.TimeStep):
        """Defines the interface for an adder's `add_first` method.

        We expect this to be called at the beginning of each episode and it will
        start a trajectory to be added to replay with an initial observation.

        Args:
          timestep: a dm_env TimeStep corresponding to the first step.
        """

    @abc.abstractmethod
    def add(
        self,
        action: Any,
        next_timestep: dm_env.TimeStep,
        extras: Any = (),
    ):
        """Defines the adder `add` interface.

        Args:
          action: A possibly nested structure corresponding to a_t.
          next_timestep: A dm_env Timestep object corresponding to the resulting
            data obtained by taking the given action.
          extras: A possibly nested structure of extra data to add to replay.
        """

    @abc.abstractmethod
    def reset(self):
        """Resets the adder's buffer."""


class ReverbAdder(Adder):
    """Base class for Reverb adders."""

    def __init__(
        self,
        client: reverb.Client,
        max_sequence_length: int,
        max_in_flight_items: int,
        delta_encoded: bool = False,
        priority_fns: Optional[PriorityFnMapping] = None,
        validate_items: bool = True,
    ):
        """Initialize a ReverbAdder instance.

        Args:
          client: A client to the Reverb backend.
          max_sequence_length: The maximum length of sequences (corresponding to the
            number of observations) that can be added to replay.
          max_in_flight_items: The maximum number of items allowed to be "in flight"
            at the same time. See `block_until_num_items` in
            `reverb.TrajectoryWriter.flush` for more info.
          delta_encoded: If `True` (False by default) enables delta encoding, see
            `Client` for more information.
          priority_fns: A mapping from table names to priority functions; if
            omitted, all transitions/steps/sequences are given uniform priorities
            (1.0) and placed in DEFAULT_PRIORITY_TABLE.
          validate_items: Whether to validate items against the table signature
            before they are sent to the server. This requires table signature to be
            fetched from the server and cached locally.
        """
        if priority_fns:
            priority_fns = dict(priority_fns)
        else:
            priority_fns = {DEFAULT_PRIORITY_TABLE: None}

        self._client = client
        self._priority_fns = priority_fns
        self._max_sequence_length = max_sequence_length
        self._delta_encoded = delta_encoded
        # TODO(b/206629159): Remove this.
        self._max_in_flight_items = max_in_flight_items
        self._add_first_called = False

        # This is exposed as the _writer property in such a way that it will create
        # a new writer automatically whenever the internal __writer is None. Users
        # should ONLY ever interact with self._writer.
        self.__writer = None
        # Every time a new writer is created, it must fetch the signature from the
        # Reverb server. If this is set too low it can crash the adders in a
        # distributed setup where the replay may take a while to spin up.
        self._validate_items = validate_items

    def __del__(self):
        if self.__writer is not None:
            timeout_ms = 10_000
            # Try flush all appended data before closing to avoid loss of experience.
            try:
                self.__writer.flush(0, timeout_ms=timeout_ms)
            except reverb.DeadlineExceededError as e:
                logging.error(
                    "Timeout (%d ms) exceeded when flushing the writer before "
                    "deleting it. Caught Reverb exception: %s",
                    timeout_ms,
                    str(e),
                )
            self.__writer.close()
            self.__writer = None

    @property
    def _writer(self) -> reverb.TrajectoryWriter:
        if self.__writer is None:
            self.__writer = self._client.trajectory_writer(
                num_keep_alive_refs=self._max_sequence_length,
                validate_items=self._validate_items,
            )
            self._writer_created_timestamp = time.time()
        return self.__writer

    def add_priority_table(self, table_name: str, priority_fn: Optional[PriorityFn]):
        if table_name in self._priority_fns:
            raise ValueError(
                f"A priority function already exists for {table_name}. "
                f'Existing tables: {", ".join(self._priority_fns.keys())}.'
            )
        self._priority_fns[table_name] = priority_fn

    def reset(self, timeout_ms: Optional[int] = None):
        """Resets the adder's buffer."""
        if self.__writer:
            # Flush all appended data and clear the buffers.
            self.__writer.end_episode(clear_buffers=True, timeout_ms=timeout_ms)

            # Create a new writer unless the current one is too young.
            # This is to reduce the relative overhead of creating a new Reverb writer.
            if (
                time.time() - self._writer_created_timestamp
                > _MIN_WRITER_LIFESPAN_SECONDS
            ):
                self.__writer = None
        self._add_first_called = False

    def add_first(self, timestep: dm_env.TimeStep):
        """Record the first observation of a trajectory."""
        if not timestep.first():
            raise ValueError(
                "adder.add_first with an initial timestep (i.e. one for "
                "which timestep.first() is True"
            )

        # Record the next observation but leave the history buffer row open by
        # passing `partial_step=True`.
        self._writer.append(
            dict(observation=timestep.observation, start_of_episode=timestep.first()),
            partial_step=True,
        )
        self._add_first_called = True

    def add(self, action: Any, next_timestep: dm_env.TimeStep, extras: Any = ()):
        """Record an action and the following timestep."""

        if not self._add_first_called:
            raise ValueError("adder.add_first must be called before adder.add.")

        # Add the timestep to the buffer.
        has_extras = (
            len(extras) > 0
            if isinstance(extras, Sized)  # pylint: disable=g-explicit-length-test
            else extras is not None
        )
        current_step = dict(
            # Observation was passed at the previous add call.
            action=action,
            reward=next_timestep.reward,
            discount=next_timestep.discount,
            # Start of episode indicator was passed at the previous add call.
            **({"extras": extras} if has_extras else {}),
        )
        self._writer.append(current_step)

        # Record the next observation and write.
        self._writer.append(
            dict(
                observation=next_timestep.observation,
                start_of_episode=next_timestep.first(),
            ),
            partial_step=True,
        )
        self._write()

        if next_timestep.last():
            # Complete the row by appending zeros to remaining open fields.
            # TODO(b/183945808): remove this when fields are no longer expected to be
            # of equal length on the learner side.
            dummy_step = tree.map_structure(np.zeros_like, current_step)
            self._writer.append(dummy_step)
            self._write_last()
            self.reset()

    @classmethod
    def signature(
        cls, environment_spec: specs.EnvironmentSpec, extras_spec: types.NestedSpec = ()
    ):
        """This is a helper method for generating signatures for Reverb tables.

        Signatures are useful for validating data types and shapes, see Reverb's
        documentation for details on how they are used.

        Args:
          environment_spec: A `specs.EnvironmentSpec` whose fields are nested
            structures with leaf nodes that have `.shape` and `.dtype` attributes.
            This should come from the environment that will be used to generate
            the data inserted into the Reverb table.
          extras_spec: A nested structure with leaf nodes that have `.shape` and
            `.dtype` attributes. The structure (and shapes/dtypes) of this must
            be the same as the `extras` passed into `ReverbAdder.add`.

        Returns:
          A `Step` whose leaf nodes are `tf.TensorSpec` objects.
        """
        spec_step = Step(
            observation=environment_spec.observations,
            action=environment_spec.actions,
            reward=environment_spec.rewards,
            discount=environment_spec.discounts,
            start_of_episode=specs.Array(shape=(), dtype=bool),
            extras=extras_spec,
        )
        return tree.map_structure_with_path(spec_like_to_tensor_spec, spec_step)

    @abc.abstractmethod
    def _write(self):
        """Write data to replay from the buffer."""

    @abc.abstractmethod
    def _write_last(self):
        """Write data to replay from the buffer."""
