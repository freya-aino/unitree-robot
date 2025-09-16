import functools
import brax

from brax import envs
from brax.training.agents.ppo import train as ppo

import jax


env = envs.get_environment(
    env_name="inverted_double_pendulum",
    backend="positional"
)
state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed = 0))

train_fn = functools.partial(
    ppo.train,
    num_timesteps = 20_000,
    num_evals = 16,
    num_envs = 16,
    num_minibatches = 32,
    batch_size = 4,
    reward_scaling = 10,
    episode_length = 1000,
    normalize_observations = True,
    action_repeat = 1,
    unroll_length = 5,
    num_updates_per_batch = 4,
    discounting = 0.97,
    learning_rate = 3e-4,
    entropy_cost = 1e-2,
    seed = 0
)
make_inference_fn, parameters, _ = train_fn(
    environment = env,
    progress_fn = lambda num_steps, metrics: print(metrics["eval/episode_reward"])
)
