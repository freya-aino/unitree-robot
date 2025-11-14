

# -----------------------------------------------

import time
import random
import warnings
import hydra
import jax
import mlflow
import numpy as np
import torch as T
import torch.nn.functional as F
import tqdm
from omegaconf import DictConfig, OmegaConf
from mujoco.mjx import Data as MjxData
from flatten_dict import flatten
from torch import optim
from sys import platform

from unitree_robot.common.agents import PPOAgent
from unitree_robot.common.environments import MujocoMjxEnv
from unitree_robot.common.datastructure import UnrollData
from unitree_robot.common.experiments import MjxExperiment, Go2WalkingExperiment



# -----------------------------------------------

def reset_global_seed(seed: int):
    # random.seed(seed)
    T.random.manual_seed(seed)
    np.random.seed(seed)


def render(render_fps: int):

    # # print(env.action_space)
    # env.reset()
    # try:
    #     for i in range(1000):
    #         # print(env.get_state_vector())
    #         t_start = time.perf_counter()

    #         action = np.random.uniform(-1, 1, size=env.action_space.shape).astype(
    #             np.float32
    #         )
    #         env.do_simulation(ctrl=action, n_frames=5)

    #         env.render()

    #         elapsed = time.perf_counter() - t_start
    #         target = 1.0 / render_fps
    #         if elapsed < target:
    #             time.sleep(target - elapsed)

    #         print(f"effective fps: {1.0 / (time.perf_counter() - t_start)}")

    # except Exception as e:
    #     raise e
    # finally:
    #     env.close()

    raise NotImplementedError

def unroll_step(
    agent: PPOAgent,
    environment: MujocoMjxEnv,
    mjx_data: MjxData,
    observation: T.Tensor,
    experiment: MjxExperiment,
    device: str
):
    with T.no_grad():
        action, logits = agent.get_action_and_logits(observation)

    next_observation, mjx_data = environment.step(action=F.tanh(action.squeeze()), mjx_data=mjx_data)
    next_observation = next_observation.unsqueeze(1)

    reward = experiment.calculate_reward(mjx_data).unsqueeze(1).unsqueeze(1).to(device=device)

    return next_observation, reward, action, logits



@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    # ----------------------------------------------------------------------------------------------------------------
    print("----------- CONFIGURATIONS -----------")

    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))

    # ----------------------------------------------------------------------------------------------------------------
    print("----------- SYSTEM VARIABLES -----------")

    from os import environ

    # set xla flags for some GPUs
    xla_flags = environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    environ["XLA_FLAGS"] = xla_flags
    environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{cfg.xla_gpu_memory_fraction}"

    print(f"[INFO]: XLA_FLAGS={environ['XLA_FLAGS']}")
    print(f"[INFO]: XLA_PYTHON_CLIENT_PREALLOCATE={environ['XLA_PYTHON_CLIENT_PREALLOCATE']}")
    print(f"[INFO]: XLA_PYTHON_CLIENT_MEM_FRACTION={environ['XLA_PYTHON_CLIENT_MEM_FRACTION']}")

    # ----------------------------------------------------------------------------------------------------------------
    print("----------- SETTING VARIABLES -----------")

    if platform == "win32":
        warnings.warn("Disabled GPU for win32 systems for both pytoch and jax because of jax not having cuda available on windows")
    warnings.warn("only 1 device is used at a time, so 1 cpu or 1 gpu, change if need more")

    device = cfg.device if platform != "win32" else "cpu"
    random_seed = cfg.random_seed
    train_epochs = cfg.training.train_epochs

    # ----------------------------------------------------------------------------------------------------------------
    print("----------- INITIALIZE RANDOM GENERATORS -----------")

    # sanity check seed setting
    reset_global_seed(random_seed)
    t_seed_a = T.randint(low=0, high=2**32, size=[1], dtype=T.uint32).item()
    np_seed_a = np.random.randint(low=0, high=2**32, size=[1], dtype=np.uint32).item()
    reset_global_seed(random_seed)
    t_seed_b = T.randint(low=0, high=2**32, size=[1], dtype=T.uint32).item()
    np_seed_b = np.random.randint(low=0, high=2**32, size=[1], dtype=np.uint32).item()
    print(f"torch seed: {t_seed_a} == {t_seed_b}")
    print(f"numpy seed: {np_seed_a} == {np_seed_b}")

    # set cudnn for reproducibility
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False

    # ----------------------------------------------------------------------------------------------------------------
    print("----------- CONFIGURING JAX -----------")

    # TODO: set jax cuda memory allocation percentage
    jax.config.update("jax_platforms", device)
    jax.config.update("jax_default_device", jax.devices(device)[0])
    # jax.config.update("jax_distributed_debug", True)
    # jax.config.update("jax_log_compiles", False)
    jax.config.update("jax_disable_most_optimizations", True) # reduce non determinism
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_enable_x64", False)
    # jax.config.update("jax_numpy_rank_promotion", "raise")
    jax.config.update("jax_default_matmul_precision", "float32")

    print(f"jax devices: {jax.devices()}")
    print(f"jax backend: {jax.default_backend()}")

    # ----------------------------------------------------------------------------------------------------------------
    print("----------- SETUP ENVIRONMENT, AGENT, EXPERIMENT AND UNROLL DATA -----------")

    unroll_data = UnrollData(
        num_unrolls=cfg.environment.num_parallel_environments,
        unroll_length=cfg.training.unroll_length,
        observation_size=cfg.environment.observation_size,
        action_size=cfg.environment.action_size,
    ).to(device=device, dtype=T.float32)
    agent = PPOAgent(**cfg.agent).to(device=device, dtype=T.float32)
    environment = MujocoMjxEnv(**cfg.environment)
    optimizer = optim.Adam(agent.parameters(), **cfg.optimizer)
    experiment = Go2WalkingExperiment(mjx_model = environment.mjx_model, **cfg.experiment)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **cfg.lr_scheduler)

    # ----------------------------------------------------------------------------------------------------------------
    print("----------- ENVIRONMENT INFO -----------")

    print(f"[INFO]: initial qpos: {environment.mjx_data_initial.qpos}")
    print(f"[INFO]: simulation timestep: mjx = {1 / environment.mjx_model.opt.timestep}; mj = {1 / environment.mj_model.opt.timestep}")
    print(f"[INFO]: ctrl range: {environment.mjx_model.actuator_ctrlrange.copy()}")

    # ----------------------------------------------------------------------------------------------------------------
    print("----------- MLFLOW SETUP -----------")

    # set experiment
    if not mlflow.get_experiment_by_name(cfg.experiment_name):
        mlflow.create_experiment(cfg.experiment_name)
    mlflow.set_experiment(cfg.experiment_name)
    print(f"[INFO]: mlflow experiment set to: '{cfg.experiment_name}'")

    # enable system metric logging (cpu util, etc.)
    mlflow.enable_system_metrics_logging()

    # ----------------------------------------------------------------------------------------------------------------
    print("----------- TRAINING LOOP -----------")

    # reset seed once again since model generation would always change the training seeds
    reset_global_seed(random_seed)

    # reset environment and jit warmup
    jax_seed = T.randint(low=0, high=2**32, size=[1], dtype=T.uint32).item()
    observation, mjx_data = environment.reset(seed=jax_seed)
    observation = observation.unsqueeze(1)
    observation, _, _, _ = unroll_step(agent=agent, environment=environment, mjx_data=mjx_data, observation=observation, experiment=experiment, device=device)

    # run training
    with mlflow.start_run():

        # log params
        mlflow.log_params(flatten(cfg, reducer="dot"))

        # training loop
        for e in range(train_epochs):

            # unroll step
            with T.no_grad():
                mean_reward = 0.0
                for i in tqdm.tqdm(range(cfg.training.unroll_length), desc="Unrolling"):

                    new_observation, reward, action, logits = unroll_step(
                        agent=agent,
                        environment=environment,
                        mjx_data=mjx_data,
                        observation=observation,
                        experiment=experiment,
                        device=device
                    )
                    unroll_data.update(
                        unroll_step=i,
                        observation=observation,
                        action=action,
                        logits=logits,
                        reward=reward,
                    )
                    observation = new_observation

                    # print(f"unroll_data.action:       {unroll_data.actions.mean()}, {unroll_data.actions.min()}, {unroll_data.actions.max()}")
                    # print(f"unroll_data.observations: {unroll_data.observations.mean()}, {unroll_data.observations.min()}, {unroll_data.observations.max()}")
                    # print(f"unroll_data.logits:       {unroll_data.logits.mean()}, {unroll_data.logits.min()}, {unroll_data.logits.max()}")
                    # print(f"unroll_data.rewards:      {unroll_data.rewards.mean()}, {unroll_data.rewards.min()}, {unroll_data.rewards.max()}")

                    mean_reward += reward.mean().detach().cpu().item() / cfg.training.unroll_length

            # train step
            agent.update_normalization(unroll_data.observations)
            agent.train()
            assert unroll_data.validate(), "unroll data is not valid, some NaN or inf values found"
            with T.autograd.detect_anomaly():
                loss, raw_losses = agent.train_step(unroll_data=unroll_data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

            mlflow.log_metric("learning_rate", lr_scheduler.get_last_lr()[0], step=e)

            mlflow.log_metric("reward", mean_reward, step=e)
            mlflow.log_metric("loss", loss, step=e)
            mlflow.log_metrics(raw_losses, step=e)
            mlflow.log_metric("reward_variance", unroll_data.rewards.var(1).mean().detach().item(), step=e)
            mlflow.log_metric("action_variance", unroll_data.actions.var(1).mean().detach().item(), step=e)
            mlflow.log_metric("logit_variance", unroll_data.logits.var(1).mean().detach().item(), step=e)
            mlflow.log_metric("observation_variance", unroll_data.observations.var(1).mean().detach().item(), step=e)
            mlflow.log_metric("action_mean", unroll_data.actions.mean(1).mean().detach().item(), step=e)
            mlflow.log_metric("logit_mean", unroll_data.logits.mean(1).mean().detach().item(), step=e)
            mlflow.log_metric("observation_mean", unroll_data.observations.mean(1).mean().detach().item(), step=e)


            # reset environment
            jax_seed = T.randint(low=0, high=2 ** 32, size=[1], dtype=T.uint32).item()
            observation, mjx_data = environment.reset(seed=jax_seed)
            observation = observation.unsqueeze(1)
            observation, _, _, _ = unroll_step(agent=agent, environment=environment, mjx_data=mjx_data, observation=observation, experiment=experiment, device=device)


if __name__ == "__main__":
    main()

