import time
import warnings
import hydra
import jax
import mlflow
import torch as T
import tqdm
# from flatten_dict import flatten
from omegaconf import DictConfig, OmegaConf
from torch import optim
from os import environ
from sys import platform
from mujoco.mjx import Data as MjxData

from unitree_robot.common.agents import PPOAgent
from unitree_robot.training.environments import MujocoMjxEnv
from unitree_robot.common.datastructure import UnrollData
from unitree_robot.training.experiments import Experiment, TestExperiment

# -----------------------------------------------

# set xla flags for some GPUs
xla_flags = environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
environ["XLA_FLAGS"] = xla_flags

# -----------------------------------------------

def unroll_step(
    agent: PPOAgent,
    environment: MujocoMjxEnv,
    mjx_data: MjxData,
    observation: T.Tensor,
    experiment: Experiment,
    device: str
):
    with T.no_grad():
        action, logits = agent.get_action_and_logits(observation)

    next_observation, mjx_data = environment.step(action=action.squeeze(), mjx_data=mjx_data)
    next_observation = next_observation.unsqueeze(1)

    reward = experiment(mjx_data).unsqueeze(1).unsqueeze(1).to(device=device)

    return next_observation, reward, action, logits



@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    TODO:

        please check out how 'model.opt.timestep' should be set in mujoco
            model.opt.timestep
            Simulation step size (dt)
            Convert per‑step quantities to rates (e.g., power = work / dt).

    """
    print("----------- CONFIGURATIONS -----------")
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))
    print("----------- CONFIGURATIONS -----------")

    warnings.warn("Disabled GPU for win32 systems for both pytoch and jax because of jax not having cuda avialable on windows")
    warnings.warn("only 1 device is used at a time, so 1 cpu or 1 gpu, change if need more")

    # -- set variables

    minibatch_size = cfg["training"]["minibatch_size"]
    device = cfg["device"] if platform != "win32" else "cpu"
    random_seed = cfg["random_seed"]

    # -- jax config
    jax.config.update("jax_platforms", device)
    jax.config.update("jax_default_device", jax.devices(device)[0])
    # jax.config.update("jax_distributed_debug", True)
    # jax.config.update("jax_log_compiles", False)
    jax.config.update("jax_logging_level", "ERROR")

    print(f"jax devices: {jax.devices()}")
    print(f"jax backend: {jax.default_backend()}")

    # -- setup environment and agent

    environment = MujocoMjxEnv(**cfg.environment)
    agent = PPOAgent(**cfg.agent).to(device=device)
    optimizer = optim.AdamW(agent.parameters(), **cfg.optimizer)
    experiment = TestExperiment()

    # -- printin some information about the environment

    print(f"[INFO]: simulation fps: mjx = {1 / environment.mjx_model.opt.timestep}; mj = {1 / environment.mj_model.opt.timestep}")
    print(f"[INFO]: ctrl range: {environment.mjx_model.actuator_ctrlrange.copy()}")

    # -- unroll and training

    unroll_data = UnrollData(
        num_unrolls=cfg.environment.num_parallel_environments,
        unroll_length=cfg.training.unroll_length,
        observation_size=cfg.environment.observation_size,
        action_size=cfg.environment.action_size,
    ).to(device=device)


    # reset environment and jit warmup
    observation, mjx_data = environment.reset(seed=random_seed)
    observation = observation.unsqueeze(1)
    observation, _, _, _ = unroll_step(agent=agent, environment=environment, mjx_data=mjx_data, observation=observation, experiment=experiment, device=device)

    # training loop
    for e in range(cfg.training.train_epochs):
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

            mean_reward += reward.detach().cpu().item() / cfg.training.unroll_length

        loss, raw_losses = agent.train_step(unroll_data=unroll_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"loss:   {loss}")
        print(f"reward: {mean_reward}")


    # # create and select experiment
    # if not mlflow.get_experiment_by_name(cfg["experiment_name"]):
    #     mlflow.create_experiment(cfg["experiment_name"])
    # mlflow.set_experiment(cfg["experiment_name"])

    # # enable system metric logging (cpu util, etc.)
    # mlflow.enable_system_metrics_logging()

    # # run training
    # with mlflow.start_run():
    #     mlflow.log_params(flatten(cfg, reducer="dot"))
    #     trainer.train(**cfg["training"])


if __name__ == "__main__":
    main()

    # render_fps = 60

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
