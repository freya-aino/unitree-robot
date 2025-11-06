import time
import warnings
import hydra
import jax
import mlflow
import torch as T
import tqdm
from flatten_dict import flatten
from omegaconf import DictConfig, OmegaConf
from torch import optim
from os import environ

from unitree_robot.common.agents import PPOAgent
from unitree_robot.training.environments import Go2EnvMJX
from unitree_robot.common.datastructure import UnrollData

# set xla flags for some GPUs
xla_flags = environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
environ["XLA_FLAGS"] = xla_flags


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    TODO:

        please check out how 'model.opt.timestep' should be set in mujoco
            model.opt.timestep
            Simulation step size (dt)
            Convert per‑step quantities to rates (e.g., power = work / dt).

    """

    cfg = OmegaConf.to_object(cfg)
    print(cfg)

    # -- set variables

    MINIBATCH_SIZE = cfg["training"]["minibatch_size"]
    TORCH_DEVICE = T.device(cfg["device"])
    RANDOM_SEED = cfg["random_seed"]

    # -- jax config

    jax.config.update("jax_platforms", "cuda" if TORCH_DEVICE == T.device("cuda") else "cpu")
    # jax.config.update("jax_distributed_debug", True)
    # jax.config.update("jax_log_compiles", False)
    jax.config.update("jax_logging_level", "ERROR")

    # -- setup environment

    environment = Go2EnvMJX(
        **cfg["environment"],
    )

    # mjx_data_batch = environment.reset(seed=0)
    # dt = time.time()
    # for i in tqdm.tqdm(range(1000)):
    #     action = T.randn(size=[environment.num_parallel_environments, 12]).to(device="cuda")
    #     # action = jax.random.uniform(key=jax.random.PRNGKey(0), shape=(environment.num_parallel_environments, 12))
    #     environment.step(mjx_data_batch, action_batch=action)
    # print(f"time: {(time.time() - dt) / 1000}")

    # batched_mj_data = mjx.get_data(environment.mj_model, batch)
    # environment = Go2Env(**cfg["environment"])
    # experiment = StandUpExperiment(mj_model=environment.model, **cfg["experiment"])

    agent = PPOAgent(
        input_size=environment.observation_space_size,
        policy_output_size=environment.action_space_size * 2,
        train_sequence_length=MINIBATCH_SIZE,
        **cfg["agent"],
    ).to(device=TORCH_DEVICE)


    # -- unroll

    unroll_data = UnrollData.initialize_empty(
        num_unrolls=environment.num_parallel_environments,
        unroll_length=cfg["training"]["unroll_length"],
        observation_size=environment.observation_space_size,
        action_size=environment.action_space_size,
    )


    observation = environment.reset(seed=RANDOM_SEED)
    for i in tqdm.tqdm(range(cfg["training"]["unroll_length"]), desc="Unrolling"):
        with T.no_grad():
            action, logits = agent.get_action_and_logits(observation)
        observation = environment.step(action=action)

    # print(agent(
    #     observations,
    #     actions,
    #     logits,
    #     rewards,
    # ))

    # optimizer = optim.AdamW(agent.parameters(), **cfg["optimizer"])

    # trainer = MultiEnvTrainer(
    #     environment=environment,
    #     agent=agent,
    #     experiment=experiment,
    #     copies=cfg["parallel_environments"],
    #     device=cfg["device"],
    #     optimizer=optimizer,
    # )

    # trainer.train(**cfg["training"])

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
