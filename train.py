import hydra
import mlflow
from flatten_dict import flatten
from omegaconf import DictConfig, OmegaConf
from torch import optim

from unitree_robot.common.agents import PPOAgent
from unitree_robot.training.environments import Go2Env
from unitree_robot.training.experiments import StandUpExperiment
from unitree_robot.training.trainer import Trainer


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_object(cfg)

    print(cfg)

    environment = Go2Env(**cfg["environment"])

    experiment = StandUpExperiment(mj_model=environment.model, **cfg["experiment"])

    agent = PPOAgent(
        input_size=environment.observation_space_size,
        policy_output_size=environment.action_space_size * 2,
        train_sequence_length=cfg["training"]["minibatch_size"],
        **cfg["agent"],
    )

    optimizer = optim.AdamW(agent.parameters(), **cfg["optimizer"])

    trainer = Trainer(
        environment=environment,
        experiment=experiment,
        agent=agent,
        optimizer=optimizer,
        max_gradient_norm=cfg["max_gradient_norm"],
        seed=cfg["seed"],
        device=cfg["device"],
    )

    if not mlflow.get_experiment_by_name(cfg["experiment_name"]):
        mlflow.create_experiment(cfg["experiment_name"])
    mlflow.set_experiment(cfg["experiment_name"])

    with mlflow.start_run():
        mlflow.log_params(flatten(cfg, reducer="dot"))
        trainer.train(**cfg["training"])


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
