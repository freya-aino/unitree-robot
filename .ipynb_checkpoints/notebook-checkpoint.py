import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # import jax
    # import brax

    # import mujoco as mj
    # import polars as pl
    # import marimo as mo
    # import mujoco.viewer

    # from brax.io import mjcf
    # from mujoco import mjx
    # from time import sleep, perf_counter
    # from os import getcwd

    # from threading import Lock, Thread
    return


@app.cell
def _():
    import gymnasium as gym

    env = gym.make("Ant-v5", render_mode="rgb_array", width=1280, height=720)
    return (env,)


@app.cell
def _(env):
    env
    return


@app.cell
def _(getcwd):
    MJCF_SCENE_PATH = f"{getcwd()}/external/mjcf-robot-files/go2/scene.xml"

    from unitree_robot.train.training import Trainer
    from unitree_robot.train.environments import Go2Environment

    env = Go2Environment(mjcf_path=MJCF_SCENE_PATH)
    Trainer(env=env, device="cpu", network_hidden_size=16)
    return MJCF_SCENE_PATH, env


@app.cell
def _(Lock, MJCF_SCENE_PATH, Thread, mj, mujoco, perf_counter, sleep):
    lock = Lock()

    mj_model = mj.MjModel.from_xml_path(MJCF_SCENE_PATH)
    mj_data = mj.MjData(mj_model)


    mj_viewer = mj.viewer.launch_passive(mj_model, mj_data)

    mj.mj_resetData(mj_model, mj_data)
    mj_model.opt.timestep = 1/200

    sleep(0.2)

    def ViewerThread():
        while mj_viewer.is_running():
            lock.acquire()
            mj_viewer.sync()
            lock.release()
            sleep(1/50)

    def SimulationThread():
        while mj_viewer.is_running():
            step_start = perf_counter()

            lock.acquire()
            mujoco.mj_step(mj_model, mj_data)
            lock.release()

            time_until_next_step = mj_model.opt.timestep - (
                perf_counter() - step_start
            )
            if time_until_next_step > 0:
                sleep(time_until_next_step)


    viewer_thread = Thread(target=ViewerThread)
    simulation_thread = Thread(target=SimulationThread)

    viewer_thread.start()
    simulation_thread.start()
    return


@app.cell
def _():
    return


@app.cell
def _():
    # from unitree_robot.common.controler import XboxController

    # XboxController
    return


if __name__ == "__main__":
    app.run()
