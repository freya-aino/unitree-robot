import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import jax
    import brax

    import mujoco as mj
    import polars as pl
    import marimo as mo
    import mujoco.viewer

    from brax.io import mjcf
    from mujoco import mjx
    from time import sleep, perf_counter
    from os import getcwd

    from threading import Lock, Thread
    return Lock, Thread, getcwd, mj, mjcf, mujoco, perf_counter, sleep


@app.cell
def _(Lock, Thread, getcwd, mj, mujoco, perf_counter, sleep):
    lock = Lock()

    mj_model = mj.MjModel.from_xml_path(f"{getcwd()}/external/files/scenes/scene.xml")
    # mj_model = mj.MjModel.from_xml_path(f"{getcwd()}/external/unitree-mujoco/unitree_robots/go2/scene.xml")
    mj_data = mj.MjData(mj_model)


    mj_viewer = mj.viewer.launch_passive(mj_model, mj_data)

    # mj.mj_resetData(mj_model, mj_data)
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
def _(getcwd, mjcf):
    brax_system = mjcf.load_mjmodel(f"{getcwd()}/external/unitree-mujoco/unitree_robots/go2/scene.xml")

    # brax_system.actuator_acc0
    brax_system.sensor()
    return


@app.cell
def _():
    from unitree_robot.common.controler import XboxController

    XboxController
    return


if __name__ == "__main__":
    app.run()
