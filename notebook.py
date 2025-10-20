import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
<<<<<<< HEAD
    from brax.io import mjcf
    from mujoco import mjx
=======
    import marimo as mo
    return


@app.cell
def _():
    from mujoco import mjx
    from mujoco import MjModel
    from brax.io import mjcf
    return MjModel, mjx
>>>>>>> 27bb1f0 (notebook changes)

    import mujoco as mj
    import polars as pl
    import marimo as mo
    import mediapy as media

<<<<<<< HEAD
    import jax
=======
@app.cell
def _(MjModel, mjx):
    mj_model = MjModel.from_xml_path("./external/files/mjcf/Go2/go2.xml")
    mjx_model = mjx.put_model(mj_model)
    mjx_model
    return


@app.cell
def _():
    # import functools
    # import brax

    # from brax import envs
    # from brax.training.agents.ppo import train as ppo

    # import jax


    # # env = envs.get_environment(
    # #     env_name="inverted_double_pendulum",
    # #     backend="positional"
    # # )
    # # state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed = 0))

    # train_fn = functools.partial(
    #     ppo.train,
    #     num_timesteps = 20_000,
    #     num_evals = 16,
    #     num_envs = 16,
    #     num_minibatches = 32,
    #     batch_size = 4,
    #     reward_scaling = 10,
    #     episode_length = 1000,
    #     normalize_observations = True,
    #     action_repeat = 1,
    #     unroll_length = 5,
    #     num_updates_per_batch = 4,
    #     discounting = 0.97,
    #     learning_rate = 3e-4,
    #     entropy_cost = 1e-2,
    #     seed = 0
    # )
    # # make_inference_fn, parameters, _ = train_fn(
    # #     environment = env,
    # #     progress_fn = lambda num_steps, metrics: print(metrics["eval/episode_reward"])
    # # )

    # import brax_envs
    # env = brax_envs.CustomEnv(
    #     backend = "positional"
    # )

>>>>>>> 27bb1f0 (notebook changes)
    return


app._unparsable_cell(
    r"""
    xml = \"\"\"
    <mujoco>
      <worldbody>
        <light name=\"top\" pos=\"0 0 1\"/>
        <body name=\"box_and_sphere\" euler=\"0 0 -30\">
          <joint name=\"swing\" type=\"hinge\" axis=\"1 -1 0\" pos=\"-.2 -.2 -.2\"/>
          <geom name=\"red_box\" type=\"box\" size=\".2 .2 .2\" rgba=\"1 0 0 1\"/>
          <geom name=\"green_sphere\" pos=\".2 .2 .2\" size=\".1\" rgba=\"0 1 0 1\"/>
        </body>
      </worldbody>
    </mujoco>
    \"\"\"

    mj_model = mj.MjModel.from_xml_string(xml)
    mj_data = mj.MjData(mj_model)
    renderer = mj.Renderer(mj_model)

    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)


    # enable joint visualization option:
    scene_option = mj.MjvOption()
    scene_option.flags[mj.mjtVisFlag.mjVIS_JOINT] = True

    duration = 3.8  # (seconds)
    framerate = 60  # (Hz)


    jit_step = jax.jit(mjx.step)

    frames = []
    mj.mj_resetData(mj_model, mj_data)

    while mjx_data.time < duration:
      mjx_data = jit_step(mjx_model, mjx_data)
      if len(frames) < mjx_data.time * framerate:
      
        mj_data = mjx.get_data(mj_model, mjx_data)
        renderer.update_scene(mj_data, scene_option=scene_option)
        pixels = renderer.render()
        frames.append(pixels)

    media.show_video(frames, fps=framerate)

    media.
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
