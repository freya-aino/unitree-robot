import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from brax.io import mjcf
    from mujoco import mjx

    import mujoco as mj
    import polars as pl
    import marimo as mo
    import mediapy as media

    import jax
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
