import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    from brax.io import mjcf
    return (mjcf,)


@app.cell
def _(mjcf):
    mjcf.load_model
    return


if __name__ == "__main__":
    app.run()
