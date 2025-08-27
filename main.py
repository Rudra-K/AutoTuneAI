import typer

from examples.experiments.exp_lr import run_lr_experiment
from examples.experiments.exp_cnn import run_cnn_experiment

app = typer.Typer()

@app.command()
def tune_lr(
    data_path: str = typer.Option("tests/creditcard.csv", help="Path to the CSV dataset."),
    n_trials: int = typer.Option(20, help="Number of tuning trials to run."),
    adaptive: bool = typer.Option(True, "--adaptive/--no-adaptive", help="Enable adaptive search."),
):
    """
    Tunes a Logistic Regression model using the experiment script.
    """
    run_lr_experiment(data_path=data_path, n_trials=n_trials, adaptive=adaptive)

@app.command()
def tune_cnn(
    n_trials: int = typer.Option(30, help="Number of tuning trials to run."),
    epochs: int = typer.Option(10, help="Number of epochs to train each trial."),
    adaptive: bool = typer.Option(True, "--adaptive/--no-adaptive", help="Enable adaptive search."),
):
    """
    Tunes a CNN model on the CIFAR-10 dataset.
    """
    run_cnn_experiment(n_trials=n_trials, epochs=epochs, adaptive=adaptive)


if __name__ == "__main__":
    app()