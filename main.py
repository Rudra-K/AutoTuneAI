import typer

from examples.experiments.exp_lgbm import run_lgbm_experiment
from examples.experiments.exp_rf import run_rf_experiment
from examples.experiments.exp_svm import run_svm_experiment
from examples.experiments.exp_cnn import run_cnn_experiment
from examples.experiments.exp_transformer import run_transformer_experiment

app = typer.Typer()

@app.command()
def tune_lgbm(
    data_path: str = typer.Option("tests/creditcard.csv", help="Path to the CSV dataset."),
    n_trials: int = typer.Option(50, help="Number of tuning trials to run."),
    adaptive: bool = typer.Option(False, "--adaptive/--no-adaptive", help="Enable adaptive search."),
):
    """
    Tunes a LightGBM model on the specified dataset.
    """
    run_lgbm_experiment(data_path=data_path, n_trials=n_trials, adaptive=adaptive)

@app.command()
def tune_rf(
    data_path: str = typer.Option("tests/creditcard.csv", help="Path to the CSV dataset."),
    n_trials: int = typer.Option(25, help="Number of tuning trials to run."),
    adaptive: bool = typer.Option(True, "--adaptive/--no-adaptive", help="Enable adaptive search."),
):
    """
    Tunes a RandomForest model using the experiment script.
    """
    run_rf_experiment(data_path=data_path, n_trials=n_trials, adaptive=adaptive)

@app.command()
def tune_svm(
    data_path: str = typer.Option("tests/creditcard.csv", help="Path to the CSV dataset."),
    n_trials: int = typer.Option(30, help="Number of tuning trials to run."),
):
    """
    Tunes a Support Vector Machine (SVM) model.
    """
    run_svm_experiment(data_path=data_path, n_trials=n_trials)

@app.command()
def tune_cnn(
    n_trials: int = typer.Option(30, help="Number of tuning trials to run."),
    epochs: int = typer.Option(10, help="Number of epochs to train each trial."),
):
    """
    Tunes a CNN model on the CIFAR-10 dataset.
    """
    run_cnn_experiment(n_trials=n_trials, epochs=epochs)

@app.command()
def tune_transformer(
    n_trials: int = typer.Option(20, help="Number of tuning trials to run."),
    epochs: int = typer.Option(5, help="Number of epochs to train each trial."),
):
    """
    Tunes a Transformer model on the IMDb dataset.
    """
    run_transformer_experiment(n_trials=n_trials, epochs=epochs)

if __name__ == "__main__":
    app()