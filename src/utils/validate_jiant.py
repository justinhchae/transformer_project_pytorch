from src import utils

from jiant.proj.simple import runscript as run
import jiant.scripts.download_data.runscript as downloader


def validate_jiant():
    # validation scripts
    # https://github.com/nyu-mll/jiant

    EXP_DIR = utils.constants.DATA_PATH

    # Download the Data
    downloader.download_data(["mrpc"], f"{EXP_DIR}/tasks")

    # Set up the arguments for the Simple API
    args = run.RunConfiguration(
        run_name="simple",
        exp_dir=EXP_DIR,
        data_dir=f"{EXP_DIR}/tasks",
        hf_pretrained_model_name_or_path="roberta-base",
        tasks="mrpc",
        train_batch_size=16,
        num_train_epochs=1
    )

    # Run!
    run.run_simple(args)
