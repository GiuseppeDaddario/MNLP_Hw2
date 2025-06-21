## Second Homework of MNLP

### Structure of the Repository

The repository is organized into the following main components:

- **`CINECA/`**: Contains files and scripts used to run experiments on the CINECA cluster.
    - `logs/`: SLURM logs and model outputs from submitted jobs.
    - `minerva/`: Scripts and configurations specific to the Minerva environment.
    - `main.sh`: SLURM submission script for running batch jobs.

- **`datasets/`**: Contains all datasets used in the project, organized by language.
    - `eng/`: English datasets.
        - `corrections/`: Model outputs and corrections used for evaluation, grouped by model (`deep_mount`, `minerva`, `minerva_finetuned_llima`, `minerva_finetuned_post_ocr`).
        - `llima/`: Input data for the Llima model (e.g., prompts, preprocessed texts).
    - `ita/`: Italian datasets (currently empty).

- **`extra_stuff/`**: Temporary files, exploratory notebooks, and miscellaneous materials to be reviewed or cleaned up.

- **`src/`**: Source code of the project.
    - `models/`: Implementations and wrappers for the models used (`deepmount.py`, `gemini.py`, `llama4.py`, `minerva.py`, `prometheus.py`).
    - `utils/`: Utility functions such as data loading and evaluation metrics.

- **`main.py`**: Main script for running local experiments.

- **`requirements.txt`**: Python dependencies for the project.

- **`README.md`**: This documentation file.