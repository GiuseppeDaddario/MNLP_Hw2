## Second Homework of MNLP
#### Sapienza University of Rome, Artificial Intelligence and Robotics Master Degree 
#### D'Addario Giuseppe MAT:2177530, Benucci Lorenzo MAT:2219690
***
### Overview
The goal of this homework is to explore the capabilities of different language models in correcting OCR errors in text. The evaluation of the results is carried out, based on the paradigm known as "LLM-as-a-Judge", using Prometheus-eval and Gemini. Results of the evaluation are assessed with correlation metrics including the Cohen's Kappa coefficient.
***
### How to run the code
The code both for re-generating the corrections and analyzing the results is available in this 
[Googl colab notebook](https://colab.research.google.com/drive/1ixKbLo5EVUr1gbbYYy1jvVuDGWIKiHK7?usp=sharing)

***
### Structure of the Repository

The repository is organized into the following main components:

- **`report.pdf`**: The report of the project.


- **`CINECA/`**: Contains files and scripts used to run experiments and generations on the CINECA cluster.

- **`outputs/`**: Contains output files to be delivered for the homework. Files are organized by model.
    - `llama4/`: Outputs from the Llama4 model.
    - `minerva/`: Outputs from the Minerva model.
    - `minerva_finetuned_lima/`: Outputs from the Minerva model finetuned on the LIMA dataset.
    - `minerva_finetuned_post_ocr/`: Outputs from the Minerva model finetuned also on the post-OCR Corrections dataset.
    - `t5/`: Outputs from the t5 model.


- **`plots/`**: Contains generated plots to be included in the report

- **`datasets/`**: Contains all datasets used in the project, organized by language.
  - `corrections/`: Model outputs and corrections used for evaluation, grouped by model (`t5`, `llama4`, `minerva`, `minerva_finetuned_lima`, `minerva_finetuned_post_ocr`).
  - `lima/`: LIMA dataset for the finetuning of Minerva.
  - `post_ocr_corrections/`: This directory actually doesn't exist here on github, because the dataset is too big to be uploaded. It was used after downloading it locally on Leonardo supercomputer.


- **`src/`**: Source code of the project.
    - `models/`: Implementations and wrappers for the models used (`t5`, `gemini`, `llama4`, `minerva`, `prometheus`).
    - `utils/`: Utility functions including functions for statistics.

- **`main.py`**: Main script for running experiments in local environment.

- **`requirements.txt`**: Python dependencies for the project.

- **`README.md`**: This documentation file.