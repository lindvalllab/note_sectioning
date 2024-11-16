# Setup

1. :warning: Execute this command in the root of this repository to ensure notebook outputs are stripped automatically

    ```bash
    git config filter.strip-notebook-output.clean 'jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR'  
    ```

2. Create and activate a conda environment

    To do this, start by creating the provided environment. After navigating to the root of this repository, enter in your terminal:
    ```
    conda env create -f environment.yml
    ```
    And to activate the environment:
    ```
    conda activate notesectioning
    ```

3. Provide your Client Endpoint and Entra Scope

    Rename the file **.env.sample** to **.env**

    Enter your Client Endpoint and Entra Scope in this file.