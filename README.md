# ML_algorithmic_development_mental_disease_prediction

## About this project
### Type of project/purpose of repository
This project is mainly a Data science project with the overall purpose of developing complex Machine-learning pipelines to iteratively improve classification performance for monitoring heart patient's mental state.

The purpose of this repository is to showcase the usefullnes of using ML for classifying mental health issues using patient-reported and demographic data. This project also reflects a special emphasis on solving problems related to real-world data, more precisely, working with smaller datasets with serious problems related to missing values, etc. 

The overall emphasis will be to explain the key concepts and thoughts about the algorithmic development through visualization and code in this README-file.

### Project contributors and credits
This project is the outcome of three sequantially carried out master thesis projects from Syddansk Universitet (SDU), that has been developed over a two year period. The project is a part of a larger initiative that has been presented by the ACQUIRE-ICD foundation which is a collaboratory research-based project managed by OUH and SDU, which aims to increase patient related outcomes in relation to heart patients. For more information about this project and its organization see [ACQUIRE-ICD](https://www.sdu.dk/en/om-sdu/instituttercentre/institut_psykologi/forskning/forskningsprojekter_/acquire-icd).
 
The main authors of the final rendition of the project which is presented here are Jakob Eriksen and David Krogh Kølbæk. However this project is greatly influenced by our former supervisors Uffe Will and Ali Ebrahimi, which have provided continuous feedback on key processes related to project management and ML-based development. The earlier iterations of this project has been carried out by Jonas Pedersen and Ebbe Christensen. Also a special thanks is givin to the phycological department of OUH which have provided great feedback on clinical variables through their extensive domain knowledge. 

### Practical circumstances
Important notice: The exact dataset that have been used for this project is not sharable since it contains sensisitve patient information and cannot be assessed without special authorization from Odense Universitets Hospital (OUH). If one wishes to work further on this project it is recommended to contact Odense Universitets Hospital and ask for the ACQUIRE-ICD project.

The general dataformat and a visualization of what variables have been used will still be presented to showcase what type of data will work for development of ML-pipelines.


## How to run this project / Requirements
Important notice: Some of the processes in this project have been catered to a very specific instance of data variables and might not be best practise for data in other formats. Therefore, it is recommended to consider the general outline of the data variables that have been used and take the data-format into context and consideration when creating your own project.

### Setting Up the Project

1. **Clone the Repository**: Start by cloning the GitHub repository to your local machine:
    
    ```
    git clone <repository_url>
    cd <repository_directory>
    ```
    
2. **Set Up a Virtual Environment** (optional but recommended): Create a virtual environment to manage the dependencies.
    
    ```
    python -m venv env
    env\Scripts\activate
    ```
    
3. **Install Dependencies**: Use the `requirements.txt` file to install all the required libraries.
    
    ```
    pip install -r requirements.txt
    ```
    
4. **Jupyter Notebook Installation**: Since the project is developed in Jupyter notebooks, ensure that Jupyter is installed. You can install it with:
    
    ```
    pip install jupyter
    ```
    

### Running the Project

1. **Launch Jupyter Notebook**: To run the project, start Jupyter Notebook from the command line:
    
    ```
    jupyter notebook
    ```
    
    This command will open Jupyter in your default web browser. From there, navigate to the notebook files of the project.
    
2. **Select the Notebook**: Open the specific notebook(s) you want to explore. Each notebook will contain data preprocessing steps, model training, evaluation, and visualization.
3. **Run Cells**: Run each cell in the notebook sequentially to reproduce the analysis. Ensure that the data files are in the appropriate locations as mentioned in the notebook. The precise details of the data that have been used will be explained in another segment. Important notice: The exact dataset that have been used for this project is not sharable since it contains sensisitve patient information and cannot be assessed without special authorization from Odense Universitets Hospital (OUH). 

- Explain how data should be formattted


### Background/problem
Use introduction segment from report.
Details about the ML-based problem to be solved - Dataset size, missing values, feature space, type of data (medical data, patient reported data, demographic data).
Highlight problems in relation to the nature of the data.
### Project solution
Use Project overview figure from report.

## ML-development process
### Preprocesssing
### Model development
### Model evaluation
### XAI 

## Code implementation
### Overview of development steps (codefiles)
Explain the environment the code was build and executed in - Ucloud, computational ressources. 
Explain what each file do in general and in order.

1: pipeline preproccesing
2: Scaling
3: all_tuned_models
4: stacking_voting
5: metrics_final_models
6: shap

Explain what the purpose of each file is. 
