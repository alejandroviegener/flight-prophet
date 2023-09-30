
# MLE Challenge Documentation

## General Description

This documentation describes the main ideas in the implementation of the MLE Challenge. The repository for the project can be found here: https://github.com/alejandroviegener/flight-prophet/

The workflow used during the implementation of the challenge was **gitflow**, this is:
 - define a **main** branch
 - define a **develop** branch
 - features to implement are branched from **develop**, named **feature/some-feature**. On termination, a pull request to **develop** is opened. 
 - when a group of features integrated into **develop** compose a new version, a pull request from **develop** to **main** is performed.

 For a graphical representation of the workflow implemented during this callenge, see the git [network](https://github.com/alejandroviegener/flight-prophet/network) graph of the repo. The list of the main feature branches that are found in the project can serve as an idea of the different features that have been implemented:

- **feature/initial-scaffolding**:  initial directory structure 
- **feature/model-v1**: first version model implementation. Used notebook to abstract the model into the DelayModel class
- **feature/docker-compose**: dockerized the solution
- **feature/api**: implementation of the api using fast-api framework
- **feature/ci**: implementation of a CI pipeline that runs the model and api tests
- **feature/hooks**: implementation of code quality pre commit hooks (isort and black) 
- **feature/model-v2**: implemented second model
- **feature/cd**: implementation of continous delivery pipeline => docker image gets build and pushed to docker hub


## Directory structure and files

This section describes the main directories and files in the project.

- **challenge**
    - *api.py*: contains the api implementation
    - *model.py*: contains the DelayModel implementation
    - *classifier.py*: contains the Classifier abstract class definition and its implementations (LogisticRegressionClassifier and XGBoostClassifier)
    - *exploration.ipnb*: exploration analysis done by the DS
- **tests**: contains the api, model and stress test files 
    - **resources**
        - *test_model.pkl*: serialized model used in the api tests
- *train.py*: script to train the model, parametrized to select input data and model output
- *Makefile*: containes several targets, added **api-run**, **model-train** and **code-quality-check** targets to the existing ones.
- *docker-compose.yml*: base compose file
- *docker-compose.dev.yml*: extends the base compose file for dev purpuses
- *docker-compose.test.yml*: extends the base compose file for test purpuses
- *Dockerfile.prod*: dockerfile that defined the image for production
- *Dockerfile.test*: dockerfile that defined the image for dev
- *Dockerfile.dev*: dockerfile that defined the image for test

## Dockerization

## Implementation details

### Part 1

### Part 2

### Part 3

### Part 4



## TODOs
- Use poetry to manage dependencies
