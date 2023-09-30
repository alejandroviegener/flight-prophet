
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

## Implementation details

### Part 1

The task consisted in transcribing the `.ipynb` file into the `model.py` file. In the notebook, the DS analysis concluded that by balancing the target in the classification problem, better metrics were achieved. Two models using this technique were implemented:

- Linear regression    
- XGBoost

Both model have very similar metrics. Performance of the models is pretty low, specially for the 1 label (delay). So, the decision will be made in relation to simplicity of the algorithm.

The chosen classification model for this part is: **LogisticRegression**

For the implementation of this part the following things were done:
- Implementation of a **Classifier** abstract class in the **classifier** module. **LogisticRegression** and **XGBoostClassifier** are implementations of this interface. The **Classifier** interface is used by the **DelayModel**.
- Implementation of the **DelayModel**. Class methods and static methods we added to modularize and make the code more readable. Checks for the right data columns and allowed categorical values were implented to pass the tests. The delay model can be serialized and loadeed from a serialized file (pickle).


### Part 2

A REST API using fast-api was implemented. The code can be found in **challenge.api**. 

The app creation was encapsulated inside a function

```
def create_app(model_filepath: str) -> fastapi.FastAPI:
```

This function is then called passing the serialized model file as a parameter. The path is read from an environment variable. This variable is set in the production docker file to point to the corresponding serialized model file located in the binaries folder.

```
model_filepath = os.getenv("MODEL_FILEPATH", "./tmp/delay_model.pkl")
app = create_app(model_filepath)
```


### Part 3

The api was deployed in an EC2 instance in AWS. AWS was used simply because of familiarity.

To deploy the API, the following steps where performed:
- An EC2 instance was created
- ssh was used to connect into the instance.
- docker was installed and the docker production container image was downloaded from docker hub. (the image was published in the CD pipeline)
- the application was launched by starting a container of the productive image

The API can be found in:
- http://ec2-18-118-36-230.us-east-2.compute.amazonaws.com:8000/health


**Note**: why an EC2 instance? why not a serverless approach? The idea in this case was just to expose the API for demonstration purposes. This is by no means a productive system. The characteristics of the API usage would probably determine if a server or serverless deployment would be best. For example, if the API traffic is low, then probably a serverless approach would be better in relation to costs. For a high demanding API, a server approach would probably have to be considered (and as well the implementation of a load balncer). System configuration, cost, cold starts, availability, load, are all things that should be considered in a productive case to decide what approach to use.


### Part 4

Continous integration and Continous delivery pipelines were implemented. See *ci.yml* and *cd.yml* files in **.github** folder.
CI pipeline is triggered on a **pull request** to **develop** or **main**. The CD pipeline is triggered on a tag push that has the format of a SemVer version number; the idea is to generate a productive docker image and push it to Docker Hub when a version TAG is created (example: v0.1.0-test, c0.2.1-prod, etc).

**CI pipeline steps**:
- Create a simbolic link for the test compose file (ln -s docker-compose.test.yml docker-compose.override.yml). This merges the test compose file with the base compile file when using the docker-compose application.
- Build the image
- Run the **code-quality-check**, **model-test** and **api-test** commands on the service. These are all targets of the **Makefile** and **make** command is set as the **ENTRYPOINT** in the test **Dockerfile.test**:
    - run: docker-compose run -T flight-prophet code-quality-check
    - run: docker-compose run -T flight-prophet model-test
    - run: docker-compose run -T flight-prophet api-test


**NOTE**:  pre-commit was added as a dependency. Hooks for code quality checks (**black** and **isort**) were added on commit (see the file *.pre-commit-config.yaml*)

**CD pipeline steps**:
- Checkout the code
- Log in to Docker Hub (secrets for docker hub username and token are set in Git repo) 
- Build Dockerfile.prod and push the image created to Docker Hub with SemVer tag => aviegener/latam-mle-challenge:SemVer


## Dockerization

**docker** and **docker-compose** are used in the challenge implementation.

- docker-compose.yml: defines the service flight-prophet and the port mapping.
- docker-compose.dev.yml: configures the Dockerfile.dev to use to build the service image. Creates a volume and attaches the challenge application folder to it. This enables editing the code and running the app inside the container.
- docker-compose.test.yml: configures the Dockerfile.test to use to build the service image. Also creates a volume and attaches the challenge application folder to it. 

To work in test or dev mode, a simlink named **docker-compose.override.yml** must be created  and targetted accordingly. For example:
- ln -s docker-compose.test.yml docker-compose.override.yml

Then docker-compose interpretes this override file to be merged with the base docker compose file.

Dockerfile files were created for each of the configurations:
- Dockerfile.test
- Dockerfile.dev
- Dockerfile.prod


## TODOs
- Use **Poetry** to manage dependencies: some versions had to be changed in the requirement files in order to solve certain dependency issues. This can be better achieved using **Poetry**.

