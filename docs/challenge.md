
Initial scaffolding


Docker

Makefile

TODOS:
- check if problems running are because of windows or not


Data science

- Changed notebook, barplot requires paramters for x and y data.


Model

- Fixed test in model. Prediction was missing in one of the tests
- Fixed type hint in model class predict method
- Changed data path to absolute in tests
- added static method to compute delay target
- added Classifier abstraction and extracted that from the DelayModel class

TODOS:
- abstract model into Classifier class
- implement classifier for LogisticRegression and XGBoost classifier




