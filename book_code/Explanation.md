# Methodology
This section explains the entire process that was carried out to find potential future companies.

The main idea is to segment customers. This process was carried out observing the main characteristics of them. Thus, it could be seen that 86% of the clients were distributed in the following autonomous communities (GALICIA, PAIS VASCO, CATALUNYA, COMUNITAT VALENCIANA, MADRID, ANDALUCIA) and have the following economic activity (INDUSTRIA MANUFACTURERA, ACTIVIDADES PROFESIONALES, CIENTIFICAS Y TECNICAS,
 COMERCIO AL POR MAYOR Y AL POR MENOR; REPARACION DE VEHICULOS DE MOTOR Y MOTOCICLETAS, INFORMACION Y COMUNICACIONES). With these characteristics, a filtering is applied to the dataset to find the companies that do not comply with this features and thus select those who are not clients and then create a training dataset of clients and non-clients in order to train a propensity model.

## OBJECTIVE: Create a model to find next potential clients.

## Pre-process

To perform pre-processing, a class called "Preprocces" was developed. This class has different methods:

* clean_dataframe: It was used to clean / pre-process the data.
* fill_numerical_na: Null values are replace with the method `IterativeImputer`
* fill_categorical_na: Impute NaN using `KNNImputer`

## Model 

### Classifier:

For this task, a dataset were generated to train a propensity model.

* Different classifiers were trained, among them the following stand out:
    * MultinomialNB()
    * GaussianNB()
    * AdaBoostClassifier()
    * MLPClassifier()
    * DecisionTreeClassifier()
    * RandomForestClassifier()
    * XGBClassifier()
    * LogisticRegression()
    * GradientBoostingClassifier()

In this case, all models were falling into an overfitting, for that reason it decided, train a **LogisticRegression**, adding stronger regularization using the hyperparameter **C**. This helps to achieve better results for each classes using the recall and roc_auc_score metrics, for this reason it was chosen to make the final model and be used to make predictions for the test dataset.

## Criteria final in the choice of companies

* Autonomous communities: (GALICIA, PAIS VASCO, CATALUNYA, COMUNITAT VALENCIANA, MADRID, ANDALUCIA)
* Economic activity: (INDUSTRIA MANUFACTURERA, ACTIVIDADES PROFESIONALES, CIENTIFICAS Y TECNICAS,COMERCIO AL POR MAYOR Y AL POR MENOR REPARACION DE VEHICULOS DE MOTOR Y MOTOCICLETAS, INFORMACION Y COMUNICACIONES)
* Advertising impacts: (0,1,2,3)
* Probability of class 1: >0.95

Companies must meet the requirements mentioned above to be potential clients.

To see the all companies go to `./result/predict_potencial_client.csv`

# Conclusion

In accordance with the objectives set out in the test, different classifiers were used to classify if a company could be a next potencial client or not. The results showed that the optimal way to classify is using **LogisticRegression** , segment customers and impute NaN with the methods mentionate before.

It was decided to use recall and roc_auc_score metrics to measure the performance of the model optimally in each class. 

