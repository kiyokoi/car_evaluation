# Car Evaluation
Created a model to evaluate cars according to their cost and technical characteristics.

###Code

Code is provided in `car_evaluation.py`. 
This program requres Python 2.7 and the following Python libraries installed:

* NumPy
* Pandas
* matplotlib
* Seaborn
* Scikit-learn

###Data

Dataset used in thie project is included as car.csv and weather.csv. Dataset was obtained from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation) and contains the following attributes.

Price attributes:
* `buying`: buying price (v-high, high, med, low)
* `maint`: price of the maintenance (v-high, high, med, low)

Technical characteristics and comfort attributes:
* `doors`: number of doors (2, 3, 4, 5-more)
* `persons`: capacity in terms of persons to carry (2, 4, more)
* `lug_boot`: the size of luggage boot (small, med, big)
* `safety`: estimated safety of the car (low, med, high)

Car class attribute:
* `class`: classification (v-good, good, acc, unacc)
