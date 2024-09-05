# Fetal-Health-Classification

![Healthcare for the world image](/images/title_image.jpg)

## Dataset

This dataset is from Kaggle and can be found [here](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification).

Column Information:
- baseline value: Baseline Fetal Heart Rate (FHR)
- accelerations: Number of accelerations per second
- fetal_movement: Number of fetal movements per second
- uterine_contractions: Number of uterine contractions per second
- light_decelerations: Number of LDs per second
- severe_decelerations: Number of SDs per second
- prolongued_decelerations: Number of PDs per second
- abnormal_short_term_variability: Percentage of time with abnormal short term variability
- mean_value_of_short_term_variability: Mean value of short term variability
- percentage_of_time_with_abnormal_long_term_variability: Percentage of time with abnormal long term variability
- mean_value_of_long_term_variability: Mean value of long term variability
- histogram_width: Width of the histogram made using all values from a record
- histogram_min: Histogram minimum value
- histogram_max: Histogram maximum value
- histogram_number_of_peaks: Number of peaks in the exam histogram
- histogram_number_of_zeroes: Number of zeroes in the exam histogram
- histogram_mode: Hist mode
- histogram_mean: Hist mean
- histogram_median: Hist Median
- histogram_variance: Hist variance
- histogram_tendency: Histogram trend
- fetal_health: | 1 - Normal | 2 - Suspect | 3 - Pathological |

## Objectives

The main objective of this project is:
> Develop a system to detect abnormal fetal health in order to reduce child-under-5 mortality, especially in low-resource countries.

To achieve the main objective, here are sub-objectives:
1. Perform extensive exploratory data analysis of tabular data.
2. Perform feature selection to extract only the most important predictors.
3. Develop supervised machine learning approach to classify fetal health into 3 states.
4. Perform analysis on best-performing model
5. Deploy model (if time permits)

## Main Insights (EDA and Feature Selection)
- Data is heavily skewed towards Normal (1.0) fetal health target.
- Removing outliers will likely lead to loss of important examples in this dataset.
- Univariate analysis lends some insight into the features' relation to the target, but not enough to solely determine the target.
- Higher percentage of time with abnormal long term variability tends to cause pathological fetal health
- Higher amount of accelerations leads to fewer instances of suspect and pathological fetuses
- Features were deemed poor predictors for 2 reasons:
    - Highly correlated with each other (> 0.9)
    - Information gain < 0.1

## Business Metrics
- When deciding the scoring by which GridSearch would find the optimum model, I decided to go with **recall** (true positive rate). This is important in a medical setting because it is very important to never miss a patient with an unhealthy fetus (False Negative). This could endanger the life of the fetus and the mother.
- **Precision** is another metric which was used when evaluating models. It gives the number of False Positive predictions, which is when the model predicts a fetus is unhealthy, while in reality the fetus was actually healthy. This is definitely dangerous, but not to the extent as a False Negtive. The consequence of a false positive would be an unnecessary intervention, extra tests, and added stress to the family.
- **F-1 score** provides a harmonic mean of precision and recall, so it provides another metric to balance false positives and false negatives.
- **Confusion matrix** visually showcases the number of predictions correct and incorrect in addition to the predicted value. This gives a big picture view of TN, TP, FN and FP at one glance. 
- **AUC-ROC** tested my model at different classification thresholds using One vs Rest evaluation (ex. Healthy as + and Suspect/Pathological as -). This showed the balance of precision and recall and what impact it had on their values. Having a high AUC-ROC will demonstrate the model's ability to distinguish between all combinations of the classes in One vs Rest evaluation style.
- **Accuracy** has its place as a metric. In this project, it is not the best indicator of performance due to the heavy class imbalance. In other words, if the model predicted every fetus as healthy, then the model's accuracy would be 78%. Therefore, using only accuracy would be misleading to those trusting the efficacy of the project. Therefore, it is a supplementary metric and not an integral one.

## Model Selection
- KNN is a good benchmark model due to its simplicity and effectiveness.
- I decided to go with GridSearchCV to save time in finding the best hyper-parameters. This way, I can try multiple combinations and see which produces the best results. Also, I can choose which hyper-parameters to try, which will prevent underfitting and overfitting.
- The next model I utilized was Decision Tree. This model performed on par with KNN in in terms of the metric scores. Decision tree, however, is more interpretable than KNN.
- Next I used Random Forest Classifier. This model performed better than single decision tree model due to the fact that random forest builts multiple trees on different subset of features and data, so it learns different patterns when splitting the data. Also, it produces better results since it aggregates the predictions of multiple trees, so it takes the best of all trees it makes.

## Prediction Analysis
For each incorrect prediction, how much did its feature values differ from the average given its label? For example, if a fetus was predicted healthy, but was in reality suspect or pathological,
- What were its feature values?
- What is the average value for the features of a healthy fetus?
- What is the difference between these values with respect to a given feature?

The output of this can be found in `log.txt`

- Difference between value and mean value are very high for few features, namely `accelerations`, `percentage_of_time_with_abnormal_long_term_variability`, and `histogram_variance`.
- Percent difference for `accelerations` is noticeably 100% in a lot of cases. This is for a simple reason. The value for the given instance is 0 and the average value for that feature + label is a very small number. So, when calculating percent difference, it becomes _(0 - 0.001) / 0.001 * 100 = 100%_ 

_Further analysis pending..._

## Try it out!
