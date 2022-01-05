# Check My Skin project
Creation of a web application for skin cancer detection/prevention for Jedha Bootcamp's final project.

## General observations
- The number of dermatologists in France, especially in rural areas, has been decreasing for the last 25 years. Medical desertification is getting even more important nowadays, leading to a lack of medical care to persons suffering from skin disorders and skin cancers.
- Pople living in rural areas have to wait for more than 6 months to have an appointment with a dermatologist. 
- This phenomenon is endangering the lives of persons who do not have access to proper care at the right time.
- The number of French citizens suffering from skin disorders is increasing year by year.
- On the other hand, the number of mobile phones is growing exponentially in the French society.

## The idea
Make use of technology to solve a societal problem and encourage prevention measures, namely towards persons living in places with limited access to care.

## The technology
Combine computer vision, a branch of artificial intelligence, with mobile technology to build an application which will help in detecting and preventing skin cancer.
Therefore, in this project, we plan to use computer vision techniques to recognize different types of skin lesions and provide a supporting tool for the detection of skin cancer. In current medical diagnosis, identifying skin cancer has always been challenging because of its close assemblance to other types of skin diseases. In this project, by applying deep learning techniques, we aim to classify skin lesion/mole in three categories (not dangerous, moderately dangerous or very dangerous) based on images of skin diseases on different locations of the patient's body.

## Data used
We have used the HAM10000 ("Human Against Machine with 10000 training images") that consists of 10015 dermatoscopic images, seven diagnostic categories reduced to three categories for our use case. We also used a metadata file with information of sex, age, and the location of the skin feature. All photos were 600 pixels wide and 450 pixels high. The data can be found online at https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

## Production phase (10 days)
- Exploratory data anaysis (EDA) to understand the data
- Development of a baseline model
- Tests involving diffrerent hyperparameter tuning of the baseline model
- Tests on existing transfer learning models
- Combining images and metadata file available in a tabular format to build a deep learning model
- Ensuring transparency of the deep learning model
- Deploying the deep learning model as an application on Streamlit

## Outcomes
- The model has an overall accuracy of 94%.
- Instead of accuracy, we considered recall (probability for the "moderately dangerous" and "very dangerous" cases to be correctly predicted) as our most important metric.
- The Minimum Viable Product our team developed can be accessed on the following link: https://lnkd.in/dRGTuAzX
- For a presentation and a demo of the CheckMySkin App, please watch the following video : https://lnkd.in/dV7h834h