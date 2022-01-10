# zillow_clustering_project

## About the project

### Project Goals

* To identify drivers of error in the Zestimate in order to improve accuracy of predicting home values.

### Project Description

Decreasing the Zestimate error is important because it provides better information to our users and ensures Zillow maintains it's competitive advantage in the marketplace. 
The model selected will be evaluated by how well it performs over the baseline and previous models.

### Initial Hypotheses/Questions

* Question 1
Is logerror significantly different for properties in LA County vs Orange County vs Ventura County?

* Question 2
Is there a relationship between logerror and longitude and latitude?

* Question 3
Is there a relationship between logerror and bedrooms?

* Question 4
Would clustering on the physical characteristics be useful?


### Data dictionary

|   Column_Name   | Description | Type      |
|   -----------   | ----------- | ---------- |
| bathrooms |  Number of bathrooms in home | float |
| bedrooms   |  Number of bedrooms | int64  |
| area      |  Calculated total finished living area of the home   | int64 |
| counties      | los_angeles, orange, & ventura county | object |
| latitude      |  Latitude of the middle of the parcel multiplied by 10e6| int64 |
| longitude      |  Longitude of the middle of the parcel multiplied by 10e6 | int64 |
| taxvalue   | The total tax assessed value of the parcel   | int64    |
| logerror   | log(Zestimate)−log(SalePrice)       | float64    |
| age      | Years since home was built      | int64 |
| bath_bed_ratio   | Number of bedrooms / Number of bathrooms       |  float64 |

### Project Planning

**Planning**

* Define goals
* Determine audience and delivery format
* What is my MVP?
* Ask questions/formulate hypotheses

**Acquisition**
* Create function for establishing connection to zillow db
* Create function for SQL query and reading in results
* Create function for caching data
* Create wrangle.py to save these functions for importing
* Test functions
* Get familiar with data
* Document takeaways & things to address during cleaning 

**Preparation**
* Create function that cleans data
  * Handle missing values
  * Drop columns that contain duplicate information or are unnecessary
  * Convert data types
  * Rename columns 
  * Create dummy variables for columns with object datatype
* Create function that splits data into train, validate, and test samples
  * Split 20% (test), 24% (validate), and 56% (train)
* Create function that scales the data
* Test functions
* Add functions to wrangle.py to save for importing

**Exploration**
* Ask questions/form hypotheses
  * Question 1
  * Question 2
  * Question 3
  * Question 4
* Use clustering to explore data
* Create visualizations to help identify drivers
* Use statistical tests to test hypotheses
* Document answers to questions and takeaways
  * Answer/Takeaway 1
  * Answer/Takeaway 2
  * Answer/Takeaway 3
  * Answer/Takeaway 4

**Modeling**
* Identify, select, and create features that affect target variable (logerror)
* Establish a baseline
* Build, fit and use models to make predictions
* Compute evaluation metrics to evaluate models' performance
* Select best model and use on test dataset

**Delivery**
* Final Report in Jupyter Notebook
* Video recording of presentation
* README with project details
* Python modules with data preparation functions
* Audience will be the Zillow data science team

### To Recreate This Project:
* You will need an env file with your database credentials (user, password, hostname) saved to your working directory
* Create a gitignore and add your env file to prevent your credentials from getting pushed to Github
* Download the wrangle.py file to your working directory
* Download the zillow_clustering_project_final notebook to your working directory
* Read this README.md
* Run the zillow_clustering_project_final.ipynb notebook

### Recommendations and Next Steps
* Recommendation 1
* Recommendation 2

* Next Steps
