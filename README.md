# Credit Card Fraud Detection and Analysis

This project focuses on building a machine learning model to detect fraudulent credit card transactions. Using Python and several data science libraries, the notebook walks through the entire process from data cleaning and exploration to model training and evaluation. The core challenge of this project is handling a highly imbalanced dataset, a common scenario in fraud detection.

<br>

---

## Features

* **Data Loading & Preprocessing**: Loads and prepares a credit card transaction dataset.
* **Exploratory Data Analysis (EDA)**: Visualizes data to understand the distribution of features.
* **Handling Imbalanced Data**: Employs **SMOTE** (Synthetic Minority Over-sampling Technique) to address the class imbalance, a critical step for building an effective fraud detection model.
* **Machine Learning Model**: Uses a **Logistic Regression** model to classify transactions as legitimate or fraudulent.
* **Model Evaluation**: Evaluates the model's performance using metrics such as a **confusion matrix**, **classification report**, and **ROC curve**, which are essential for assessing performance on imbalanced datasets.

<br>

---

## Technologies Used

* **Python**: The primary programming language.
* **Pandas**: For data manipulation and analysis.
* **NumPy**: For numerical operations.
* **Matplotlib & Seaborn**: For data visualization.
* **Scikit-learn**: For machine learning model training and evaluation.
* **Imbalanced-learn (imblearn)**: For handling the imbalanced dataset.
* **Jupyter Notebook**: The environment where the analysis and code are executed.

<br>

---

## Getting Started

To run this project, you will need to clone the repository, install the necessary dependencies, and have the dataset available.

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/Manarsenic/your_repo_name.git](https://github.com/Manarsenic/your_repo_name.git)
    cd your_repo_name
    ```

2.  **Install dependencies**:
    * It is recommended to use a virtual environment.
    * Install all required libraries using the provided `requirements.txt` file (you can create one using `pip freeze > requirements.txt` if you don't have one).
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the dataset**:
    * This project uses the `creditcard.csv` dataset, which is publicly available. You can find it on platforms like **Kaggle**.
    * Download the dataset and place it in the project directory.

4.  **Run the notebook**:
    * Open `fraud_analysis.ipynb` in your Jupyter Notebook or Google Colab environment.
    * Execute the cells in order to see the full analysis and model results.

<br>

---

## Output & Analysis

### Class Distribution Plot
This bar chart shows the severe imbalance between fraudulent and non-fraudulent transactions.
![A bar chart showing a heavily imbalanced dataset with two classes.](https://i.ibb.co/3mNky9f/class-distribution.png)

### SMOTE Plot
A visualization of the dataset after applying SMOTE, demonstrating how the synthetic data points help balance the classes.
![A scatter plot showing an imbalanced dataset with new synthetic points.](https://i.ibb.co/6P26z1F/smote-plot.png)

### Confusion Matrix
A heatmap showing the true positives, false positives, true negatives, and false negatives of the model.
![A heatmap showing a confusion matrix for a classification model.](https://i.ibb.co/p3q8s5Y/confusion-matrix.png)

### ROC Curve
A plot that illustrates the diagnostic ability of the model as its discrimination threshold is varied.
![A line graph showing an ROC curve with AUC score.](https://i.ibb.co/w7R2vFp/roc-curve.png)

### Classification Report
A table providing key metrics such as precision, recall, and F1-score for both classes.
![A text table showing a classification report with precision, recall, and f1-score.](https://i.ibb.co/C0600Wq/classification-report.png)

<br>
