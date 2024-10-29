
# Traffic Flow Prediction System (TFPS)

## Overview
The Traffic Flow Prediction System (TFPS) is a machine learning-based application designed to predict traffic flow and provide route optimization for the city of Boroondara, Melbourne. This system utilizes historical traffic data from VicRoads and implements various machine learning techniques to offer accurate traffic predictions.

## Prerequisites

1. **Anaconda**: Ensure that you have Anaconda installed on your machine. You can download it from [Anaconda's official website](https://www.anaconda.com/products/distribution).

## Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a Virtual Environment**
   ```bash
   conda create --name tfps python=3.8
   ```

3. **Activate the Virtual Environment**
   ```bash
   conda activate tfps
   ```

4. **Install Required Packages**
   Install the necessary packages using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Training the Model (Optional)

> **Note**: Training the model may take several hours, depending on the dataset and hardware capabilities.

To train the model, run the following command:
```bash
python train.py
```

### Running the GUI

To quickly test the system without training the model, run the GUI application:
```bash
python main.py
```

### Fetching Traffic Data

1. Open the GUI.
2. Navigate to the **Basic** section by running the `basic.py` file:
   ```bash
   python basic.py
   ```
3. Enter the date and SCATS number and Date Between 2006-10-01 to 2006-10-31.
4. Press the **Fetch Data** button to auto-fill the data fields.
5. Fill in any additional required fields.
6. Click the **Predict Traffic flow** button to test the routes.

### Entering Route Information

1. Enter the **origin** SCATS number and **destination** SCATS number in the respective fields.
2. Press the **Find Routes** button to get the predicted routes and estimated travel times.

## Conclusion

The TFPS application provides a user-friendly interface for predicting traffic flow and optimizing routes based on real-time data. For any questions or issues, please refer to the project documentation or contact the project team.
