# 😴 Drowsiness Detection Using dlib 🤖

## Overview
This project focuses on developing a system to detect signs of driver drowsiness using dlib’s landmark detection shape_predictor utility. The system is designed to identify drowsiness indicators such as closed eyes, yawning, and head movements using a custom dataset. The ultimate goal is to enhance driver safety by providing timely alerts when signs of drowsiness are detected.

## Performance Analysis
### Training and Testing Accuracy
- **Training Accuracy**: The custom dlib shape predictor model achieved a Mean Absolute Error (MAE) of 3.66 during training, indicating a good ability to detect drowsiness-related facial features.
- **Testing Accuracy**: The model's performance on the test set yielded an MAE of 7.19, which is higher than desired for robust detection. This discrepancy highlights the need for further model optimization to effectively capture the complexity of drowsiness indicators.

### Insights
The relatively high testing error suggests that the model struggles with the diversity of drowsiness indicators, such as varied eye gaze directions and yawning. Improving the model's accuracy in these areas is crucial for real-world application in driver monitoring scenarios.

## Results
- **`status_plot.png`**: ![Training and Testing Status Plot](https://github.com/ParthJohri/Drowsiness_Detection_Using_Dlib/blob/d3cd0a1f306da64aede44a21b1c7ba81acef588b/status_plot.png)
- **`ear.png`**: ![Eye Aspect Ratio Calculation](https://github.com/ParthJohri/Drowsiness_Detection_Using_Dlib/blob/d3cd0a1f306da64aede44a21b1c7ba81acef588b/ear.png)
- **`dlib_result.png`**: ![Sample Output of dlib Shape Predictor](https://github.com/ParthJohri/Drowsiness_Detection_Using_Dlib/blob/d3cd0a1f306da64aede44a21b1c7ba81acef588b/dlib_result.png)

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/ParthJohri/Drowsiness_Detection_Using_Dlib.git
   ```
2. Navigate to the project directory:
   ```sh
   cd Drowsiness_Detection_Using_Dlib
   ```
3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```
4. [Optional] Custom Training of the model
   ```sh
   The notebook is also provided, if you would like to custom train your model with your dataset.
   ```
## Usage
1. Prepare your dataset with labeled images for training and testing.
2. Train the dlib shape predictor model using the provided training script.
3. Test the model on the test dataset to evaluate its performance.
4. Use the model in real-time applications to monitor drivers and detect signs of drowsiness.

## Future Work
- **Model Optimization**: Refine the dlib shape predictor model to reduce testing error and improve robustness.
- **Dataset Expansion**: Include more diverse and comprehensive datasets to capture a wider range of drowsiness indicators.
- **Real-time Implementation**: Develop and test the system in real-world driving scenarios to ensure its practical applicability and reliability.

## Contributing
Contributions to enhance the drowsiness detection system are welcome. Please fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
