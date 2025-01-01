# Autism_DL
Project Overview:
    This project enhances Autism Spectrum Condition (ASC) screening by implementing Deep Learning (DL) models to analyze images and predict the likelihood of autism. It aims to complement behavioral assessments with image-based diagnostics for more comprehensive screening.

Key Features
1.Input Method: Users upload a child’s image for analysis.
2.Algorithms: Includes pre-trained models like InceptionV3, DenseNet169, and a custom-built Convolutional 
              Neural Network (CNN).
3.Best Performing Model: Custom CNN with Adam optimizer.
4.Deployment: The model is deployed on a Streamlit web app, enabling real-time predictions.

Technical Highlights
 Dataset: Sourced from Kaggle, containing labeled images for training and testing.
Tools Used:
          1.Programming Language: Python
          2.Libraries: TensorFlow, Keras, NumPy, Streamlit
Steps Involved:
1.Data preprocessing and augmentation.
2.Training and evaluating multiple DL models.
3.Deploying the best model for image-based ASC screening.

How It Works
1.Users upload a child’s image to the web app.
2.The image is processed and analyzed by the DL model.
3.The app outputs a probability score, indicating the likelihood of ASC.

Future Scope
1.Improve accuracy by expanding the image dataset.
2.Integrate speech or video-based features for multimodal ASC screening.
3.Optimize the model for faster real-time predictions.
