# Autism_DL
Project Overview:
The dataset used in this project consists of child facial images sourced from Kaggle. The dataset is first uploaded to Google Colab, where it is preprocessed by resizing images to 224x224 pixels, applying normalization, and performing data augmentation techniques like flipping and zooming. The dataset is then split into 80% training and 20% testing to ensure better generalization.

To achieve high accuracy, multiple deep learning models were trained and compared, including:

InceptionV3
DenseNet169
Custom CNN models (using Adam, RMSprop, and SGD optimizers)
CNN with SVM as the last layer
The models were evaluated using accuracy scores, confusion matrices, and classification reports. The best-performing model was then deployed using Streamlit, allowing users to upload a child’s image through a web interface. The trained model processes the image and predicts whether the child has a high or low probability of autism.

Steps included:
1. The user uploads an image to the web app.
2. The model extracts facial features and processes the image using deep learning algorithms.
3. The model predicts the probability of ASC based on the extracted features.
4. If the probability is ≥ 50%, the system suggests a higher likelihood of autism; otherwise, autism is not detected.
5. The result is displayed on the website for the user

Conclusion:
This project serves as a supportive tool for early autism detection, leveraging AI to assist parents and healthcare professionals. Although it does not replace clinical diagnosis, it provides an initial assessment to help decide whether further medical consultation is needed. Future improvements could involve larger datasets, better feature extraction, and additional deep learning enhancements to improve accuracy
