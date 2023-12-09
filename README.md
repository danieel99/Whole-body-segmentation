# Whole-body-segmentation

# Project Overview
This project explores the application of deep learning in the automated analysis of cancerous lesions from whole-body PET/CT images. Aiming to enhance the accuracy and efficiency of cancer diagnosis, the project presents a comparative study of different neural network architectures for automatic segmentation of cancer lesions. This work is a significant step forward in medical imaging and diagnosis, leveraging the power of artificial intelligence to improve patient outcomes.

# Key Features
Neural Network Architectures: Utilizes multiple configurations of U-Net, nnU-Net, and UNETR models, which are leading tools for medical image processing.
Two-Step Segmentation Approach: Incorporates a two-step segmentation process to improve segmentation precision, including a strategy that refines segmentation using lower-resolution image predictions.
Performance Analysis: Includes a critical analysis of model performance, particularly the effect of excluding non-cancer cases from training datasets.

# Technology stack
Software and Tools:
3D Slicer: An open-source tool used for enhanced visualization of three-dimensional medical images. It played a significant role in assessing image quality, comparing dimensions, and visualizing modalities with appropriate markings.

Programming Language and Framework:
Python: The primary programming language for code implementation.
PyTorch: A widely-used deep learning framework chosen for its intuitive interface and dynamic computational graph compilation, facilitating experimentation and debugging.

Python Libraries:
MONAI: Used as an extension to PyTorch, particularly for medical imaging applications.
TorchIO: Another extension to PyTorch, focusing on efficient loading, preprocessing, augmentation, and patch-based sampling of medical images.
NumPy: Essential for numerical computations and array manipulations.
Pandas: Used for data manipulation and analysis.
SimpleITK: A toolkit for medical imaging, used for advanced image processing.
scikit-learn: Employed for machine learning and statistical modeling tasks.

Optimization and Loss Functions:
Adam Optimizer: A popular optimizer used for updating network weights during learning.
DiceFocalLoss: A combined loss function, integrating Dice loss and Focal Loss, with the addition of a sigmoid parameter for value calculation.
Cosine Annealing with Warm Restarts: A learning rate scheduler adjusting the learning rate based on the cosine function and resetting it after a specific number of epochs.

# Dataset
AutoPET Challenge Dataset: Employed for training and testing the deep learning models. The dataset is a collection of whole-body PET/CT images, with varying cases of cancer presence.

# Methodology
One-Step Segmentation: It is a reference point for further work and nvestigates the effects of excluding non-cancer cases on model performance.
Two-Step Segmentation:
First step involves initial segmentation on lower-resolution images.
Second step uses initial predictions to inform and refine segmentation.

# Key Findings
The nnU-Net model demonstrated exceptional performance in both scenarios: using the entire dataset and when only images with tumors were considered.
Excluding non-cancer cases from the training set significantly improves model performance across most architectures tested.

# Potential Impacts
Enhances the accuracy and efficiency of cancer diagnosis.
Contributes valuable insights into the development of deep learning algorithms in medical imaging.
Has the potential to influence clinical practices and patient outcomes positively.

# Contributions
This project invites further development and contributions, particularly in:

Enhancing model accuracy and efficiency.
Expanding the dataset with more diverse cases.
Exploring new neural network architectures for image segmentation.

# Acknowledgments
Recognition of any contributors or institutions that played a pivotal role in this project.
References to any papers, datasets, or tools used in the project.
