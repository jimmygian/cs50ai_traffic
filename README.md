# CS50AI | Lecture 5 - Neural Networks | Project 5 - [Traffic](https://cs50.harvard.edu/ai/2024/projects/5/traffic/)

This project is a mandatory assignment from **CS50AI – Lecture 5: "Neural Networks"**, focusing on creating a neural network with **TensorFlow** to classify traffic signs from image data. The objective is to build, train, and finally evaluate a convolutional neural network (CNN) that can accurately recognize traffic signs from the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

---

## 📌 Usage

1. Clone this repository or download the project files.
2. Ensure you have Python and TensorFlow installed. It’s recommended to use a conda virtual environment running python 3.9 (this is what worked for me at least).
3. Download the database from this [page](https://cs50.harvard.edu/ai/2024/projects/5/traffic/#getting-started).
4. Run the main program with the dataset directory as an argument and check process/results in terminal.



## My Implementation

My main coding goal was to implement two key functions: `load_data` and `get_model`.

The `load_data` function focuses on properly preparing the dataset for TensorFlow. To achieve this, I used Python’s `os` library to handle cross-platform file paths and OpenCV (`cv2`) to read `.ppm` image files. I converted these images into NumPy arrays with the correct shape and format that TensorFlow can process efficiently.

The `get_model` function is responsible for building the convolutional neural network using TensorFlow’s Keras API. This involved learning how to define layers, specify input shapes, and configure parameters such as filter sizes, activation functions, and dropout rates. Through extensive reading of documentation and revisiting class materials, I was able to design and compile a functional model. Working initially with a smaller sample dataset was especially helpful for quicker experimentation and debugging during development.


## Process Explanation / Evaluation

During my experimentation with building a convolutional neural network for traffic sign classification, I encountered a challenging but rewarding learning process. Initially, my limited experience with neural networks and unfamiliarity with TensorFlow/Keras made it difficult to design an effective model. However, through research and trial and error, I gradually improved the architecture and gained a deeper understanding of key concepts.

One fundamental insight was the importance of convolutional layers combined with pooling layers to reduce image size while extracting meaningful features. For instance, starting with a simple model consisting of a single convolutional layer (32 filters, 3x3 kernel), a 2x2 max pooling layer, and dropout (rate 0.5), the model achieved a modest accuracy of about 65%. The choice of 3x3 kernels was deliberate, based on popular architectures like VGG and ResNet, which use small kernels to capture fine details such as edges and textures effectively.

Removing pooling layers significantly increased the computational load without a proportional gain in accuracy. Conversely, experimenting with a 2x2 pooling layer drastically reduced training time and improved accuracy, highlighting the trade-off between model complexity, computation, and performance. Based on these results, I settled on using 2x2 pooling layers throughout the network.

Adding more convolutional layers progressively improved accuracy. Introducing a second convolutional layer with 64 filters increased accuracy to around 96%, and a third layer with 128 filters further improved it to 97%. These gains came with increased training time but were worthwhile for the performance boost.

I also experimented with dropout layers to prevent overfitting. Removing dropout caused accuracy to drop, confirming its importance in regularization. Interestingly, when I removed the fully connected hidden Dense layer at the end but kept all convolutional layers intact, the model's accuracy did not significantly degrade. This suggested that the convolutional layers’ feature extraction was the most critical part of the model.

In summary, my experiments showed that the right combination of convolutional filters and pooling is crucial for good performance. More convolutional layers generally improved accuracy but required more training time. Pooling layers help balance computational efficiency and accuracy. The 3x3 kernel size proved effective and consistent with established CNN architectures. Removing hidden Dense layers is possible without severely affecting accuracy, indicating that convolutional features were sufficient for this task.




