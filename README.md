# Realtime digit recognizer
## Introduction
- A simple digit recognizer convnet trained on mnist data set with augmentation using keras. The model will perform realtime prediction based on camera input. To improve prediction stability and reduce false positive, a prediction probability of at least 90% is required, else a 'nil' is displayed. A deque is also used so that the mode prediction out of the last 20 predictions is displayed.

## Instructions
1. Install dependencies in your virtual environment
``` 
pip install -r requirements.txt
```

2. To train a model, edit the number of epochs in the script and run `recognizer_keras_train.py`. The output will be a .h5 file.

3. Run `recognizer_keras.py`. Make sure the correct .h5 file is loaded within the code
