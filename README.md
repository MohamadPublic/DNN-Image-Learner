# Custom Drawing Neural Network

This is a very simple project that allows you to train neural networks on any image you decide to draw. The coordinates in the image correspond to some pixel intensity, this is the target to be learned. The neural network uses skip connections to perserve the structure of the image as well as the size in aim to reproduce the image as accurately as it can. Captures of the learned image are taken sequentially and once all epochs finish, the user gets an mp4 file which shows a time lapse of the learning of the image.

# Requirements
- python3
- pip3

# Setup
Install dependencies with:

`pip install -r requirements.txt`

Note that this build is stable on python 3.9.18. Other versions of python may require different versions of the packages specified in `requirements.txt`.

After setup, run `python main.py`. This will prompt you to draw an image and then you can sit back and enjoy. To monitor the frame-by-frame progress, head to the frames folder. At the end, watch video.mp4!

This code is modified and forked from MaxRobinsonTheGreat. For more viewing material, watch his spectacular video: https://www.youtube.com/watch?v=TkwXa7Cvfr8
