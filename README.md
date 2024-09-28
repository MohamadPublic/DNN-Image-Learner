# Custom Drawing Neural Network

This is a very simple project that allows you to train neural networks on any image you decide to draw. The coordinates in the image correspond to some pixel intensity, this is the target to be learned. The neural network uses skip connections to perserve the structure of the image as well as the size preservation in aim to reproduce the image as accurately as it can.

Captures of the learned image are taken sequentially and once all epochs finish, the user gets an mp4 file which shows a time lapse of the learning of the image.

# Requirements
- python3
- pip3

# Setup
Install dependencies with:

`pip install -r requirements.txt`

Note that this build is stable on python 3.9.18. Other versions of python may require different versions of the packages specified in `requirements.txt`.

After setup, run `python main.py`. This will prompt you to draw an image and then you can sit back and enjoy. To monitor the frame-by-frame progress, head to the frames folder. At the end, watch video.mp4!

# Example Run
The following showcases an example run where a 1000x800px image is drawn by the user and passed into the neural network as input. An example frame is shown during the training process. Finally, the video timelapse of 20 epochs worth of training [200 frames] is shown.
![input](https://github.com/user-attachments/assets/b321b6ec-489b-40c9-af02-f287bda727cd)
![download](https://github.com/user-attachments/assets/762e7781-7559-4053-a09a-4eca89469c4c)



https://github.com/user-attachments/assets/604cc6fa-ad63-4265-ac57-dc6c05185059



# Credits
This code is modified and forked from MaxRobinsonTheGreat. For more viewing material, watch his spectacular video: https://www.youtube.com/watch?v=TkwXa7Cvfr8
