# Drowsiness detection with OpenCV and CNN
The poject is build to detect early signs of fatigue for drivers based on their eye movement
using a real-time video stream and playing an alarm when the divers appears to be drowsy.

### Dependencies

The project uses cv2, keras and tensoflow. All the required dependencies can be found in 
[requirements.txt](https://github.com/ioanachelaru/Drowsiness-detection-system/blob/master/requirements.txt).

### Algorithm

The face is detected using haar-like features since is a quick and 
efficient enough method and the eyes are detected 
using dlib's shape_predictor_68_face_landmarks.dat 
because is a more effective method than the previous one.
<br><br>If both eyes are detected they are passed to a convolutional 
neural network build and trained for their classification.
<br><br>The CNN has been trained with only left eyes, that way creating 
a binary classifier since we're not interested in differentiating 
the right from the left eye. For that reason, before sending the 
segment containing the right eye detected, a flip operation is applied 
over it so it simulates the left eye distribution. 