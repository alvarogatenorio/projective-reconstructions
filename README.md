To execute the file "reco.py" OpenCV must be installed with the option
-D OPENCV_ENABLE_NONFREE=ON
in order to use the SIFT/SURF non-free algorithms.
To do this correctly in Ubuntu one can follow the guide given by
https://www.pyimagesearch.com/2018/08/15/how-to-install-opencv-4-on-ubuntu/

The pptk module is also required for a pop-up to appear and show a projective 3D reconstruction of the scene 

Then just type in the console
python reco.py ./img/image1.jpg ./img/image2.jpg 500
Being the last number the Hessian threshold of the SURF algorithm

Two images will be returned "keys1.png" and "keys2.png" with the interest points detected in each image,
another image "matches.png" will be returned showing the matches found between images, finally, two more
images "epilines1.png" and "epilines2.png" will show some epilines in both images.

In order to visualize the results of the "gaussiana.ipynb" SageMath must be installed
http://www.sagemath.org/

To execute the other files, the only module required is numpy

Then just type in the console

python ./other/integral.py ../img/image.jpg
The integral image of "image.jpg" will be returned as "integral.png"

python ./other/mirror.py ../img/image.jpg
The mirror-extended image of "image.jpg" will be returned as "mirror.png"



