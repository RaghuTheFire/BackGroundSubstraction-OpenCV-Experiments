# BackGroundSubstraction-OpenCV-Experiments
Background subtraction is a common technique used in computer vision to separate foreground objects from the background in a video stream. 
OpenCV provides several methods to perform background subtraction, one of the popular methods being the BackgroundSubtractorMOG2 algorithm.
 Here are some common methods available in OpenCV:

1. BackgroundSubtractorMOG: This method implements the Gaussian Mixture-based Background/Foreground Segmentation Algorithm. It is based on the paper by Z.Zivkovic: "Improved adaptive Gausian mixture model for background subtraction". This method is relatively fast but may not work well with varying lighting conditions.

2. BackgroundSubtractorMOG2: This is an improved version of the MOG method. It utilizes a Gaussian Mixture-based algorithm, but it adapts over time and maintains a history of observations. This method often performs better in varying lighting conditions and is widely used.

3. BackgroundSubtractorKNN: This method is a variant of the MOG2 algorithm but uses a K-nearest neighbors approach instead of a Gaussian mixture model. It can be more accurate in some scenarios, especially when dealing with noise or rapid changes in the background.

4. BackgroundSubtractorLSBP: Local SVD Binary Pattern (LSBP) is a local texture-based method for background subtraction. It uses local binary pattern (LBP) and singular value decomposition (SVD) to model the background texture and subtract it from the current frame.

5. BackgroundSubtractorCNT: Count-based Non-parametric Background Subtraction. This method is particularly useful for scenarios with highly dynamic backgrounds or scenes with frequent illumination changes.



# Reference 
https://www.simonwenkel.com/notes/software_libraries/opencv/background-subtraction-using-opencv.html
