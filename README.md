
 Feature points estimation
 =========================
   
  Build with
  ----------
  OpenCV 4.1.0
  Windows10
  C/C++
   About The Project
   -----------------
   The program acquires three-dimensional feature points through a single module camera. 
   It receives a video as an input and estimates a baseline using the accelerometer value. 
   Match the feature points between frames and find the disparity through the matched feature points. 
   The depth value is obtained through disparity and the depth value of the next frame is optimized based on the key frame.
   
   
  Video
  ----------
   This sample video shows the project in action. After showing the input frame, the baseline value is shown.
   Output the number of feature points obtained using SIFT and match the feature points.
   Disparity is calculated through the matched feature points and the depth value is estimated.
   Plot the matched feature points and input frame at once, and finally output the accumulated 3D feature points as a ply file.
  
  ![test_gif2](https://user-images.githubusercontent.com/93419240/139624119-f2cea1b1-288f-4ea7-91b0-5ab08f389e1c.gif)

  Contact
   -----------------
   [Young-Ho Seo] - yhseo@kw.ac.kr
   
   Link: [Intelligent Computing Lab](https://sites.google.com/view/ic-lab/home)
   

   Acknowledgement
   -----------------
   This work was supported by Institute of Information & Communications Technology Planning & Evaluation (IITP)
   grant funded by the Korean government (MSIT) (No.2020-0-00192, AR Cloud, Anchor, Augmented Reality, Fog Computing, Mixed Reality).
