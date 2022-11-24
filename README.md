# FTIR-EBSD
Combining FTIR-EBSD data. 2022-11-24

Steps:
1) Fitting the 3d Transmission shape to obtain the Euler angles using a function that takes as inputs the object T (already in cartesian coordinates, n x n x 3) and Euler angles
2) Make horizontal sections
3) Repeat (1) with the horizontal sections
