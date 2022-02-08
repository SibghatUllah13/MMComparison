# An Empirical Investigation on (Surrogate) Modeling Techniques for Robust Optimization 
Contains the code for empirically investigating the practicality of (surrogate) modeling techniques for finding robust solutions.
Robust solutions are solutions that are immune to the uncertainty/noise in the decision/search variables.
For finding robust solutions, the conceptual framework of one-shot optimization is utilized (as opposed to sequential model-based optimization).

# Introduction
This code is based on our paper, titled [An Empirical Comparison of Meta-Modeling Techniques for Robust Design Optimization](https://ieeexplore.ieee.org/abstract/document/9002805) (Ullah, Wang, Menzel, Sendhoff & Bäck, 2019), and can be used to reproduce
the experimental setup and results mentioned in the paper. The code is produced in Python 3.0. The main packages utilized in this code are presented in the next section which deals with technical requirements. 

The code is present in the main directory as well as other sub-directories. Within the main directory, the file `RBFN.py` contains the code to implement
Radial Basis Function Network (RBFN), which is one of the most popular surrogate modeling techniques. This file is to be included/imported wherever the implementation of
RBFN model is required. Similar to `RBFN.py`, we also have `Utils.py` script within the main directory which contains the basic utility functions.
Like `RBFN.py`, this file must also be imported wherever the utility of the methods inside it is needed.

There are three main directories within the main folder, which are titled `Accuracy`, `Hyper_Parameter_Optimization`, and `Results Compilation` respectively.
The first of these, namely `Accuracy` contains three further sub-directories, which are titled `NoiseLevel1`, `NoiseLevel2`, and `NoiseLevel3` respectively.
As the name suggests, these directories contain the code for a particular choice of noise level (the scale of uncertainty).
Within each of these directories, we further come up against six sub-directories which represent the test problem at hand.
If we further explore, these directories contain two jupyter notebooks, namely `Generate_Data_Sets.ipynb` and `Final_Comparison.ipynb`.
While the former contains the methods and routines for generating the training and testing data sets, the latter deals with constructing and 
appraising the surrogate models. Outside, the folder `Hyper_Parameter_Optimization` six sub-folders which represent the choice of modeling techniques.
Each of these sub-folders further contains a jupyter notebook, titled `*_Hyper.ipynb`, where `*` serves as the choice of modeling technique, e.g., Kriging, Random Forest.
This jupyter notebook contains the code for hyper parameter optimization for the chosen modeling technique.
Lastly, the folder `Results Compilation` contains several sub-folders which are named after the test problem considered (apart from folder `Graphs` which simply contains
all the plots). Each of those sub-folders contain the file `Graph.ipynb` which produces the figures and plots for the results achieved.
In the following, we describe the technical requirements as well the instructions to run the code in a sequential manner.

Requirements

Python >= 3.5 


# MMComparison
The repository contains the main experimental setup including hyper parameter optimization for the comparison of meta-modeling techniques. 
To run the code, first choose the right criteria (e.g. Noise Level, Function etc.) in the Accuracy folder and go to the desired experimental setup.
Run the Generate_Data_Sets.ipynb for the desired experimental evaluation and generate the data sets of input and output pairs.
Then go to the Hyper_Parameter_Optimization and run the code for each Meta-Model technique e.g. Kriging etc. one by one for all sample points.
The best Hyper_Parameters for the corresponding Meta Model and Sample points will be displayed. Manually select these hyper parameters and save them in the final comparison.ipynb file of your desired experimental evaluation.
Run Final_Comparison.ipynb , it will generate the result files.
To achieve the graphics, go to the results compilation folder and put the output files in your desired function evaluation and run the file
Graph.ipynb
##########################################################################
