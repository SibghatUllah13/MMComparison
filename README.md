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
