# Naive-Bayes-Document-Classifier
A classifier model to predict the news group of a given article

The code is written in such a way that it is applicable to the famous 20 Newsgroup dataset. 

README
Prem Sai Kumar Reddy Gangana 
(psreddy@unm.edu)
03/20/2015

Usage:
My Program is written on Python v3.4.3 (64 bit) on the Integration Development Environment (IDE) IDLE v3.4.3 over operating system Windows8.1. 

Following packages are imported to aid at different stages in programming and in code.

matplotlib==1.4.3
numpy==1.9.2
scikit-learn==0.15.2
scipy==0.15.1

To Change the parameters, training and test data set, please change the filenames at the following lines.
	vocablength=61188 : 10
	nooflabels=20 : 11
	train.data : 17
	train.label: 24
	test.data  : 58
	test.label : 65

Execution:
There is no need for makefile as the code is written in a single file (Project2_NBC.py). 
My whole algorithm of classsifier is written in a single Python file (.py) . 
It is executed in windows by double clicking on the file or use IDLE environment to open the file and press F5 or execute it from command line
In Linux/UNIX, execute the command : python3.4 Project2_NBC.py

Note: As the classifier is developed in the latest version of python (3.4.3), it is advisable to run the program in the latest version only due to inability of backward comaptibility.

