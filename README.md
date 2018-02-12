# Lakota
Dakota postprocessing support

This program is called via 
python lakota.py -i in.yaml
or
./lakota.py -i in.yaml
where "in.yaml" is the name of whatever input file you use

Lakota assumes you have a working numpy and scipy installation that is available with your default python.
Anaconda works. Not sure about other packages.

At the moment, Lakota mainly does things for processing samples from LHS or PCE runs.
Theoretically, you could do the same thing with any data set (experiments, other sampling routines, non-Dakota)
provided that the input file is in the proper format.

Speaking of which, input file format is column based text, first line header
var1 var2 var3 varN
num11 num12 num13 num1N
num21 num22 num23 num2N
...

Capabilities are currently linear regression with hypothesis tests, and simple statistics postprocessing.
