import numpy as np
import scipy.stats as stat



def statistics(lakota_stats,data):

	variables = lakota_stats['variables'].split()
	inputs = lakota_stats.get('inputs','').split()
	
	X = []
	Y = []
	for n in range(len(variables)):
		Y.append(data[variables[n]])
	if len(inputs) > 0:
		for n in range(len(inputs)):
			X.append(data[inputs[n]])
		X = np.array(X)
	Y = np.array(Y)

	#mean
	mean = np.mean(Y,axis=1)
	
	#variance
	variance = np.var(Y,axis=1)

	#correlations
	if len(inputs) > 0:
		corr = np.zeros((len(inputs),len(variables)))
		for m in range(len(inputs)):
			for n in range(len(variables)):
				corr[m,n] = np.corrcoef(X[m],Y[n])[0,1]

	#histogram
	hist = []
	if lakota_stats.get('histogram',None) != None:
		for n in range(len(variables)):
			bins = np.linspace(min(Y[n]),max(Y[n]),float(lakota_stats.get('histogram').get('numbins','20')))
			h,b = np.histogram(Y[n],bins=bins)
			hist.append((h,b))
	ofile = lakota_stats.get('output','stats.out')
	output(mean,variance,corr,hist,ofile,variables,inputs)

def output(mean,variance,corr,hist,ofile,variables,inputs):
	
	with open(ofile,'w') as f:
		f.write("mean of " + ' '.join(variables) + "\n")
		for n in range(len(mean)):
			f.write(str(mean[n])+ " ")
		f.write("\n\n")
		f.write("variance of " + ' '.join(variables) + "\n")
		for n in range(len(variance)):
			f.write(str(variance[n])+" ")
		f.write("\n\n")
		f.write("correlations\n" +"input "+ ' '.join(variables)+"\n")
		for n in range(len(inputs)):
			line = inputs[n]
			for m in range(len(variables)):
				line += " " + str(corr[n,m])
			f.write(line+"\n")
		
		if len(hist) > 0:
			f.write("\n")
			f.write("histograms of "+ ' '.join(variables) + "\n")
			for n in range(len(hist)):
				f.write(variables[n] + " count\n")
				for i in range(len(hist[n][0])):
					f.write(str(hist[n][1][i]) + " " + str(hist[n][0][i])+ "\n")
				f.write("\n")
