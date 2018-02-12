import yaml, sys
import numpy as np
import linear_regression
import statistics

class lakota_driver:

	def __init__(self):
		self.data = {}
		
	def load_data(self,dakota_tabular):
		data = np.genfromtxt(dakota_tabular,names=True)
	
		for v in data.dtype.names:
			self.data[v] = data[v]
		#given struct of dakota features
		#tabular data file
		#features considered for regression
		#output variables
		#return organized data
		pass

	def get_inputs(self,input_file):
		with open(input_file,'r') as f:
			params = yaml.load(f)

		#add some user validation here to make sure necessary things are present
		if not "dakota_tabular" in params.keys():
			sys.stderr("no dakota_tabular option specified in input file, exiting with extreme prejudice\n")
			sys.exit()

		return params

	def execute(self,lakota_params):
		#get data from dakota file
		self.load_data(lakota_params['dakota_tabular'])

		#execute each desired postprocessor
			#linear regression for each output vs all inputs
				#list of regressor variables
				#list of response variables
				#regression method (QR, SVD)
				#output options:
					#slopes and intercepts
					#t statistics and p-values for each regressor
					#ANOVA for entire regression
					#confidence intervals for the entire regression
					#studentized residuals
			#stats outputs
				#means, variances, correlations between each variable and the inputs, histograms
		try:
			if "linear_regression" in lakota_params.keys():
				linear_regression.linear_regression(lakota_params['linear_regression'],self.data)
			if "statistics" in lakota_params.keys():
				statistics.statistics(lakota_params['statistics'],self.data)	
			return 0
		except:
			return 1
