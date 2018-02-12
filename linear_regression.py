import numpy as np
import sys
import scipy.stats as stats

#references:
#wikipedia and the internet
#Applied Linear Regression, Sanford Weisberg

#if i were smart, I'd rewrite this code to do the linear algebra part just once
#and reuse the decomposition for each output Y. oh well.

#matrix decomposition methods for least squares
#X beta = Y, X is inputs, Y is outputs
#we want beta (the regression coeffs), 
#(X^T X)^-1 for computing covariance matrix of beta, 
#and H=X (X^T X)^-1 X.T (the hat matrix) for computing residuals	

def SVD(X,Y): #SVD method, ideal for ill-conditioned or rank-deficient systems
	U, s, V = np.linalg.svd(X)
	#numpy returns X = U*S*V instead of U*S*V.T, and s is vector, not matrix, and flipped upside down	
	#compute pseudo-inverse
	shape = np.shape(X)
	N = shape[0]
	p = shape[1]
	S = np.zeros((N,p))
	Sp = np.zeros((p,N))
	for i in range(min(N,p)): #X is underdetermined, rank is less than or equal to min(N,p)
		S[i,i] = s[i]
		if s[i] != 0:
			Sp[i,i] = 1./s[i]
	V = V.T #so X = U*S*V.T now
	#USV^T beta = Y
	#beta = V*pinv(S)*U.T*Y (pinv = pseudo-inverse)
	beta = V.dot(Sp.dot(U.T.dot(Y)))
	#XTXinv = V.dot(np.linalg.pinv(S.T.dot(S)).dot(V.T))	
	XTXinv = np.linalg.pinv(V.dot(S.T.dot(S.dot(V.T))))
	r = np.linalg.matrix_rank(X)
	Ur = U[:,0:r] #the singular vectors corresponding to the non-rank-deficient part of X
	H = Ur.dot(Ur.T)
	return beta, XTXinv, H

def QR(X,Y): #QR method, faster in general, asymptotically same speed as SVD with many data points
	#X = QR, where Q.T*Q = I (Q is orthonormal), R is upper triangular
	Q, R = np.linalg.qr(X)
	#Q*R*beta = Y --> R*beta = Q.T*Y --> beta = R^-1*Q.T*Y
	beta = np.linalg.solve(R,Q.T.dot(Y)) #this should be fast
	#X = QR --> X^T X = R.T*R --> (X^T X)^-1 = (R.T*R)^-1
	XTXinv = np.linalg.solve(R.T.dot(R),np.eye(len(R))) #this might be a bit slow
	#H = Q R (R.T Q.T Q R)^-1 R.T Q.T = Q R R^-1 R.T^-1 R.T Q.T = Q Q.T (assuming R^-1 exists. if not, use SVD!)
	H = Q.dot(Q.T)

	return beta, XTXinv, H


def regress(X,Y,method):

	if method.lower() == 'svd':
		beta, XTXinv, H = SVD(X,Y)
	elif method.lower() == 'qr':
		beta, XTXinv, H = QR(X,Y)
	else:
		print("unknown linear regression method " + method + ", reverting to QR\n")
		beta, XTXinv, H = QR(X,Y)
	#if we want, can add Cholesky and Gauss eliminiation, but there's no reason to
	return beta, XTXinv, H

def run_regress(X,Y,method):

	n = np.shape(X)[0]
	p = np.shape(X)[1] #num regressors, including intercept
	beta, XTXinv, H = regress(X,Y,method)

	#confidence intervals on intercept and slopes, and t and p values for each regressor
	Yhat = H.dot(Y)
	residuals = Y-Yhat
	RSS = np.sum(residuals**2)
	variance = RSS/(n-p)	
	stud_res = residuals/np.sqrt(variance*(1-np.diag(H)))

	#covariance matrix of regressors: Var(beta|X) = sigma2*(X.T X)^-1
	covbeta = variance*XTXinv #note that XTXinv plays the role of something like "1/N" in terms of hypothesis testing
		
	conf_int = stats.t.ppf(0.995,n-p)*np.sqrt(np.diag(covbeta)) #99% confidence interval on each regressor
	t = abs(beta/np.sqrt(np.diag(covbeta))) #t scores
	Pt = 2*(1-stats.t.cdf(t,n-p)) #two-sided p-values
	
	SYY = sum((Y-np.mean(Y))**2)
	
	F = ((SYY-RSS)/(p-1))/variance #F score of entire regression (null: Y = beta_0, alternate: Y = X*beta)
	PF = 1-stats.f.cdf(F,p-1,n-p) #corresponding p-value

	R2 = 1-RSS/SYY

	return beta, Yhat, stud_res, R2, conf_int, t, Pt, F, PF

def linear_regression(lakota_linear,data):

	outfile = lakota_linear.get('output','lin.out')
	fitted_out = lakota_linear.get('fitted_output')
	if fitted_out != None:	
		ffit = open(fitted_out,'w')	
	dependents = lakota_linear['dependent'].split()
	independents = lakota_linear['independent'].split()
	X = []
	try:
		for n in range(len(independents)):
			X.append(data[independents[n]])
	except:
		sys.stderr("variable error, could not find " + independents[n])
		sys.exit()
	X = np.array(X).T

	mode = lakota_linear.get('mode','multiple')
	#append column of ones to X
	numpts = np.shape(X)[0]
	ones = np.ones(numpts)
	Xa = np.concatenate((np.array([ones]).T,X),axis=1)

	#get regressors and hat matrix
	num_responses = len(dependents)
	results = []
	f = open(outfile,'w')
	method = lakota_linear.get('method','qr')
	stud_res_bool = False
	if lakota_linear.get('report_studentized',False) == True:
		stud_res_bool = True
	if mode.lower() == 'multiple': #multiple linear regression	
		for n in range(num_responses):
			Y = data[dependents[n]]	
			beta, Yhat, stud_res, R2, conf_int, t, Pt, F, PF = run_regress(Xa,Y,method=method)
			output(dependents[n],independents,beta,conf_int,R2,t,Pt,F,PF,f)
			if fitted_out != None:
				fitted_output(dependents[n],independents,Xa,Y,Yhat,stud_res,stud_res_bool,ffit)
	elif mode.lower() == 'oat': #one-at-a-time
		for m in range(len(independents)):
			Xf = np.concatenate((np.array([ones]).T,X[:,m]),axis=1)
			for n in range(num_responses):
				Y = data[dependents[n]]
				beta, stud_res, Yhat, R2, conf_int, t, Pt, F, PF = run_regress(Xf,Y,method=method)
				output(dependents[n],independents[m],beta,conf_int,R2,t,Pt,F,PF,f)
			if fitted_out != None:
				fitted_output(dependents[n],independents,Xa,Y,Yhat,stud_res,stud_res_bool,ffit)
	f.close()
	if fitted_out != None:
		ffit.close()
	else:
		sys.stderr("unknown mode " + mode + ", please pick multiple or OAT")
		sys.exit()

def output(dep,indep,beta,conf_int,R2,t,Pt,F,PF,f):

	f.write("linear regression of " + dep + " against intercept plus " + ', '.join(indep) + "\n")
	f.write("F score of whole regression is " + str(F) + " with corresponding p-value of " + str(PF) + ", R2 = " + str(R2) + "\n")
	f.write("regressor b0")
	for n in range(len(indep)):
		f.write(" " + indep[n])
	f.write("\n")
	f.write("coefficient")
	for n in range(len(beta)):
		f.write(" " + str(beta[n]))
	f.write("\n")
	f.write("+-99%_confidence_interval")
	for n in range(len(conf_int)):
		f.write(" " + str(conf_int[n]))
	f.write("\n")
	f.write("t-score")
	for n in range(len(t)):
		f.write(" " + str(t[n]))
	f.write("\n")
	f.write("p-value")
	for n in range(len(Pt)):
		f.write(" " + str(Pt[n]))
	f.write("\n\n")


def fitted_output(dep,indep,X,Y,Yhat,stud_res,stud_res_bool,ofile):
	header = ' '.join(indep) + " " + dep + " " + dep+"_fit"
	n,p = np.shape(X)
	if stud_res_bool:
		header += " t_residual p_residual"
		p = 2*(1-stats.t.cdf(abs(stud_res),n-p))
	ofile.write(header + "\n")	

	shape = np.shape(X)
	for n in range(shape[0]):
		line = ""
		if shape[1]>2:#X includes column of ones, shape[1]>2 if a multiple linear regression was performed
			for s in range(shape[1]-1):
				line += str(X[n,s+1]) + " "
		else: #a simple linear regression 
			line += str(X[n,1]) + " "
		line += str(Y[n]) + " " + str(Yhat[n]) 
		if stud_res_bool:
			line += " " +str(stud_res[n])+ " " + str(p[n])
		ofile.write(line+"\n")
	ofile.write("\n")
