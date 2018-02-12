#!/usr/bin/env python
import lakota_driver
import sys, argparse

description= 'Lakota: a Dakota postprocessor, version 0\n' \
	     'Linear regression and hypothesis testing on Dakota samples'
       	
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--input','-i')

if __name__ == "__main__":

	if len(sys.argv) == 1: #code was called without input file 
		parser.print_help()
		sys.exit(0)
	
	args = parser.parse_args() #read input file
	
	driver = lakota_driver.lakota_driver() #create lakota_driver. this will run all of the data reading and postprocessing

	lakota_params = driver.get_inputs(args.input) #feed input file to lakota_driver

	flag = driver.execute(lakota_params) #run the desired postprocessing
	
	if flag == 0:
		print("Lakota executed successfully")
	if flag == 1:
		print("Lakota failed somewhere.")
	
