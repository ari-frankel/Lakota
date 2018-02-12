#!/usr/bin/env python
import lakota_driver
import sys, argparse

description= 'Lakota: a Dakota postprocessor, version 0\n' \
	     'Linear regression and hypothesis testing on Dakota samples'
       	
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--input','-i')
parser.add_argument('--output','-o')

if __name__ == "__main__":
	
	args = parser.parse_args()
	
	driver = lakota_driver.lakota_driver()	

	lakota_params = driver.get_inputs(args.input)

	flag = driver.execute(lakota_params)
	
