Description
-----------
This folder contains Sentinel-2 methane retrieval code for Varon et al. (2021):

	Varon et al. (2021) Atmos. Meas. Tech., 14, 2771–2785, 2021
	https://doi.org/10.5194/amt-14-2771-2021

The code uses HAPI python code (hapi.py) by Kochanov et al. (2016):

	R.V. Kochanov, I.E. Gordon, L.S. Rothman, P. Wcislo, C. Hill, J.S. Wilzewski,
        HITRAN Application Programming Interface (HAPI): A comprehensive approach to
        working with spectroscopic data, J. Quant. Spectrosc. Radiat. Transfer 177, 
	15-30 (2016) DOI: 10.1016/j.jqsrt.2016.03.005

	The hapi.py code can be downloaded directly from https://hitran.org/hapi/

Contents
--------
	- aux_data            : Auxiliary data directory containing
				- SUNp01_4000_to_7000.txt : Clough 2005 solar spectrum
				- ch4.dat                 : US Standard CH4 profile
				- co2.dat                 : US Standard CO2 profile
				- h2o.dat                 : US Standard H2O profile
				- pressure.dat            : US Standard pressure profile
				- temperature.dat         : US Standard temperature profile
	- hapi_data           : HAPI data directory where absorption cross-section data 
                                will be stored
	- hapi.py             : HAPI code from Kochanov et al. https://hitran.org/hapi/
	- radtran.py          : Radiative transfer code to retrieve methane columns
	- setup.py            : Setup function to populate the hapi_data/ directory
	- demonstration.ipynb : Jupyter notebook demonstrating the retrieval with a toy
				example
	- README.txt

Instructions
------------
	1. Use setup.py to setup the hapi_data directory for one or both Sentinel-2 satellites.
	   This could take an hour or more for each MSI instrument.
	2. Define an array of fractional reflectance differences from Sentinel-2 data.
	   i.e. Evaluate eq. (1) or (3) from Varon et al. (2021) for a Sentinel-2 scene.
	3. Pass the data from 2. to radtran.retrieve() along with instrument and method selections.

See demonstration.ipynb for example implementation using Sentinel-2A, the single-band/multi-pass (SBMP) 
method, and a small array of synthetic fractional reflectance data.