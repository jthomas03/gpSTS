# -*- coding: utf-8 -*-
#
# John C. Thomas 2021 gpSTS

################################################################
######Run gpSTS (built off gpCAM) with LabVIEW interface#######
################################################################
import sys
from gpsts.Classification.cnn_main import main_cnn, main_class
import Config
import Config as conf


print("\n\
Copyright (C) 2021 Version\n\
\n\
gpSTS is free software: you can redistribute it and/or modify\n\
it under the terms of the GNU General Public License as published by\n\
the Free Software Foundation, either version 3 of the License, or\n\
(at your option) any later version.")
test = input("Execute training by entering a 1, classify data by entering a 2, or exit otherwise: ")
testcorr = ['1','2']
if test not in testcorr:
    print("gpSTS is exiting.")
    sys.exit(1)

if test == '1':
    main_cnn()

if test == '2':
    main_class()
    
