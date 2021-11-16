# -*- coding: utf-8 -*-
#
# John C. Thomas 2021 gpSTS

################################################################
######Run gpSTS (built off gpCAM) with LabVIEW interface#######
################################################################
import sys
from gpsts.Classification.main import main_post



print("\n\
Copyright (C) 2021 Version\n\
\n\
gpSTS is free software: you can redistribute it and/or modify\n\
it under the terms of the GNU General Public License as published by\n\
the Free Software Foundation, either version 3 of the License, or\n\
(at your option) any later version.")
test = input("Execute by entering a 1 or quit otherwise: ")
if test != '1':
    print("gpSTS is exiting.")
    sys.exit(1)

if len(sys.argv) == 1:
    main_post()
