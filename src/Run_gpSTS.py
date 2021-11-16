# -*- coding: utf-8 -*-
#
# John C. Thomas 2021 gpSTS

################################################################
######Run gpSTS (built off gpCAM) with LabVIEW interface#######
################################################################
import sys
import Config
import gpcam
gpcam.global_config = Config
from gpcam.main import main
from gpsts.Classification.data_collect import InsertData
from pathlib import Path

target_path = Path.cwd() / "data"
dstore_path = target_path.parents[1]

data_path = target_path / "new_experiment_result" / "result.txt"
cmd_path = target_path / "new_experiment_command" / "command.txt"
im_path = target_path / "impath" / "current.png"
current_path = dstore_path / "data" / "current_data" / "Data_model_1.npy"
historical_path = dstore_path / "data" / "historic_data" / "Data_model_1.npy"
data_path.parent.mkdir(parents=True, exist_ok=True)
cmd_path.parent.mkdir(parents=True, exist_ok=True)
im_path.parent.mkdir(parents=True, exist_ok=True)
current_path.parent.mkdir(parents=True, exist_ok=True)
historical_path.parent.mkdir(parents=True, exist_ok=True)

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
    main()
