This is a dm_control (mujoco) project with mesh assets. The meshes 
were done in Rhinoceros 3D, and the .3dmbak files are for this 
program. Training is the revamped RPO from CleanRL. The new 
version is made using dm_control's logic, while the old is 
xml-based.

Python 3.10.20.  I believe Python above 3.12 won't work because of dm_control.
Probably need to run pip install mujoco before requirements.txt.
Tested only in Ubuntu.