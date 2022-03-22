# Assemblies-Simulations

How to run save_assembly_gif.py


python3 save_assembly_gif.py -param_name <param_val>
Parameters:
-n      number of nodes (default 10000)
-k      size of cap     (default 100)
-beta   plasticity param (default 0.0)
-T      number of steps (default 12)
-sigma  param for gaussian (default 0.01)


-save or -s   name of .gif file to save to (if not specified, it displays the gif)
-speed or -i  number of ms to display each frame (default 200)


example:

python3 save_assembly_gif.py -n 10000 -k 100 -s "test_gif.gif" -i 150
