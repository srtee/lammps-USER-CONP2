echo '100' > nsteps

mpirun -np 8 lmp -i example_input_v0 > output_v0
