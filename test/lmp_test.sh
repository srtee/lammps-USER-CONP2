echo '250' > nsteps

tests=(v0 v1 v2 q2 v3)

for t in ${tests[*]}
do
  mpirun -np 8 lmp -i example_input_$t | tee output_$t
done

for t in ${tests[*]}
do
  LASTLINE=$(tail -n1 'output_'"$t")
  echo "${t} ${LASTLINE}"
done
