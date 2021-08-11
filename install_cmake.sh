#!/bin/bash

# This script really shouldn't be run in a pipe
set -o pipefail

# First, check that the user has specified where to install the files
if [[ -z "${LAMMPS_PREFIX}" ]]; then
  echo "Error: LAMMPS install directory not found. Set LAMMPS_PREFIX environment variable and retry."
  exit 1
else
  CONP_DOWNLOAD_DIR=$(pwd -P)
  CONP_SRC_DIR="${LAMMPS_PREFIX}"/src/USER-CONP2
  INTEL_DIR="${LAMMPS_PREFIX}"/src/USER-INTEL

  echo "Creating CONP source directory..."
  mkdir "${CONP_SRC_DIR}"
  echo "Done"
fi

# Copy the source files to the source directory
# Do the base files first
echo "Copying base CONP files..."
cp -v compute_potential_atom.* fix_conp.* fix_conq.* \
km_ewald.* km_ewald_split.* kspacemodule.h pppm_conp.* incl_pppm_intel_templates.cpp \
Install.sh \
${CONP_SRC_DIR} || exit 1
echo "Done"
# And the Intel accelerated styles
echo "Copying Intel-accelerated CONP files..."
cp -v pppm_conp_intel.* ${INTEL_DIR} || exit 1
echo "Done"

# Now we need to patch the LAMMPS CMakeLists.txt to detect and build the CONP2 files
echo "Moving to ${LAMMPS_PREFIX} and patching CMakeLists.txt..."
cd ${LAMMPS_PREFIX}
git apply ${CONP_DOWNLOAD_DIR}/patchfile || exit 1
echo "Done"
cd ${CONP_DOWNLOAD_DIR}
echo "================="
echo "Finished copying USER-CONP2 files." 
echo "You can now build LAMMPS with CMake by setting -D PKG_USER-CONP2=on"
