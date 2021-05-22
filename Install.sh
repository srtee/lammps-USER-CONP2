# Install/unInstall package files in LAMMPS
# mode = 0/1/2 for uninstall/install/update

# this is default Install.sh for all packages
# if package has an auxiliary library or a file with a dependency,
# then package dir has its own customized Install.sh

mode=$1

# enforce using portable C locale
LC_ALL=C
export LC_ALL

# arg1 = file, arg2 = file it depends on

action () {
  if (test $mode = 0) then
    rm -f ../$1
  elif (! cmp -s $1 ../$1) then
    if (test -z "$2" || test -e ../$2) then
      cp $1 ..
      if (test $mode = 2) then
        echo "  updating src/$1"
      fi
    fi
  elif (test -n "$2") then
    if (test ! -e ../$2) then
      rm -f ../$1
    fi
  fi
}

# all package files with dependencies

action kspacemodule.h
action km_ewald.cpp 
action km_ewald.h
action km_ewald_split.cpp
action km_ewald_split.h
action compute_potential_atom.cpp pppm.h
action compute_potential_atom.h pppm.h
action pppm_conp.cpp pppm.h
action pppm_conp.h pppm.h
action pppm_conp_intel.cpp pppm_intel.cpp
action pppm_conp_intel.h pppm_intel.h
action fix_conp.cpp
action fix_conp.h
