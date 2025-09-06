#!/bin/bash
################## BindCraft installation script
################## specify conda/mamba folder, and installation folder for git repositories, and whether to use mamba or $pkg_manager
# Default value for pkg_manager
pkg_manager='conda'
cuda=''
install_pyrosetta=true

# Define the short and long options
OPTIONS=p:c:n
LONGOPTIONS=pkg_manager:,cuda:,no-pyrosetta

# Parse the command-line options
PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTIONS --name "$0" -- "$@")
eval set -- "$PARSED"

# Process the command-line options
while true; do
  case "$1" in
    -p|--pkg_manager)
      pkg_manager="$2"
      shift 2
      ;;
    -c|--cuda)
      cuda="$2"
      shift 2
      ;;
    -n|--no-pyrosetta)
      install_pyrosetta=false
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      echo -e "Invalid option $1" >&2
      exit 1
      ;;
  esac
done

# Example usage of the parsed variables
echo -e "Package manager: $pkg_manager"
echo -e "CUDA: $cuda"
echo -e "Install PyRosetta: $install_pyrosetta"

############################################################################################################
############################################################################################################
################## initialisation
SECONDS=0

# set paths needed for installation and check for conda installation
install_dir=$(pwd)
CONDA_BASE=$(conda info --base 2>/dev/null) || { echo -e "Error: conda is not installed or cannot be initialised."; exit 1; }
echo -e "Conda is installed at: $CONDA_BASE"

# check if FBC env installed
FBC_ENV_PATH=$(conda env list | grep FreeBindCraft | awk '{print $2}')
if [[ -z "${FBC_ENV_PATH}" ]];then
    ### BindCraft install begin, create base environment
    echo -e "Installing FreeBindCraft environment\n"
    $pkg_manager create --name FreeBindCraft python=3.10 -y || { echo -e "Error: Failed to create FreeBindCraft conda environment"; exit 1; }
    conda env list | grep -w 'FreeBindCraft' >/dev/null 2>&1 || { echo -e "Error: Conda environment 'FreeBindCraft' does not exist after creation."; exit 1; }
else
    echo -e "BindCraft environment existed at ${FBC_ENV_PATH}"
fi

# Load newly created FreeBindCraft environment
echo -e "Loading FreeBindCraft environment\n"
FBC_ENV_PATH=$(conda env list | grep FreeBindCraft | awk '{print $2}')
source ${CONDA_BASE}/bin/activate ${FBC_ENV_PATH} || { echo -e "Error: Failed to activate the BindCraft environment."; exit 1; }
[ "$CONDA_DEFAULT_ENV" = "FreeBindCraft" ] || { echo -e "Error: The FreeBindCraft environment is not active."; exit 1; }
echo -e "BindCraft environment activated at ${FBC_ENV_PATH}"

# install required conda packages
echo -e "Installing conda requirements\n"

# Base packages (needed regardless of PyRosetta)
BASE_PACKAGES="pip pandas matplotlib numpy<2.0.0 biopython scipy pdbfixer openmm seaborn libgfortran5 tqdm jupyter ffmpeg fsspec py3dmol chex dm-haiku flax<0.10.0 dm-tree joblib ml-collections immutabledict optax"

# Pin JAX to stable version 0.6.0
echo -e "Using JAX/jaxlib version 0.6.0 for stability\n"

# Install packages with or without PyRosetta
if [ "$install_pyrosetta" = true ]; then
    echo -e "Installing with PyRosetta\n"
    if [ -n "$cuda" ]; then
        CONDA_OVERRIDE_CUDA="$cuda" $pkg_manager install \
            $BASE_PACKAGES pyrosetta "jaxlib=0.6.0=*cuda*" "jax=0.6.0" cuda-nvcc cudnn \
            -c conda-forge -c nvidia -c "https://conda.rosettacommons.org" -y \
            || { echo -e "Error: Failed to install conda packages with PyRosetta."; exit 1; }
    else
        $pkg_manager install \
            $BASE_PACKAGES pyrosetta "jaxlib=0.6.0" "jax=0.6.0" cuda-nvcc cudnn \
            -c conda-forge -c nvidia -c "https://conda.rosettacommons.org" -y \
            || { echo -e "Error: Failed to install conda packages with PyRosetta."; exit 1; }
    fi
else
    echo -e "Installing without PyRosetta\n"
    if [ -n "$cuda" ]; then
        CONDA_OVERRIDE_CUDA="$cuda" $pkg_manager install \
            $BASE_PACKAGES "jaxlib=0.6.0=*cuda*" "jax=0.6.0" cuda-nvcc cudnn \
            -c conda-forge -c nvidia -y \
            || { echo -e "Error: Failed to install conda packages without PyRosetta."; exit 1; }
    else
        $pkg_manager install \
            $BASE_PACKAGES "jaxlib=0.6.0" "jax=0.6.0" cuda-nvcc cudnn \
            -c conda-forge -c nvidia -y \
            || { echo -e "Error: Failed to install conda packages without PyRosetta."; exit 1; }
    fi
fi

# Define required packages based on installation mode
if [ "$install_pyrosetta" = true ]; then
    required_packages=(pip pandas libgfortran5 matplotlib numpy biopython scipy pdbfixer openmm seaborn tqdm jupyter ffmpeg pyrosetta fsspec py3dmol chex dm-haiku dm-tree joblib ml-collections immutabledict optax jaxlib jax cuda-nvcc cudnn)
else
    required_packages=(pip pandas libgfortran5 matplotlib numpy biopython scipy pdbfixer openmm seaborn tqdm jupyter ffmpeg fsspec py3dmol chex dm-haiku dm-tree joblib ml-collections immutabledict optax jaxlib jax cuda-nvcc cudnn)
fi
missing_packages=()

# Check each package
echo -e "Check each package\n"
ALL_CONDA_INSTALLED=$(conda list | awk '{print $1}')
for pkg in "${required_packages[@]}"; do
     echo $ALL_CONDA_INSTALLED |grep -w "$pkg" >/dev/null 2>&1 && echo " - [x] $pkg " || { missing_packages+=("$pkg") && echo " - [ ] $pkg"; }
done

# If any packages are missing, output error and exit
if [ ${#missing_packages[@]} -ne 0 ]; then
    echo -e "Error: The following packages are missing from the environment:"
    for pkg in "${missing_packages[@]}"; do
        echo -e " - $pkg"
    done
    exit 1
fi

# install ColabDesign
MISSING_COLABDESIGN=$(python -c "import colabdesign")
if [[ ! -z "$MISSING_COLABDESIGN" ]];then
    echo -e "Installing ColabDesign\n"
    pip3 install git+https://github.com/sokrypton/ColabDesign.git --no-deps || { echo -e "Error: Failed to install ColabDesign"; exit 1; }
    python -c "import colabdesign" >/dev/null 2>&1 || { echo -e "Error: colabdesign module not found after installation"; exit 1; }
else
    echo -e "Installing ColabDesign - Skipped\n"
fi

# install FreeSASA Python module
echo -e "Installing FreeSASA Python module\n"
pip3 install freesasa || { echo -e "Warning: Failed to install FreeSASA Python module via pip - FreeSASA SASA calculations will fall back to Biopython"; }
python -c "import freesasa" >/dev/null 2>&1 && echo -e "FreeSASA Python module installed successfully" || echo -e "Warning: FreeSASA Python module not available - using Biopython fallback for SASA"

# AlphaFold2 weights
echo -e "Downloading AlphaFold2 model weights \n"
params_dir="${install_dir}/params"
params_file="${params_dir}/alphafold_params_2022-12-06.tar"

# download AF2 weights
mkdir -p "${params_dir}" || { echo -e "Error: Failed to create weights directory"; exit 1; }
wget -O "${params_file}" "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar" || { echo -e "Error: Failed to download AlphaFold2 weights"; exit 1; }
[ -s "${params_file}" ] || { echo -e "Error: Could not locate downloaded AlphaFold2 weights"; exit 1; }

# extract AF2 weights
tar tf "${params_file}" >/dev/null 2>&1 || { echo -e "Error: Corrupt AlphaFold2 weights download"; exit 1; }
tar -xvf "${params_file}" -C "${params_dir}" || { echo -e "Error: Failed to extract AlphaFold2weights"; exit 1; }
[ -f "${params_dir}/params_model_5_ptm.npz" ] || { echo -e "Error: Could not locate extracted AlphaFold2 weights"; exit 1; }
rm "${params_file}" || { echo -e "Warning: Failed to remove AlphaFold2 weights archive"; }

# chmod executables
echo -e "Changing permissions for executables\n"
chmod +x "${install_dir}/functions/dssp" || { echo -e "Error: Failed to chmod dssp"; exit 1; }

# chmod sc binary for shape complementarity (always needed for PyRosetta-free mode)
if [ -f "${install_dir}/functions/sc" ]; then
    chmod +x "${install_dir}/functions/sc" || { echo -e "Error: Failed to chmod sc"; exit 1; }
    echo -e "Made sc binary executable for shape complementarity calculations"
else
    echo -e "Warning: sc binary not found at ${install_dir}/functions/sc - shape complementarity will use placeholder values"
fi

# Only setup DAlphaBall.gcc if installing PyRosetta
if [ "$install_pyrosetta" = true ]; then
    chmod +x "${install_dir}/functions/DAlphaBall.gcc" || { echo -e "Error: Failed to chmod DAlphaBall.gcc"; exit 1; }
else
    echo -e "Skipping DAlphaBall.gcc setup as PyRosetta is not being installed"
fi

# finish
conda deactivate
echo -e "FreeBindCraft environment set up\n"

############################################################################################################
############################################################################################################
################## cleanup
echo -e "Cleaning up ${pkg_manager} temporary files to save space\n"
$pkg_manager clean -a -y
echo -e "$pkg_manager cleaned up\n"

################## finish script
t=$SECONDS 
echo -e "Successfully finished FreeBindCraft installation!\n"
echo -e "Activate environment using command: \"$pkg_manager activate FreeBindCraft\""
echo -e "\n"
echo -e "Installation took $(($t / 3600)) hours, $((($t / 60) % 60)) minutes and $(($t % 60)) seconds."