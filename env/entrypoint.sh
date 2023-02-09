#!/bin/bash

whoami

echo "Starting with UID : ${NB_UID}, GID: ${NB_GID}"
usermod -u ${NB_UID} user
groupmod -g ${NB_GID} usergroup

if [[ "${GRANT_SUDO}" == "1" || "${GRANT_SUDO}" == 'yes' ]]; then 
  echo "Granting user sudo access" 
  echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/usergroup 
fi 

su user << EOF
. /opt/conda/etc/profile.d/conda.sh
conda activate
eval "$@"
