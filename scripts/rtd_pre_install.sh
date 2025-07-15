#!/bin/bash

# Stop and exit on error
set -euox pipefail

# Check for required tools
java -version

# This folder is on PATH and does not require sudo
# Download latest plantuml.jar from github
curl -o ${READTHEDOCS_VIRTUALENV_PATH}/bin/plantuml.jar -L https://github.com/plantuml/plantuml/releases/latest/download/plantuml.jar
# Create an executable script for plantuml
printf '#!/bin/bash\nexec java -Djava.awt.headless=true -jar ${READTHEDOCS_VIRTUALENV_PATH}/bin/plantuml.jar "$@"' > ${READTHEDOCS_VIRTUALENV_PATH}/bin/plantuml
chmod +x ${READTHEDOCS_VIRTUALENV_PATH}/bin/plantuml

# Check plantuml version
plantuml -version
