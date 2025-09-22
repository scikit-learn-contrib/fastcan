#!/bin/bash

# Stop and exit on error
set -euox pipefail

# Check for required tools
java -version
dot -V

# Download latest plantuml.jar from github
mkdir -p "${HOME}/.local/bin"
export PATH="${HOME}/.local/bin:$PATH" # Add it to PATH
curl -o "${HOME}/.local/bin/plantuml.jar" -L https://github.com/plantuml/plantuml/releases/latest/download/plantuml.jar
# Create an executable script for plantuml
printf '#!/bin/bash\nexec java -Djava.awt.headless=true -jar "${HOME}/.local/bin/plantuml.jar" "$@"\n' > "${HOME}/.local/bin/plantuml"
chmod +x "${HOME}/.local/bin/plantuml"

# Check plantuml version
plantuml -version
