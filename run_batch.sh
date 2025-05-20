#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

RUNFILE="run_all_found_files.sh"

# Add #!/bin/bash at the top and prefix with ./run.sh for each .mp4 file
echo '#!/bin/bash' > $RUNFILE

# Python doesn't like a loop and therefore we need to use xargs to a file instead
find "$1" -type f -name "*.mp4" ! -path "*trimmed*" ! -path "*portrait*" | xargs realpath | sed 's|^|./run.sh -i |' >> $RUNFILE

# Make the file executable
chmod +x $RUNFILE

# Run the file
./$RUNFILE

# rm $RUNFILE