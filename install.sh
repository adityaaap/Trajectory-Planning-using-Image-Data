# Packages to install
PACKAGES="numpy numpngw control opencv-python tqdm gym"

# Loop through the list of packages and install each one using pip
for package in $PACKAGES; do
    echo "Installing $package..."
    pip install $package
done

# All packages installed successfully
echo "All packages have been installed successfully!"