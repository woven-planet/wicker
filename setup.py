import os

from setuptools import find_packages, setup

this_dir = os.path.dirname(os.path.abspath(__file__))
version_file_path = os.path.join(this_dir, "VERSION")

# Read in the current "semantic" version (X.Y.Z)
with open(version_file_path) as version_file:
    org_contents = version_file.read()
    version = org_contents.strip()

# BUILDKITE_BUILD_NUMBER could be unset or set to an empty string
buildkite_number = os.environ.get("BUILDKITE_BUILD_NUMBER") or "dev0"
pkg_version = "{}.{}".format(version, buildkite_number)
# Append the BUILD number to the "semantic" version (X.Y.Z) and record it in the VERSION file.
# This ensures that the VERSION file in the published packaged remains the source of truth.
with open(version_file_path, 'w') as version_file:
    version_file.write("{}\n".format(pkg_version))

setup(
    name="wicker",
    url="https://github.tri-ad.tech/level5/wicker",
    version=pkg_version,
    packages=find_packages(where="."),
    install_requires=[
        "numpy>=1.18.3",
        "pyarrow",
    ],
    python_requires='>=3.8',
)

# Revert VERSION file changes.
with open(version_file_path, 'w') as version_file:
    version_file.write(org_contents)
