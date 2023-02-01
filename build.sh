# clone the git repo
git clone -b develop --recursive --shallow-submodules https://github.com/rrmeister/pyQuEST

# load python, numpy, and cmake
module load numpy/1.22.3-python-3.9.11-openblas-0.3.19
module load cmake/3.19.4

# make the virtual environment
echo "Creating virtual environment ..."
python3 -m venv quantum-playground
source venv/bin/activate

# get the NumPy include folder
NPINCL=`python3 -u -c 'import numpy; print(numpy.get_include())'`
export NPINCL

# here we need to patch pyQuEST/pyquest/CMakeLists.txt
patch pyQuEST/pyquest/CMakeLists.txt <<EOF
--- pyQuEST/pyquest/CMakeLists.txt.org	2022-06-21 17:00:04.000000000 +0200
+++ pyQuEST/pyquest/CMakeLists.txt	2022-06-21 17:00:53.000000000 +0200
@@ -28,7 +28,7 @@
 # scikit-build modules needed for compilation.
 find_package(PythonExtensions REQUIRED)
 find_package(Cython REQUIRED)
-find_package(NumPy REQUIRED)
+find_package(NumPy REQUIRED PATH \$ENV{NPINCL})
 
 # Cython targets we need to compile. File extensions are automatically
 # added by add_cython_target later.
@@ -52,6 +52,7 @@
     set(CMAKE_MESSAGE_LOG_LEVEL WARNING)
     python_extension_module(\${cy_target})
     unset(CMAKE_MESSAGE_LOG_LEVEL)
+    set(NumPy_INCLUDE_DIRS PATH \$ENV{NPINCL})
     target_include_directories(\${cy_target} PUBLIC \${NumPy_INCLUDE_DIRS})
     target_link_libraries(\${cy_target} QuEST quest_error.h)
 
EOF

# pip install
echo "Install pyquest with local pip ..."
python3 -m pip install ./pyQuEST

# done
echo "Now activate the venv with"
echo "module load numpy/1.22.3-python-3.9.11-openblas-0.3.19"
echo "source venv/bin/activate"
echo "... and have fun!"
echo "Test e.g. with"
echo "python test_pyquest.py"

exit 0
