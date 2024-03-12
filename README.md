
---

# Comprehensive CUDA Setup Guide on Ubuntu with NVIDIA GPU

This README provides detailed instructions for setting up your Ubuntu system with an NVIDIA GPU to run CUDA programs, specifically focusing on a SHA256 hashing example.

## Table of Contents
1. System Requirements
2. Installing NVIDIA Drivers
3. Installing CUDA Toolkit
4. Verifying CUDA Installation
5. Setting up a Development Environment
6. Compiling and Running a CUDA Program
7. Troubleshooting
8. Additional Resources

### 1. System Requirements
- An NVIDIA GPU that supports CUDA.
- A fresh installation of Ubuntu (version 20.04 LTS or later is recommended).

### 2. Installing NVIDIA Drivers
1. **Update Your System**:
   - Open a terminal and run:
     ```bash
     sudo apt update
     sudo apt upgrade
     ```
2. **Install NVIDIA Drivers**:
   - Identify the recommended driver for your GPU:
     ```bash
     ubuntu-drivers devices
     ```
   - Install the recommended NVIDIA driver:
     ```bash
     sudo ubuntu-drivers autoinstall
     ```
   - Reboot your system.

### 3. Installing CUDA Toolkit
1. **Download the CUDA Toolkit**:
   - Visit the [NVIDIA CUDA Toolkit webpage](https://developer.nvidia.com/cuda-downloads) and select Linux > x86_64 > Ubuntu > the version matching your Ubuntu.
   - Choose the `deb (local)` installer type.
2. **Install CUDA Toolkit**:
   - Follow the instructions on the NVIDIA website to add the CUDA repository and install the CUDA Toolkit.

### 4. Verifying CUDA Installation
1. **Check CUDA Version**:
   - Verify the installation by running:
     ```bash
     nvcc --version
     ```
   - This should display the version of the CUDA compiler.

### 5. Setting up a Development Environment
1. **Install Build Essentials**:
   - Install necessary compilers and development tools:
     ```bash
     sudo apt install build-essential
     ```
2. **Install Text Editor**:
   - (Optional) Install a text editor like Visual Studio Code or Sublime Text for code editing.

### 6. Compiling and Running a CUDA Program
1. **Write Your CUDA Program**:
   - Use the text editor to write your CUDA program (e.g., `sha256_cuda_example.cu`).
2. **Compile the Program**:
   - Compile the program using `nvcc`:
     ```bash
     nvcc -o myprogram sha256_cuda_example.cu
     ```
3. **Run the Program**:
   - Run the compiled executable:
     ```bash
     ./myprogram
     ```

### 7. Troubleshooting
- If you encounter issues, verify that your GPU is CUDA-capable and that the correct NVIDIA drivers are installed.
- Ensure that the CUDA Toolkit is correctly installed and that your environment variables (like `PATH`) are properly configured.
- Check for compilation errors and ensure that your CUDA code is correctly written.

### 8. Additional Resources
- [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [Ubuntu Community NVIDIA Documentation](https://help.ubuntu.com/community/BinaryDriverHowto/Nvidia)
- [Stack Overflow](https://stackoverflow.com/) for specific programming questions.

---

This README is a starting point for users new to CUDA programming on Ubuntu with an NVIDIA GPU. It covers the basic steps needed to set up the environment and run a simple CUDA program. For more complex setups or specific issues, refer to the additional resources provided.
