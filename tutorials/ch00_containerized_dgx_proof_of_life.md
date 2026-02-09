# Chapter 0: A Containerized "Proof of Life" on Spark DGX

## Abstract

This chapter establishes the foundational experimental environment upon which all subsequent work depends. We address a problem that is logically prior to reinforcement learning itself: the problem of *reproducible computation*. Our deliverables are not trained policies but verified infrastructure—a container that runs, a renderer that produces images, a training loop that completes without error.

The reader who dismisses this chapter as mere "setup" misunderstands its purpose. In empirical machine learning, the experimental environment is not scaffolding to be discarded; it is the laboratory in which results are produced. A result that cannot be reproduced because the environment cannot be reconstructed is not a result at all. This chapter ensures that our laboratory is sound.

---

## Part I: The Problem

### 1.1 WHY: The Reproducibility Crisis and Its Resolution

Consider the following scenario, which is unfortunately common in empirical machine learning research. A researcher trains a policy that achieves impressive results. They write a paper, submit it, and it is accepted. Six months later, a colleague attempts to reproduce the results. The code is available, but it fails to run: dependencies have changed, the CUDA version is different, an environment variable is missing. After days of debugging, the colleague achieves results that are qualitatively different from the original paper. Was there an error in the original work? Or is the discrepancy due to environmental differences? It is impossible to know.

This scenario illustrates what we might call the *reproducibility problem* in empirical machine learning:

**Problem (Reproducibility).** *Given a computational experiment $E$ that produces result $R$ on machine $M$ at time $t$, under what conditions can we guarantee that $E$ produces result $R'$ with $\|R - R'\| < \epsilon$ on machine $M'$ at time $t' > t$?*

The problem is harder than it appears. The result $R$ depends not only on the code but on the entire computational environment: the operating system, the installed libraries, the GPU driver version, the CUDA toolkit, the Python interpreter, and dozens of other components. Any of these may change between $t$ and $t'$, and any change may affect $R$.

In the tradition of Hadamard, we ask: Is the reproducibility problem *well-posed*?

**Definition (Well-Posedness for Reproducibility).** *The reproducibility problem is well-posed if: (1) there exists an environment specification $S$ such that running $E$ in any environment satisfying $S$ produces consistent results; (2) the specification $S$ is unique up to equivalence; (3) small perturbations to $S$ produce small perturbations to $R$.*

Condition (1) requires that we can *specify* an environment precisely enough to guarantee consistency. Condition (2) requires that the specification be *canonical*—that there not be multiple incompatible specifications claiming to represent the same environment. Condition (3) requires *stability*—that the result not be arbitrarily sensitive to minor environmental variations.

Containerization addresses all three conditions. A Docker image provides a complete, self-contained specification of the computational environment. The image is identified by a content-addressable hash, ensuring uniqueness. And the layered filesystem ensures that small changes to the specification (adding a package, changing a configuration) produce small changes to the resulting environment.

This is why we containerize: not for convenience, but for *scientific validity*.

### 1.2 The Specific Problem of This Chapter

Within the general reproducibility framework, this chapter addresses a specific sub-problem:

**Problem (Environment Verification).** *Construct and verify a containerized environment $S$ such that: (1) GPU computation is available within $S$; (2) MuJoCo physics simulation runs correctly within $S$; (3) headless rendering produces valid images within $S$; (4) a complete training loop executes without error within $S$.*

Each condition is necessary for the reinforcement learning experiments that follow. Without GPU access, training is prohibitively slow. Without MuJoCo, we cannot simulate the Fetch robot. Without rendering, we cannot generate evaluation videos. Without a working training loop, we cannot learn policies.

The verification is not optional. A researcher who skips this chapter and proceeds directly to training will eventually encounter failures—rendering errors, CUDA misconfigurations, import failures—and will spend more time debugging than if they had verified the environment systematically from the start.

---

## Part II: The Method

### 2.1 HOW: Containerization as Environment Specification

Our approach is to specify the environment as a Docker container and to verify each requirement through explicit tests.

**Definition (Container).** *A container is a lightweight, isolated execution environment defined by an image. The image specifies the filesystem contents, environment variables, and default commands. Containers share the host kernel but have isolated process and network namespaces.*

**Definition (Image).** *An image is an immutable template from which containers are instantiated. Images are identified by a content-addressable hash (digest) and may be tagged with human-readable names (e.g., `robotics-rl:latest`).*

The key property of containers for our purposes is *isolation with specification*. The container is isolated from the host environment—packages installed on the host do not affect the container—but the isolation is precisely specified by the image, which can be versioned, shared, and reconstructed.

**Remark (On the Choice of Docker).** *We use Docker rather than alternatives (Singularity, Podman, etc.) because Docker is ubiquitous, well-documented, and fully supported by the NVIDIA Container Toolkit. For HPC environments where Docker is unavailable, the concepts transfer to Singularity with minor modifications.*

### 2.2 The Verification Protocol

Our verification protocol consists of four tests, each corresponding to one requirement from the problem statement:

**Test 1 (GPU Access).** Run `nvidia-smi` inside a container with `--gpus all`. The test passes if the output lists at least one GPU.

**Test 2 (MuJoCo Functionality).** Import MuJoCo and create a Fetch environment. The test passes if `env.reset()` returns a valid observation dictionary.

**Test 3 (Headless Rendering).** Render a frame from the Fetch environment using `render_mode="rgb_array"`. The test passes if the output is a non-empty image file.

**Test 4 (Training Loop).** Run PPO for a small number of timesteps and save a checkpoint. The test passes if the checkpoint file exists and is non-empty.

These tests are implemented in `scripts/ch00_proof_of_life.py`. The script provides subcommands for running each test individually or all tests in sequence.

### 2.3 The Container Architecture

Our container architecture consists of two layers:

**Base Layer.** We use the NVIDIA PyTorch image (`nvcr.io/nvidia/pytorch:25.12-py3`) as the base. This image provides CUDA, cuDNN, PyTorch, and other deep learning infrastructure pre-configured and tested by NVIDIA.

**Project Layer.** On top of the base, we install system dependencies for MuJoCo rendering (EGL, OSMesa) and Python dependencies for the project (Gymnasium, Gymnasium-Robotics, Stable Baselines 3). These are specified in the `docker/Dockerfile`.

**Remark (On the Two-Layer Architecture).** *The separation into base and project layers reflects a design principle: heavyweight, stable dependencies (CUDA, PyTorch) belong in the base layer; lightweight, project-specific dependencies belong in the project layer. This separation enables faster iteration—changing project dependencies does not require rebuilding the entire CUDA stack—while maintaining reproducibility.*

### 2.4 The Virtual Environment Within the Container

A subtle point deserves elaboration. We use a Python virtual environment *inside* the container, even though the container already provides isolation.

**Proposition.** *A virtual environment inside a container provides additional benefits: (1) it enables `pip install -e .` for editable installs of the project; (2) it allows project dependencies to shadow container dependencies when necessary; (3) it makes the dependency specification explicit in `requirements.txt` rather than implicit in the Dockerfile.*

The virtual environment is created with `--system-site-packages`, which allows it to inherit packages from the container's system Python. This avoids reinstalling PyTorch (which is large and CUDA-specific) while still allowing project-specific packages to be installed separately.

---

## Part III: The Implementation

### 3.1 WHAT: The Concrete Steps

We now describe the concrete steps to establish and verify the environment.

#### Step 1: Verify Host Prerequisites

Before using containers, we must verify that the host system is properly configured.

**Verification (Docker Availability).** Run `docker --version` on the host. The test passes if Docker reports a version number.

**Verification (NVIDIA Runtime).** Run `docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.12-py3 nvidia-smi` on the host. The test passes if `nvidia-smi` output appears showing at least one GPU.

If either test fails, consult your system administrator. This tutorial does not cover Docker installation or NVIDIA runtime configuration.

#### Step 2: Enter the Development Environment

The repository provides a wrapper script that automates container setup:

```bash
cd /home/vladp/src/robotics
bash docker/dev.sh
```

This script performs the following operations:

1. Checks for the `robotics-rl:latest` image; builds it if missing
2. Launches a container with GPU access and appropriate environment variables
3. Mounts the repository at `/workspace`
4. Runs as your host UID/GID to avoid permission issues
5. Creates `.venv` if it does not exist and installs `requirements.txt`
6. Activates the virtual environment and drops you into a shell

Alternatively, to run a single command without entering an interactive shell:

```bash
bash docker/dev.sh python scripts/ch00_proof_of_life.py all
```

#### Step 3: Verify Fetch Environment Availability

Inside the container (or via `docker/dev.sh`), run:

```bash
python scripts/ch00_proof_of_life.py list-envs
```

**Expected Output.** A list of environment IDs including `FetchReach-v4`, `FetchReachDense-v4`, `FetchPush-v4`, `FetchPickAndPlace-v4`, and their variants.

**Failure Mode.** If no Fetch environments appear, `gymnasium-robotics` is not installed correctly. Verify with `pip list | grep gymnasium`.

#### Step 4: Verify Headless Rendering

```bash
python scripts/ch00_proof_of_life.py render --out smoke_frame.png
```

**Expected Output.** A message indicating successful rendering and the creation of `smoke_frame.png`.

**Failure Mode.** If rendering fails with EGL errors, the script attempts fallback to OSMesa. If both fail, rendering is disabled. You can still train policies, but you cannot generate videos.

**Remark (Rendering Backend Hierarchy).** *The script implements a fallback chain: EGL (hardware-accelerated) → OSMesa (software) → disabled. EGL requires NVIDIA drivers and EGL libraries; OSMesa requires the Mesa library. The `robotics-rl:latest` image includes both.*

#### Step 5: Verify Training Loop

```bash
python scripts/ch00_proof_of_life.py ppo-smoke --n-envs 8 --total-steps 50000 --out ppo_smoke
```

**Expected Output.** Training progress messages followed by the creation of `ppo_smoke.zip`.

**Failure Mode.** CUDA errors indicate GPU misconfiguration. Shape mismatch errors indicate problems with observation space handling. Import errors indicate missing dependencies.

#### Step 6: Combined Verification

For convenience, all tests can be run in sequence:

```bash
python scripts/ch00_proof_of_life.py all
```

This is the recommended way to verify a fresh installation.

---

## Part IV: Analysis and Verification

### 4.1 Interpreting the Results

Upon successful completion of all tests, the following artifacts should exist:

| Artifact | Purpose | Expected Properties |
|----------|---------|---------------------|
| `.venv/` | Python virtual environment | Contains `gymnasium`, `stable-baselines3`, etc. |
| `smoke_frame.png` | Rendered frame | Non-empty PNG, shows Fetch robot |
| `ppo_smoke.zip` | Trained checkpoint | Non-empty ZIP, loadable by SB3 |

The existence and properties of these artifacts constitute empirical verification that the environment is correctly configured.

### 4.2 On the Meaning of "Proof of Life"

The term "proof of life" is borrowed from hostage negotiations, where it refers to evidence that a hostage is still alive. In our context, it refers to evidence that the computational environment is functional.

This is not mere metaphor. A misconfigured environment is, from the perspective of reproducible science, *dead*: it cannot produce results that can be trusted or reproduced. The tests in this chapter establish that our environment is *alive*—capable of producing valid, reproducible results.

### 4.3 What This Chapter Does Not Verify

This chapter verifies that the environment is functional; it does not verify that training produces good policies. The PPO smoke test runs for only 50,000 timesteps, which is insufficient for convergence. The purpose is to confirm that the training loop executes, not that it produces useful results.

Verification of learning performance is deferred to Chapter 2, where we establish PPO baselines with proper evaluation protocols.

---

## Part V: Deliverables

Upon completion of this chapter, the following must be true:

**D1.** The command `bash docker/dev.sh` enters a container shell with an activated virtual environment.

**D2.** The file `smoke_frame.png` exists and contains a valid image (viewable in an image viewer, showing the Fetch robot).

**D3.** The file `ppo_smoke.zip` exists and is a valid Stable Baselines 3 checkpoint (loadable via `PPO.load("ppo_smoke.zip")`).

**D4.** All four tests in `scripts/ch00_proof_of_life.py all` pass without error.

A reader who cannot satisfy all four conditions has not completed this chapter and should not proceed.

---

## Appendix A: Troubleshooting

### A.1 "Permission denied" When Running Docker

**Symptom.** `docker: Got permission denied while trying to connect to the Docker daemon socket.`

**Cause.** Your user is not in the `docker` group.

**Resolution.** Either add your user to the `docker` group (`sudo usermod -aG docker $USER`, then log out and back in) or run Docker commands with `sudo`.

### A.2 "I have no name!" in the Container Shell

**Symptom.** The shell prompt shows `I have no name!@<container-id>`.

**Cause.** The container is running as your numeric UID, which has no entry in `/etc/passwd` inside the container.

**Impact.** None. This is cosmetic. File permissions work correctly.

### A.3 EGL Initialization Failures

**Symptom.** `mujoco.FatalError: gladLoadGL error` or similar EGL errors.

**Cause.** EGL libraries are missing or the GPU driver does not support EGL.

**Resolution.** The `robotics-rl:latest` image includes EGL libraries. If using a different base image, install `libegl1` and `libgl1`. If EGL is unavailable, the script falls back to OSMesa.

### A.4 Dependency Hash Mismatch

**Symptom.** `docker/dev.sh` reinstalls packages on every run.

**Cause.** The requirements hash file (`.venv/.requirements_hash`) is missing or corrupt.

**Resolution.** Delete `.venv` and re-run `docker/dev.sh` to recreate the environment.

---

## Appendix B: Environment Variable Reference

| Variable | Value | Purpose |
|----------|-------|---------|
| `MUJOCO_GL` | `egl` | Selects EGL rendering backend |
| `PYOPENGL_PLATFORM` | `egl` | Configures PyOpenGL for EGL |
| `NVIDIA_DRIVER_CAPABILITIES` | `all` | Enables full GPU access in container |

These variables are set automatically by `docker/dev.sh`. If you launch containers manually, you must set them explicitly.

---

**Next.** With the experimental environment verified, proceed to Chapter 1 to examine the structure of goal-conditioned Fetch environments.
