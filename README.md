# aw_nas

## Introduction

### NAS

### Components of a NAS system

There are multiple actors that are working together in a NAS system, and they can be categorized into these components:

* search space
* controller
* weights manager
* evaluator
* objective

The interface between these components are somehow well-defined. TODO (what is Rollout)

## Install

`pip install -r requirements.txt`

## Usage

After installation, you can run `awnas --help` to see what sub-commands are available.

### Prepare data

When running `awnas` program, it will assume the data of a dataset with `name=<NAME>` under `AWNAS_DATA/<NAME>`, in which `AWNAS_DATA` base directory is read from the environment variable `AWNAS_DATA`. If the environment variable is not specified, the default is `AWNAS_HOME/data`, in which `AWNAS_HOME` is an environment variable default to be `~/awnas`.

* Cifar10: No specific preparation needed.
* PTB: `bash scripts/get_data.sh`, the ptb data will be downloaded under `data/ptb` directory. For using `~/awnas/data` as the data directory, you can run `mkdir -p ~/awnas/data && ln -s ./data/ptb ~/awnas/data/ptb`.

### Run NAS search

### Derive

## Develop New Components

TOOD (The detailed interface of different components, to write new components, these interface method must be overrided.)