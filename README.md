# Deep Drone Acrobatics

This repo contains the code associated to our paper Deep Drone Acrobatics. 

<p align="center">
  <img src="./img/fma_powerloop.gif" alt="dda">
</p>


#### Citing

If you use this code in an academic context, please cite the following publication:

Paper: [Deep Drone Acrobatics](http://rpg.ifi.uzh.ch/docs/RSS20_Kaufmann.pdf)

Video: [YouTube](https://youtu.be/2N_wKXQ6MXA)

```
@article{kaufmann2020RSS,
  title={Deep Drone Acrobatics},
  author={Elia, Kaufmann and Antonio, Loquercio and René, Ranftl and Matthias, Müller and Vladlen, Koltun and Davide, Scaramuzza},
  journal={RSS: Robotics, Science, and Systems},
  year={2020},
  publisher={IEEE}
}
```

## Installation

### Requirements

The code was tested with Ubuntu 18.04, ROS Melodic, Anaconda v4.8.3.
Different OS and ROS versions are possible but not supported.


### Step-by-Step Procedure

Use the following commands to create a new catkin workspace and a virtual environment with all the required dependencies.

```bash
export ROS_VERSION=melodic
mkdir drone_acrobatics_ws
cd drone_acrobatics_ws
export CATKIN_WS=./catkin_dda
mkdir -p $CATKIN_WS/src
cd $CATKIN_WS
catkin init
catkin config --extend /opt/ros/$ROS_VERSION
catkin config --merge-devel
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-fdiagnostics-color
cd src

git clone https://github.com/uzh-rpg/deep_drone_acrobatics.git
cd deep_drone_acrobatics
git checkout release
cd ..
vcs-import < deep_drone_acrobatics/dependencies.yaml

#install extra dependencies (might need more depending on your OS)
sudo apt-get install libqglviewer-dev-qt5
```

Before continuing, make sure that your protobuf compiler version is 3.0.0.
To check this out, type in a terminal ``protoc --version``.
If This is not the case, then check out [this guide](https://github.com/linux-on-ibm-z/docs/wiki/Building-ProtoBuf-3.0.0) on how to do it.



```bash
# Build and re-source the workspace
catkin build
. ../devel/setup.bash

# Create your learning environment
cd deep_drone_acrobatics
conda create --name drone_flow python=3.6
conda activate drone_flow
# Install (in an hacky way) python requirements
pip install -r pip_requirements.txt
conda install --file conda_requirements.txt

```


## Let's do a Power Loop

Once you have installed the dependencies, you will be able to fly in simulation with our pre-trained checkpoint. You don't need necessarely need a GPU for execution. Note that if the network can't run at least at 15Hz, you won't be able to fly successfully.

Lauch a simulation! Open a terminal and type:
```bash
cd drone_acrobatics_ws
source catkin_dda/devel/setup.bash
roslaunch fpv_aggressive_trajectories simulation.launch
```

Run the Network in an other terminal:
```bash
cd
cd drone_acrobatics_ws
. ./catkin_dda/devel/setup.bash
conda activate drone_flow
python test_trajectories.py --settings_file=config/test_settings.yaml

```

## Train your own acrobatic controller

You can use the following commands to generate data in simulation and train your model on it. The trained checkpoint can then be used to control a physical platform (if you have one!).

### Generate data

Launch the simulation in one terminal
```bash
cd drone_acrobatics_ws
source catkin_dda/devel/setup.bash
roslaunch fpv_aggressive_trajectories simulation.launch
```

Launch data collection (with dagger) in an other terminal
```bash
cd
cd drone_acrobatics_ws
. ./catkin_dda/devel/setup.bash
conda activate drone_flow
python iterative_learning_trajectories.py --settings_file=config/dagger_settings.yaml
```

It is possible to change parameters (number of rollouts, history length, etc. ) in the file [dagger\_settings.yaml](./controller_learning/config/dagger_settings.yaml). Keep in mind that if you change the network (for example by changing the history length), you will need to adapt the file [test_settings.yaml](./controller_learning/config/test_settings.yaml) for compatibility.


### Train the Network

If you want to train the network on data already collected (for example to do some parameter tuning) use the following commands.
Make sure to adapt the settings file to your configuration.

```bash
cd
cd drone_acrobatics_ws
. ./catkin_dda/devel/setup.bash
conda activate drone_flow
python train.py --settings_file=config/train_settings.yaml
```

### Test the Network

To test the network you trained, adapt the [test_settings.yaml](./controller_learning/config/test_settings.yaml) with the new checkpoint path. Then follow the instruction to test!

Lauch a simulation! Open a terminal and type:
```bash
cd drone_acrobatics_ws
source catkin_dda/devel/setup.bash
roslaunch fpv_aggressive_trajectories simulation.launch
```

Run the Network in an other terminal:
```bash
cd
cd drone_acrobatics_ws
. ./catkin_dda/devel/setup.bash
conda activate drone_flow
python test_trajectories.py --settings_file=config/test_settings.yaml

```
