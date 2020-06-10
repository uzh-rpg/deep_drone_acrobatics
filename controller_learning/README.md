## Learning Code for Thrust+Bodyrates Prediction


### Test Network on throws
Launch the simulation
```
roslaunch controller_learning simulation.launch
```

In an other terminal, with cv\_bridge and geometry2, run
```
roscd controller_learning
python evaluate_network.py --settings_file=config/test_settings.yaml
```

To check if network is able to re-initialize VIO, run in a separate terminal:
```
roslaunch vins_estimator raf_simulation.launch
```

### Test Network on loops
Make sure you have VinsMono built in the branch "imu\_constant\_integration".

Launch all the non-python related stuff in one terminal (catkin build with python2.7)
```
roslaunch fpv_aggressive_trajectories simulation.launch
```

In an other terminal, with cv\_bridge and geometry2 built with python3, run
```
roscd controller_learning
python test_trajectories.py --settings_file=config/test_settings.yaml
```

### Generate data 
Launch the simulation
```
roscore
roslaunch controller\_learning simulation.launch
```
Start the data generation & learning script
```
python iterative_learning_node.py --settings_file=config/settings.yaml
```
To stop the learning script:
```
rostopic pub /shutdown_learner std_msgs/Empty "{}" -1
```


### Train the network
```bash
python train.py --settings_file=config/settings.yaml
```

### How to resolve the Python2/3 issues
Our learning code runs on Python3, ROS runs on Python2. These steps show how you can have both:

Install some needed dependencies:
```
sudo apt-get install python3-pip python3-yaml
sudo pip3 install rospkg catkin_pkg
```
To resolve the `cv_bridge` issues:
```
sudo apt-get install python-catkin-tools python3-dev python3-numpy
```
Create a new catkin workspace to avoid any complications with your existing workspace:
```
mkdir ~/catkin_py3_ws && cd ~/catkin_py3_ws
catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
catkin config --install
mkdir src
cd src
git clone -b melodic https://github.com/ros-perception/vision_opencv.git
git clone -b melodic-devel git@github.com:ros/geometry2.git
catkin build cv_bridge
```
Overlay this workspace to the previously sourced workspace, now `cv_bridge` should work. The other environment should be sourced, this one works on top of it.
```
source /catkin_py3_ws/devel/setup.sh
```

To test if things are going well:
```
python
import sys
sys.path
```
