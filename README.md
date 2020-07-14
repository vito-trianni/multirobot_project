# Pysage Version 1.0.0

This is a simple multi-agent simulation environment for mobile agents.
Install it system-wide and use the 'run_pysage' script to launch the simulation.

Usage:
run_pysage -c <config_xml>

Example
run_pysage -c pysage/config/config.xml

The basic classes contain an example of the flocking behaviour by a group of agents.
The xml file provides the basic configuration options for the arena and the agents, as well as for the GUI.
Everything can be configured by overloading the base classes.
The xml file can also be used to switch off the GUI, by commenting the respective line.

For this code, run from: /home/ccapontep/Documents/1_AIRO/Y2S2/Elective_Robotics/Multi-Robot/collective_decision_making_in_dynamical_environments
as: 
python2 run_pysage -c config/myconfig.xml



