# -*- coding: utf-8 -*-
"""
@author: vtrianni
"""
import sys
import random, math, copy
import numpy as np
from pysage import pysage
from levy_f import distribution_functions

class CRWLEVYAgent(pysage.Agent):

    linear_speed = 1
    num_motion_steps = 1
    interaction_range = 1
    size = 0.01

    class Factory:
        def create(self, config_element, arena): return CRWLEVYAgent(config_element, arena)

    ##########################################################################
    # standart init function
    ##########################################################################
    def __init__( self, config_element, arena ):
        pysage.Agent.__init__(self, config_element, arena )

        # parse custom parameters from configuration file
        CRWLEVYAgent.size = 0.02 if config_element.attrib.get("size") is None else float(config_element.attrib["size"])

        # control parameter: motion speed
        sspeed = config_element.attrib.get("linear_speed")
        if sspeed is not None:
            CRWLEVYAgent.linear_speed = float(sspeed)

        # control parameter: interaction range
        srange = config_element.attrib.get("interaction_range")
        if srange is not None:
             CRWLEVYAgent.interaction_range = float(srange)

        # control parameter : value of CRW_exponent
        cc= config_element.attrib.get("CRW_exponent")
        if cc is not None:
            CRWLEVYAgent.CRW_exponent = float(cc)
            if (CRWLEVYAgent.CRW_exponent < 0) or (CRWLEVYAgent.CRW_exponent >= 1):
                raise ValueError, "parameter for correlated random walk outside of bounds ( should be in [0,1[ )"

        # control parameter : value of alpha that is the Levy_exponent
        salpha= config_element.attrib.get("levy_exponent")
        if salpha is not None:
            CRWLEVYAgent.levy_exponent = float(salpha)

        # control parameter : value of standard deviation (in seconds)
        ssigma= config_element.attrib.get("std_motion_steps")
        if ssigma is not None:
            CRWLEVYAgent.std_motion_steps = float(ssigma)/arena.timestep_length

        # counters for the number of steps used for straight motion
        self.count_motion_steps = 0

        # counter time enter on the target
        self.steps_on_target = 0
        self.residence_times = []

        # pointer to the current target (None when out of any target)
        self.current_target = None


    ##########################################################################
    # String representaion (for debugging)
    ##########################################################################
    def __repr__(self):
        return 'CRWLEVY', pysage.Agent.__repr__(self)


    ##########################################################################
    #  initialisation/reset of the experiment variables
    ##########################################################################
    def init_experiment( self ):
        pysage.Agent.init_experiment( self )

        self.count_motion_steps = int(math.fabs(distribution_functions.levy(CRWLEVYAgent.std_motion_steps,CRWLEVYAgent.levy_exponent)))

        # data for colletive decision making
        self.step_neighbours = []
        self.step_target = None
        self.target_committed = None
        self.target_value = 0
        self.target_color = "black"
        self.next_committed_state = None
        self.decision_made_Neigh = False # save info if decision made by agent

    ##########################################################################
    #  initialisation/reset of the experiment variables
    ##########################################################################
    def init_on_target( self ):
        # intiialise data for passage on target
        self.current_target = None
        self.steps_on_target = 0
        self.residence_times = []
        for t in self.arena.targets:
            if (t.position - self.position).get_length() < t.size:
                self.current_target = t
                break
        return self.current_target

    
    ##########################################################################
    # compute the desired motion as a random walk
    ##########################################################################
    def control(self, num_steps):

        # check if the agent is over any target
        if self.current_target is not None:
            if (self.current_target.position - self.position).get_length() > self.current_target.size:
                self.on_target = False
                self.current_target = None
                self.residence_times.append(self.steps_on_target)
            else:
                self.steps_on_target += 1
        else:
            self.current_target = None
            for t in self.arena.targets:
                if (t.position - self.position).get_length() < t.size:
                    self.current_target = t
                    self.steps_on_target = 0
                    break


        # agent basic movement: go straight
        self.apply_velocity = pysage.Vec2d(CRWLEVYAgent.linear_speed,0)
        self.apply_velocity.rotate(self.velocity.get_angle())

        # agent random walk: decide step length and turning angle
        self.count_motion_steps -= 1
        if self.count_motion_steps <= 0:
            # step length
            self.count_motion_steps = int(math.fabs(distribution_functions.levy(CRWLEVYAgent.std_motion_steps,CRWLEVYAgent.levy_exponent)))
            # turning angle
            crw_angle = 0
            if CRWLEVYAgent.CRW_exponent == 0:
                crw_angle = distribution_functions.uniform_distribution(0,(2*math.pi))
            else:
                crw_angle = distribution_functions.wrapped_cauchy_ppf(CRWLEVYAgent.CRW_exponent)

            self.apply_velocity.rotate(crw_angle)

        # check if close to the arena border
        if self.position.get_length() >= self.arena.size_radius - CRWLEVYAgent.size:
            self.apply_velocity = pysage.Vec2d(CRWLEVYAgent.linear_speed,0)
            self.apply_velocity.rotate(math.pi+self.position.get_angle()+random.uniform(-math.pi/2,math.pi/2))



pysage.AgentFactory.add_factory("randomwalk.agent", CRWLEVYAgent.Factory())
