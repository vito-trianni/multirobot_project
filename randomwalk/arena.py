# -*- coding: utf-8 -*-
"""
@author: vtrianni and cdimidov
"""
import numpy as np
import math, random
import sys
from pysage import pysage
from results import Results
from agent import CRWLEVYAgent
from target import Target
from collections import defaultdict
import os
directory = os.getcwd()
# Set directory with dataset



class CRWLEVYArena(pysage.Arena): #quindi è una sottoclasse della classe Arena del modulo pysage

    class Factory:
        def create(self, config_element): return CRWLEVYArena(config_element)


    ##########################################################################
    # class init function
    ##########################################################################
    def __init__(self,config_element ):
        pysage.Arena.__init__(self,config_element)

        self.exploitation_rate = 1 if config_element.attrib.get("exploitation_rate") is None else float(config_element.attrib["exploitation_rate"])

        self.timestep_length = 0.5 if config_element.attrib.get("timestep_length") is None else float(config_element.attrib.get("timestep_length"))

        self.integration_step = 0.001 if config_element.attrib.get("integration_step") is None else float(config_element.attrib.get("integration_step"))

        self.size_radius = 0.7506 if config_element.attrib.get("size_radius") is None else float(config_element.attrib.get("size_radius"))

        self.results_filename   = "CRWLEVY" if config_element.attrib.get("results") is None else config_element.attrib.get("results")
        self.results = Results()

        # initialise num runs from the configuration file
        nnruns = config_element.attrib.get("num_runs")
        if nnruns is not None:
            self.num_runs = int(nnruns)
        else:
            self.num_runs = 1

        #  size_radius
        ssize_radius = config_element.attrib.get("size_radius");
        if ssize_radius is not None:
            self.dimensions_radius = float(ssize_radius)
        elif ssize_radius is None:
            self.dimensions_radius = float(self.dimensions.x/2.0)

        # initialise targets from the configuration file
        self.targets = []
        self.num_targets = 0
        for target_element in config_element.iter("target"): # python 2.7
            num_targets_to_configure = 1 if target_element.attrib.get("num_elements") is None else int(target_element.attrib.get("num_elements"))
            for i in range(num_targets_to_configure):
                new_target = Target(target_element)                
                self.targets.append(new_target)
                new_target.id = self.num_targets
                self.num_targets += 1
                print "Initalised target", new_target.id, "(quality value:", new_target.value, ")"




    ##########################################################################
    # initialisation of the experiment
    ##########################################################################
    def init_experiment( self ):
        pysage.Arena.init_experiment(self)
        resource_utility = 0
        for i in range(self.num_targets):
            target = self.targets[i]
            target.value = target.value_start
            resource_utility += target.value
            target.num_committed_agents = 0
            # select target position randomly in the arena
            min_dist = self.dimensions_radius - target.size
            while True:
                # compute a random position within the arena
                while True:
                    target.position = pysage.Vec2d(np.random.uniform(-min_dist, min_dist),np.random.uniform(-min_dist, min_dist))
                    if target.position.get_length() < self.dimensions_radius-target.size:
                        break

                # check non overlapping position with other targets
                overlap = False
                for j in range(i):
                    t=self.targets[j]
                    if target.position.get_distance(t.position) < target.size+t.size:
                        overlap = True
                        break
                if not overlap:
                    break
               
            print"Target id", i, "is at position", target.position

        # initialise agents uniformly within the arena
        max_pos_a = (self.dimensions_radius - CRWLEVYAgent.size)
        agents_on_targets = np.zeros(self.num_targets)
        agents_out = 0
        for agent in self.agents:
            while True:
                agent.position = pysage.Vec2d(random.uniform(-max_pos_a, max_pos_a),random.uniform(-max_pos_a, max_pos_a))
                if agent.position.get_length() < max_pos_a:
                    break

            t = agent.init_on_target()
            if t is not None:
                agents_on_targets[self.targets.index(t)]
            else:
                agents_out += 1

        self.results.new_run()
        self.results.store(np.mean(agents_on_targets), np.std(agents_on_targets), resource_utility)
        print agents_on_targets, agents_out, agents_out+sum(agents_on_targets)
    

    ##########################################################################
    # run experiment until finished
    def run_experiment( self ):
        while not self.experiment_finished():
            self.update()

    ##########################################################################
    # updates the status of the simulation
    ##########################################################################
    def update( self ):
        # computes the desired motion and agent state
        for a in self.agents:
            a.control(self.num_steps)

        # reset the count of aents on targets
        for t in self.targets:
            t.num_committed_agents = 0

        # apply the desired motion - count the number of agents on each target
        agents_on_targets = np.zeros(self.num_targets)
        for a in self.agents:
            a.update()
            if a.position.get_distance((0,0)) > (self.dimensions_radius- a.size/2):
                current_angle = a.position.get_angle()
                a.position = pysage.Vec2d(self.dimensions_radius- a.size/2, 0)
                a.position.rotate(current_angle)
            if a.current_target is not None:
                a.current_target.num_committed_agents += 1
                agents_on_targets[self.targets.index(a.current_target)] += 1
        
        # update target value (exponential exploitation) and compute overall utility
        resource_utility = 0
        for t in self.targets:
            for dt in range(int(self.timestep_length/self.integration_step)):
                t.value -= self.integration_step * t.value * self.exploitation_rate * (t.num_committed_agents*t.num_committed_agents)
            resource_utility += t.value

        self.results.store(np.mean(agents_on_targets), np.std(agents_on_targets), resource_utility)

        # update simulation step counter
        self.num_steps += 1


    ##########################################################################
    # return a list of neighbours
    #####################################################################
    def get_neighbour_agents( self, agent, distance_range ):
        neighbour_list = []
        for a in self.agents:
            if (a is not agent) and ((a.position - agent.position).get_length() < distance_range):
                neighbour_list.append(a)
                # print(neighbour_list)
        return neighbour_list


    ##########################################################################
    # check if the experiment si finished
    ##########################################################################
    def experiment_finished( self ):
        if (self.max_steps > 0) and (self.num_steps >= self.max_steps):
            print "Run finished"

            # save the residence times
            for a in self.agents:
                self.results.append_residence_times(a.residence_times)
            self.results.save(self.results_filename, self.num_agents)
            return True
        return False
    

pysage.ArenaFactory.add_factory("randomwalk.arena", CRWLEVYArena.Factory())
