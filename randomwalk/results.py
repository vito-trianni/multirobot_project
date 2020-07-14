"""
@author: Vito Trianni
"""
import numpy as np
import operator
import scipy.stats as st
import scipy.special as sc

class Results:
    'A class to store the results '

    def __init__( self ):

        self.avg_agents_on_target = []
        self.std_agents_on_target = []
        self.resource_utility = []
        self.residence_times = []
        self.current_run = -1

    def new_run( self ):
        self.current_run += 1
        self.avg_agents_on_target = []
        self.std_agents_on_target = []
        self.resource_utility = []
        self.residence_times = []


    def store( self, avg_agents_on_target, std_agents_on_target, resource_utility ):
            self.avg_agents_on_target.append(avg_agents_on_target)
            self.std_agents_on_target.append(std_agents_on_target)
            self.resource_utility.append(resource_utility)

    def append_residence_times(self, steps_on_target ):
        self.residence_times.extend(steps_on_target)

    def save( self, filename, num_agents ):
        run_filename = "%s_N%d_run%03d.dat" % (filename, num_agents, self.current_run)
        with open(run_filename, "w+") as f:
            all_data = zip(self.avg_agents_on_target,self.std_agents_on_target,self.resource_utility)
            for row in list(all_data):
                f.write('\t'.join(str(data) for data in row))
                f.write('\n')
            f.close()

        if self.residence_times:
            res_filename = "%s_residence_N%d_run%03d.dat" % (filename, num_agents, self.current_run)
            with open(res_filename, "w+") as f:
                f.write('\n'.join(str(d) for d in self.residence_times))
                f.write('\n')
                f.close()
