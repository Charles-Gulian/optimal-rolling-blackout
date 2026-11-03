import numpy as np
import pandas as pd

T = 24  # Time window of optimization model(s)


class Component:

    def __init__(self):
        self.system = None

    def link_system(self, system):
        self.system = system

    @property
    def model(self):
        """The gurobipy Model this component adds its variables to."""
        return self.system.model

    def create_parameters(self):
        pass

    def update_timeseries_parameters(self, date):
        pass

    def create_variables(self):
        pass

    def create_expressions(self):
        pass

    def write_constraints(self):
        pass
