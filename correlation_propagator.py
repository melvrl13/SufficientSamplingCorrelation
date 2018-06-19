#!/usr/bin/env python

# Ideally, you'll use the same atom selection you would for a correlation matrix.
# I suggest 'name CA' for proteins and 'not element H' for nucleic acids.

import argparse
import numpy
import mdtraj
import pandas
import matplotlib.pyplot
import matplotlib.colors
import seaborn
import numpy

# Initialize parser. The default help has poor labeling. See http://bugs.python.org/issue9694
parser = argparse.ArgumentParser(description='Calculate Propagating Correlation Matrix', add_help=False)

# List all possible user input
inputs = parser.add_argument_group('Input arguments')
inputs.add_argument('-h', '--help', action='help')
inputs.add_argument('-top', action='store', dest='structure',
                    help='Structure file corresponding to trajectory', type=str, required=True)
inputs.add_argument('-traj', action='store', dest='trajectory', help='Trajectory', type=str, required=True)
inputs.add_argument('-sel', action='store', dest='sel', help='Atom selection', type=str, default='not element H')
inputs.add_argument('-tau', action='store', dest='tau', help='lag time', type=int, default=1)
inputs.add_argument('-o', action='store', dest='out_name', help='Output prefix', type=str, required=True)

class Selector:
    def __init__(self, trajectory, atom_selection):
        assert isinstance(trajectory, mdtraj.Trajectory)
        self.trajectory = trajectory
        self.sel = atom_selection

    def select(self):
        raise NotImplementedError


class Slice(Selector):
    def select(self):
        indices = self.trajectory.top.select(self.sel)
        sub_trajectory = self.trajectory.atom_slice(atom_indices=indices, inplace=False)
        return sub_trajectory


class Correlation:
    def __init__(self, trajectory):
        assert isinstance(trajectory, mdtraj.Trajectory)
        self.trajectory = trajectory
        self.correlation_matrix = []

    def calculate(self):
        raise NotImplementedError


class Propagator(Correlation):
    def __init__(self, trajectory, tau):
        self.tau = tau
        super().__init__(trajectory)

    def calculate(self):
        delta_sum = numpy.zeros([self.trajectory.topology.n_atoms, 3], dtype=float)
        dot_sum = numpy.zeros([self.trajectory.topology.n_atoms, self.trajectory.topology.n_atoms], dtype=float)
        for frame in numpy.arange(self.trajectory.n_frames - self.tau):
            delta_temp = self.trajectory.xyz[frame] - self.trajectory.xyz[frame + self.tau]
            delta_sum = delta_sum + delta_temp
            dot_sum = dot_sum + numpy.inner(delta_temp, delta_temp) # same as dot o f v and v' where v is <n_atoms, 3>
        average_delta = delta_sum / (self.trajectory.n_frames - self.tau)
        average_dot = dot_sum / (self.trajectory.n_frames - self.tau)
        dot_average_delta = numpy.inner(average_delta, average_delta)
        #_, normalization_matrix = Pearson(self.trajectory).calculate()
        diagonal = numpy.diag(average_dot)
        normalization_matrix = numpy.outer(diagonal, diagonal)
        normalization_matrix = numpy.sqrt(normalization_matrix)
        normalized_average_dot = numpy.divide(average_dot, normalization_matrix)
        return normalized_average_dot, average_delta, dot_average_delta

class Plotter:
    def __init__(self, y, out_name, x_label=' ', y_label=' ', title=' '):
        self.y = y
        self.out_name = out_name
        self.x_label = x_label
        self.y_label = y_label
        self.title = title

    def plot(self):
        raise NotImplementedError


class UnityPColor(Plotter):
    def plot(self):
        matplotlib.pyplot.pcolor(self.y, cmap='jet', vmin=-1, vmax=1)
        matplotlib.pyplot.colorbar()
        matplotlib.pyplot.ylim(0, len(self.y))
        matplotlib.pyplot.xlim(0, len(self.y))
        matplotlib.pyplot.xlabel(self.x_label)
        matplotlib.pyplot.ylabel(self.y_label)
        matplotlib.pyplot.title(self.title)
        matplotlib.pyplot.savefig(self.out_name)
        matplotlib.pyplot.close()


class Saver:
    def __init__(self, out_name):
        self.out_name = out_name

    def save(self):
        raise NotImplementedError

class Array(Saver):
    def __init__(self, array, out_name):
        self.array = array
        super().__init__(out_name)
    def save(self):
        numpy.savetxt(self.out_name, self.array)


class Reader:
    def __init__(self, trajectory_path):
        self.trajectory = trajectory_path

    def load(self):
        raise NotImplementedError


class DCD(Reader):
    def __init__(self, trajectory_path, topology_path):
        self.topology = topology_path
        super().__init__(trajectory_path)

    def load(self):
        trajectory = mdtraj.load(self.trajectory, top=self.topology)
        return trajectory




# Parse into useful form
UserInput = parser.parse_args()
trajectory = Reader.DCD(topology_path=UserInput.structure, trajectory_path=UserInput.trajectory).load()
trajectory = Selector. Slice(trajectory=trajectory, atom_selection=UserInput.sel).select()
# Execute calculation
cp = Correlation.Propagator(trajectory=trajectory, tau=UserInput.tau)
average_dot, average_delta, dot_average_delta = cp.calculate()

# Save text results
Saver.Array(array=average_dot, out_name=UserInput.out_name + '_average_dot.txt').save()
Saver.Array(array=average_delta, out_name=UserInput.out_name + '_average_delta.txt').save()
Saver.Array(array=dot_average_delta, out_name=UserInput.out_name + '_dot_average_delta.txt').save()

# Save pretty pictures
# need to add matshow to Plotter
Plotter.UnityPColor(y=average_dot,
                    x_label='Atom',
                    y_label='Atom',
                    title='Average dot product',
                    out_name=UserInput.out_name + '_average_dot.png').plot()


