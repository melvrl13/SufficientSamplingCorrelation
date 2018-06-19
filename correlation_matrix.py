#!/usr/bin/env python

import argparse
from Analysis import AtomSelection, Correlation, Plotter, Saver, TrajectoryReader, TrajectoryProcessor
import numpy
import mdtraj
import pandas
import matplotlib.pyplot
import matplotlib.colors
import seaborn
import numpy

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


class Pearson(Correlation):
    def calculate(self):
        average = numpy.average(self.trajectory.xyz, axis=0)
        fluctuations = self.trajectory.xyz - average[numpy.newaxis, :]
        del average
        dots = numpy.zeros((self.trajectory.n_atoms, self.trajectory.n_atoms))
        for i in range(self.trajectory.n_frames):
            dot = numpy.dot(fluctuations[i, :, :], numpy.transpose(fluctuations[i, :, :]))
            dots = dots + dot
        del fluctuations
        dots = numpy.divide(dots, self.trajectory.n_frames)
        diagonal = numpy.diag(dots)
        normalization_matrix = numpy.outer(diagonal, diagonal)
        normalization_matrix = numpy.sqrt(normalization_matrix)
        self.correlation_matrix = numpy.divide(dots, normalization_matrix)
        return self.correlation_matrix


class TimeLagged(Correlation):
    def __init__(self, trajectory, covariance_tau):
        assert isinstance(covariance_tau, int)
        self.covariance_tau=covariance_tau
        self.normalization_matrix = []
        super().__init__(trajectory)
    def calculate(self):
        average = numpy.average(self.trajectory.xyz, axis=0)
        fluctuations = self.trajectory.xyz - average[numpy.newaxis, :]
        del average
        dots = numpy.zeros((self.trajectory.n_atoms, self.trajectory.n_atoms))
        for i in range(self.trajectory.n_frames - self.covariance_tau):
            dot = numpy.dot(fluctuations[i, :, :], numpy.transpose(fluctuations[i + self.covariance_tau, :, :]))
            dots = dots + dot
        del fluctuations
        dots = numpy.divide(dots, self.trajectory.n_frames)
        diagonal = numpy.diag(dots)
        self.normalization_matrix = numpy.outer(diagonal, diagonal)
        self.normalization_matrix = numpy.sqrt(numpy.absolute(self.normalization_matrix))
        self.correlation_matrix = numpy.divide(dots, self.normalization_matrix)
        return self.correlation_matrix

    
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


class Processor:
    def __init__(self, trajectory, atom_selection):
        assert isinstance(trajectory, mdtraj.Trajectory)
        self.trajectory = trajectory
        self.sel = atom_selection

    def process(self):
        raise NotImplementedError


class Aligner(Processor):
    def __init__(self, trajectory, atom_selection, reference=None):
        self.reference = reference
        super().__init__(trajectory, atom_selection)

    def process(self):
        indices = AtomIndexer(self.trajectory, self.sel).process()
        if self.reference:
            aligned_trajectory = self.trajectory.superpose(reference=self.reference, atom_indices=indices)
        else:
            aligned_trajectory = self.trajectory.superpose(self.trajectory, atom_indices=indices)
        return aligned_trajectory


# Initialize parser. The default help has poor labeling. See http://bugs.python.org/issue9694
parser = argparse.ArgumentParser(description='Calculate, save and plot correlation matrix', add_help=False)

# List all possible user input
inputs = parser.add_argument_group('Input arguments')
inputs.add_argument('-h', '--help', action='help')
inputs.add_argument('-top',
                    action='store',
                    dest='structure',
                    help='Structure file corresponding to trajectory',
                    type=str,
                    required=True
                    )
inputs.add_argument('-traj',
                    action='store',
                    dest='trajectory',
                    help='Trajectory',
                    type=str,
                    required=True
                    )
inputs.add_argument('-sel',
                    action='store',
                    dest='sel',
                    help='Atom selection',
                    type=str,
                    default='name CA'
                    )
inputs.add_argument('-tau',
                    action='store',
                    dest='covariance_tau',
                    default=None,
                    type=int,
                    help='Lag time for constructing a time-lagged correlation matrix',
                    )
inputs.add_argument('-align',
                    action='store_true',
                    help='Align to atom selection before calculating?',
)
inputs.add_argument('-ax',
                    action='store',
                    dest='axis_label',
                    default=None,
                    help='Label for axes',
)
inputs.add_argument('-o',
                    action='store',
                    dest='out_name',
                    help='Output prefix for text and png',
                    type=str,
                    required=True
                    )



# Parse into useful form
UserInput = parser.parse_args()

# Process trajectory
trajectory = Reader.DCD(topology_path=UserInput.structure, trajectory_path=UserInput.trajectory).load()
trajectory = Selector.Slice(trajectory=trajectory, atom_selection=UserInput.sel).select()

if UserInput.align:
    trajectory = Processor.Aligner(trajectory=trajectory, atom_selection=UserInput.sel).process()

# Make correlation matrix

if UserInput.covariance_tau:
    correlation_matrix = Correlation.TimeLagged(
        trajectory=trajectory, covariance_tau=UserInput.covariance_tau
    ).calculate()
    title = 'Correlation Matrix with tau = {0}'.format(UserInput.covariance_tau)
else:
    correlation_matrix = Correlation.Pearson(trajectory=trajectory).calculate()
    title = 'Correlation Matrix'

# Save HeatMap

if UserInput.axis_label:
    Plotter.UnityPColor(y=correlation_matrix,
                        out_name=UserInput.out_name+'.png',
                        x_label=UserInput.axis_label,
                        y_label=UserInput.axis_label,
                        title=title
                        ).plot()
else:
    Plotter.UnityPColor(y=correlation_matrix,
                        out_name=UserInput.out_name + '.png',
                        x_label=UserInput.sel,
                        y_label=UserInput.sel,
                        title=title
                        ).plot()

Saver.Array(
    array=correlation_matrix,
    out_name=UserInput.out_name+'.txt'
).save()
