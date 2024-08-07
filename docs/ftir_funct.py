# -*- coding: utf-8 -*-
#######################################################################
# This file is part of FTIR-EBSD repository                           #
#                                                                     #
# Filename: ftir_funct.py                                             #
#                                                                     #
# Copyright (c) 2024                                                  #
#                                                                     #
# ftir_funct is free software: you can redistribute it and/or modify  #
# it under the terms of the GNU General Public License as published   #
# by the Free Software Foundation, either version 3 of the License,   #
# or (at your option) any later version.                              #
#                                                                     #
# ftir_funct is distributed in the hope that it will be useful,       #
# but WITHOUT ANY WARRANTY; without even the implied warranty of      #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the        #
# GNU General Public License for more details.                        #
#                                                                     #
# You should have received a copy of the GNU General Public License   #
# along with PyRockWave. If not, see <http://www.gnu.org/licenses/>.  #
#                                                                     #
# Project Repository: https://github.com/bertopadron/FTIR-EBSD        #
#######################################################################

# Import statements
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as r
from scipy.optimize import minimize, differential_evolution, dual_annealing
from contourpy import contour_generator

# Function definitions

# ============================================================================ #
# REFERENCE FRAMES AND CHANGE OF COORDINATES                                   #
# ============================================================================ #

def sph2cart(r, azimuth, polar=np.deg2rad(90)):
    """ Convert from spherical/polar (magnitude, azimuth, polar) to
    cartesian coordinates. Azimuth and polar angles are as used in
    physics (ISO 80000-2:2019) and in radians. If the polar angle is
    not given, the coordinate is assumed to lie on the XY plane.

    Parameters
    ----------
    r : int, float or array
        radial distance (magnitud of the vector)
    azimuth : int, float or array with values between 0 and 2*pi
        azimuth angle respect to the x-axis direction in radians
    polar : int, float or array with values between 0 and pi,
        polar angle respect to the zenith (z) direction in radians
        optional. Optinal, defaults to np.deg2rad(90)

    Returns
    -------
    numpy ndarrays (1d)
        three numpy 1d arrays with the cartesian x, y, and z coordinates
    """

    x = r * np.sin(polar) * np.cos(azimuth)
    y = r * np.sin(polar) * np.sin(azimuth)
    z = r * np.cos(polar)

    return x, y, z


def cart2sph(x, y, z):
    """Converts 3D rectangular cartesian coordinates to spherical polar
    coordinates.

    Parameters
    ----------
    x, y, z : float or array_like
        Cartesian coordinates.

    Returns
    -------
    r, theta, phi : float or array_like
        Spherical coordinates:
        - r: radial distance,
        - theta: inclination angle (range from 0 to π),
        - phi: azimuthal angle (range from 0 to 2π).

    Notes
    -----
    This function follows the ISO 80000-2:2019 norm (physics convention).
    The input coordinates (x, y, z) are assumed to be in a right-handed
    Cartesian system. The spherical coordinates are returned in the order
    (r, theta, phi). The angles theta and phi are in radians.
    """
    r = np.sqrt(x**2 + y**2 + z**2)

    # calculate the inclination - polar angle
    theta = np.arccos(z / r)
    
    # Calculate the azimuthal angle ensuring that phi is within [0, 2π)
    phi = np.arctan2(y, x)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    
    # if inclination is 0 or 180 set azimuth to 0
    phi[np.isclose(theta, 0)] = 0
    phi[np.isclose(theta, np.deg2rad(180))] = 0

    return r, phi, theta


def regular_S2_grid(n_squared=100):
    """_summary_

    Parameters
    ----------
    n : int, optional
        _description_, by default 100
    degrees : bool, optional
        _description_, by default False
    """

    azimuths = np.linspace(0, 2*np.pi, n_squared)
    polar = np.arccos(1 - 2 * np.linspace(0, 1, n_squared))
    
    return np.meshgrid(polar, azimuths)


def equispaced_S2_grid(n=20809, degrees=False, hemisphere=None):
    """Returns an approximately equispaced spherical grid in
    spherical coordinates (azimuthal and polar angles) using
    a modified version of the offset Fibonacci lattice algorithm.

    Note: Matemathically speaking, you cannot put more than 20
    perfectly evenly spaced points on a sphere. However, there
    are good-enough ways to approximately position evenly
    spaced points on a sphere.

    See also:
    https://arxiv.org/pdf/1607.04590.pdf
    https://github.com/gradywright/spherepts
    https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere

    Parameters
    ----------
    n : int, optional
        the number of points, by default 20809

    degrees : bool, optional
        whether you want angles in degrees or radians,
        by default False (=radians)

    hemisphere : None, 'upper' or 'lower'
        whether you want the grid to be distributed
        over the entire sphere, the upper hemisphere,
        or the lower hemisphere.

    Returns
    -------
    numpy ndarrays
        two numpy 1d arrays with azimuth and polar coordinates
    """

    # set sample size
    if hemisphere is None:
        n = n - 2
    else:
        n = (n * 2) - 2

    # get epsilon value based on sample size
    epsilon = _set_epsilon(n)

    golden_ratio = (1 + 5 ** 0.5) / 2
    i = np.arange(0, n)

    # estimate polar (phi) and theta (azimutal) angles in radians
    theta = 2 * np.pi * i / golden_ratio
    phi = np.arccos(1 - 2 * (i + epsilon) / (n - 1 + 2 * epsilon))

    # place a datapoint at each pole, it adds two datapoints removed before
    theta = np.insert(theta, 0, 0)
    theta = np.append(theta, 0)
    phi = np.insert(phi, 0, 0)
    phi = np.append(phi, np.pi)

    if degrees is False:
        if hemisphere == 'upper':
            return theta[phi <= np.pi/2] % (2*np.pi), phi[phi <= np.pi/2]
        elif hemisphere == 'lower':
            return theta[phi >= np.pi/2] % (2*np.pi), phi[phi >= np.pi/2]
        else:
            return theta % (2*np.pi), phi
    else:
        if hemisphere == 'upper':
            return np.rad2deg(theta[phi <= np.pi/2]) % 360, np.rad2deg(phi[phi <= np.pi/2])
        elif hemisphere == 'lower':
            return np.rad2deg(theta[phi >= np.pi/2]) % 360, np.rad2deg(phi[phi >= np.pi/2])
        else:
            return np.rad2deg(theta) % 360, np.rad2deg(phi)


def _set_epsilon(n):
    """Internal method used by the funtion
    equispaced_S2_grid.
    """
    if n >= 40_000:
        return 25
    elif n >= 1000:
        return 10
    elif n >= 80:
        return 3.33
    else:
        return 2.66


# ============================================================================ #
# FTIR EQUATIONS                                                               #
# ============================================================================ #

def Tvalues(trans, azimuth, polar):
    """ Calculates the transmission value for any direction in
    spherical coordinates using the equation (5) of Asimov et al.
    (2006) for a especific wavelength ignoring the sample thickness
    (i.e. assume thickness=1).

    Parameters
    ----------
    trans : a tuple of size 3
        tuple containeing the transmission values along a-axis (Ta),
        b-axis (Tb), and c-axis (Tc). -> (Ta, Tb, Tc)
    azimuth : int or float between 0 and 2*pi
        azimuthal angle respect to the a-axis in radians
    polar : int or float between 0 and pi
        polar angle respect to the c-axis in radians, i.e. inclination

    Returns
    -------
    numpy array
        the calculated T values for any given orientation

    Note
    ----
    Ta and Tb are interchanged with respect to Asimov's
    so that Ta is aligned with the x-axis when shifted
    to Cartesian coordinates.
    """

    # extract T values
    Ta, Tb, Tc = trans

    return Ta * np.cos(azimuth)**2 * np.sin(polar)**2 + \
           Tb * np.sin(azimuth)**2 * np.sin(polar)**2 + \
           Tc * np.cos(polar)**2


def calc_unpol_absorbance(A_max, A_min):
    return -np.log10((10**-A_max + 10**-A_min) / 2)


def T_on_plane(trans, azimuth, polar):
    """_summary_ Based on Sambridge et al.

    Parameters
    ----------
    trans : _type_
        _description_
    azimuth : _type_
        _description_
    polar : _type_
        _description_
    """

    # extract T values
    Ta, Tb, Tc = trans

    return 0.5 * (
        Ta * (np.cos(polar) ** 2 * np.cos(azimuth) ** 2 + np.sin(azimuth) ** 2)
        + Tb * (np.cos(polar) ** 2 * np.sin(azimuth) ** 2 * np.cos(azimuth) ** 2)
        + Tc * np.sin(polar) ** 2
    )


# ============================================================================ #
# EXTRACT ENVELOPE SECTIONS                                                    #
# ============================================================================ #

def extract_XY_section(x, y, z):
    """ It uses the matplolib contour function to get the values
    and spherical coordinates of T values within the XY plane.
    The contour function uses the marching squares algorithm
    to fing the intersection at a defined level.

    Caveat: THIS IS SLOW! USE ONLY FOR CREATING PLOTS

    Parameters
    ----------
    x : _type_
        _description_
    y : _type_
        _description_
    z : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    # estimate the contour at z=0 (i.e. XY plane)
    section_xy = plt.contour(x, y, z, levels=[0])

    # get the vertice coordinates (array-like)
    coordinates = section_xy.allsegs[0][0]

    # get vector lengths (i.e. T values within the XY plane)
    T = np.linalg.norm(coordinates, axis=1)

    # get the angle of the vector (in radians)
    angles = np.arctan2(coordinates[:, 1], coordinates[:, 0])

    # Convert angles to the range 0-2π (0-360 degrees)
    angles = np.degrees(angles) % 360

    # Convert angles to the range 0-360 degrees clockwise
    #angles = (90 - angles) % 360

    # remove figure
    plt.close("all")

    df = pd.DataFrame({
        'x': coordinates[:, 0],
        'y': coordinates[:, 1],
        'T': T,
        'angles': angles
    })

    return df  #.sort_values('angles')


def extract_XY_section_fast(x, y, z):
    """ It uses ContourPy to get the values and spherical coordinates
    of T values within the XY plane.

    Parameters
    ----------
    x : _type_
        _description_
    y : _type_
        _description_
    z : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    # estimate the contour at z=0 (i.e. XY plane)
    sections = contour_generator(x, y, z)

    # get the vertice coordinates of the contour z=0 (array-like)
    coordinates = sections.lines(0)[0]

    # get vector lengths (i.e. T values within the XY plane)
    T = np.linalg.norm(coordinates, axis=1)

    df = pd.DataFrame({
        'x': coordinates[:, 0],
        'y': coordinates[:, 1],
        'T': T,
    })

    return df


def extract_XY_section_fast2(x, y, z):
    """ It uses ContourPy to get the values and spherical coordinates
    of T values within the XY plane.
    Parameters
    ----------
    x : _type_
        _description_
    y : _type_
        _description_
    z : _type_
        _description_
    Returns
    -------
    _type_
        _description_
    """

    # estimate the contour at z=0 (i.e. XY plane)
    sections = contour_generator(x, y, z)

    # get the vertice coordinates of the contour z=0 (array-like)
    coordinates = sections.lines(0)[0]

    # get vector lengths (i.e. T values within the XY plane)
    T = np.linalg.norm(coordinates, axis=1)

    # get the angle of the vector (in radians)
    angles = np.arctan2(coordinates[:, 1], coordinates[:, 0])

    # Convert angles to the range 0-2π (0-360 degrees)
    angles = np.degrees(angles) % 360

    df = pd.DataFrame({
        'x': coordinates[:, 0],
        'y': coordinates[:, 1],
        'T': T,
        'angles': angles
    })

    return df


# ============================================================================ #
# CRYSTALLOGRAPHY                                                              #
# ============================================================================ #

def rotate(coordinates, euler_ang, invert=False):
    """ Rotate points in 3D cartesian space using the Bunge convention
    in degrees with extrinsic rotation. This is just a wrapper for the
    r.from_euler() Scipy method for convenience

    Parameters
    ----------
    coordinates : tuple of size 3
        a tuple containing the cartesian coordinates of this form:
        (x, y, z). variables x, y and z can be scalars or arrays.
    euler_ang : tuple of size 3
        a tuple containing the three euler angles in degrees
        using Bunge convention -> (z, x, z)

    Returns
    -------
    three numpy arrays containing the x, y, z coordinates respectively

    Example
    -------
    x, y, z = rotate(coordinates=(x, y, z), euler_ang=(30, 0, 40))
    """
   
    # create a ndarray to vectorize the rotation operation (n, x, 3) or (n, 3)
    if coordinates[0].ndim == 2:
        coordinates = np.dstack(coordinates)
    elif coordinates[0].ndim == 1:
        coordinates = np.vstack(coordinates).T
    else:
        print('check array dimension!')

    # define a rotation in euler space (Bunge) for extrinsic rotations
    rotation = r.from_euler('zxz', [euler_ang[0], euler_ang[1], euler_ang[2]],
                            degrees=True)
    
    if invert is True:
        rotation = rotation.inv()

    # apply rotation
    new_coordinates = coordinates @ rotation.as_matrix().T
    
    if coordinates[0].ndim == 2:
        return new_coordinates[:, :, 0], new_coordinates[:, :, 1], new_coordinates[:, :, 2]
    else:
        return new_coordinates[:, 0], new_coordinates[:, 1], new_coordinates[:, 2]


def calc_misorientation(euler1, euler2, precision=3):
    """Calculate the misorientation angle between two crystals.

    Parameters
    ----------
    euler1 : array_like (3,)
        Euler angles representing the orientation of the first
        crystal in degrees (Bunge convention).
    euler2 : array_like (3,)
        Euler angles representing the orientation of the second
        crystal in degrees (Bunge convention).

    Note
    ----
    This method assumes that the rotation matrices represent
    proper rotations (i.e., rotations without reflection or
    inversion). This is a general method and should work for
    any crystal symmetry but for large misorientation angles
    it will not give the lowest angle (i.e. disorientation)
    but a misorientation angle as it does not take into
    account the particular symmetry of the crystal. In
    principle, this is not a problem as this function is
    designed to deal with small misorientation angles.

    Returns
    -------
    float
        The misorientation angle between the two crystals
        in degrees.
    """

    # Convert Euler angles to rotation matrices
    R1 = r.from_euler("zxz", euler1, degrees=True)
    R2 = r.from_euler("zxz", euler2, degrees=True)

    # calc rotation
    rotation = R1.inv() * R2

    # Get the magnitude(s) of the rotation(s): angle(s) in radians
    misorientation_rad = rotation.magnitude()
    #misorientation_rad = np.arccos((np.trace(rotation.as_matrix()) - 1) / 2)

    return np.around(np.rad2deg(misorientation_rad), precision)


def symmetrise_orthorhombic(Euler_angles,
                            input_unit='radians',
                            output_unit='radians'):
    """
    Apply symmetry operations for an orthorhombic crystal
    to estimate all the equivalent orientations.

    Parameters:
    -----------
    Euler_angles : array_like (3,)
        Euler angles (φ1, Φ, φ2) in degrees or radians.
        Bunge convention
    input_unit : str
        Unit of the input angles ('degrees' or 'radians').
        Default is 'radians'.
    output_unit : str
        Unit of the output angles ('degrees' or 'radians').
        Default is 'radians'.

    Returns:
    --------
        list: List of numpy arrays representing all
        equivalent orientations in the specified output unit.
    """
    # Convert input angles to radians if necessary
    if input_unit == 'degrees':
        euler_angles = np.radians(Euler_angles)
    else:
        print("angles in radians assumed, otherwise use input_unit='degrees'")

    # Extract Euler angles
    phi1, Phi, phi2 = euler_angles

    # Define symmetry operations (rotations of 180 degrees around each axis)
    sym_rotations = [
        (phi1, Phi, phi2),                           # identity
        (np.pi + phi1, np.pi - Phi, np.pi - phi2),   # 180 Rotation around z-axis (φ1)
        (phi1, Phi, np.pi + phi2),                   # 180 Rotation around x-axis (Φ)
        (np.pi + phi1, np.pi - Phi, 2*np.pi - phi2)  # 180-degree rotation about the z-axis followed by a 180-degree rotation about the x-axis
    ]

    # Apply symmetry operations to the input Bunge angles
    equivalent_orientations = []
    for rotation in sym_rotations:
        equivalent_orientations.append(rotation)

        # Apply inversion operation (change the direction of the orientation vector)
        equivalent_orientations.append((-rotation[0], -rotation[1], -rotation[2]))  # Inversion operation

    # Convert output angles to degrees if necessary
    if output_unit == 'degrees':
        equivalent_orientations = [np.degrees(angles) for angles in equivalent_orientations]

    return equivalent_orientations


def calc_disorientation(euler1, euler2, all=False):
    """Calculate the disorientation angle, i.e. minimum
    misorientation angle between two orthorhombic crystals.

    Parameters
    ----------
    euler1 : array_like (3,)
        Euler angles representing the orientation of the
        first crystal in degrees (Bunge convention).
    euler2 : array_like (3,)
        Euler angles representing the orientation of the
        second crystal in degrees (Bunge convention).
    all : bool, default False
        if True, it returs all the misorientation angles

    Returns
    -------
    float
        The misorientation angle between the two crystals
        in degrees.
    """

    # covert euler 1 to rotation matrix
    R1 = r.from_euler("zxz", euler1, degrees=True)

    # symmetrize 2nd orientation
    equivalent_orientations = symmetrise_orthorhombic(
        euler2, input_unit="degrees", output_unit="degrees"
    )

    misorientations = np.empty(len(equivalent_orientations))

    for index, orientation in enumerate(equivalent_orientations):
        R2 = r.from_euler("zxz", orientation, degrees=True)
        rotation = R1.inv() * R2
        misorientation_rad = rotation.magnitude()
        misorientations[index] = misorientation_rad
    
    if all:
        return np.around(np.rad2deg(misorientations), 3)
    else:
        return np.around(np.rad2deg(misorientations.min()), 3)


def explore_Euler_space(step=1,
                        lower_bounds=(0, 0, 0),
                        upper_bounds=(90, 90, 180)):
    """Returns a Numpy array with different combinations
    of Euler angles in degrees to explore the Euler space
    based on a defined step size. It assumes a orthorhombic
    symmetry where the Euler angle ranges for the fundamental
    zone are:
    varphi1: 0-90
        Phi: 0-90
    varphi2: 0-180

    Parameters
    ----------
    step : int, optional
        the resolution, by default 1 degree
    lower_bounds : tuple, optional
        range of Euler angles, by default (0, 0, 0)
    upper_bounds : tuple, optional
        range of Euler angles, by default (90, 90, 180)

    Returns
    -------
    _type_
        _description_
    """
    lang1, lang2, lang3 = lower_bounds
    ang1, ang2, ang3 = upper_bounds

    phi1 = np.arange(lang1, ang1 + 1, step)
    Phi = np.arange(lang2, ang2 + 1, step)
    phi2 = np.arange(lang3, ang3 + 1, step)

    # Create a meshgrid of all possible combinations
    phi1, Phi, phi2 = np.meshgrid(phi1, Phi, phi2, indexing='ij')

    # Stack the angles along the third axis to create the final array
    array = np.stack((phi1, Phi, phi2), axis=-1)

    # Reshape the array to size 3*n
    array = array.reshape(-1, 3)

    return array


def compact_axis_angle_to_axis_angle(rotvec, spherical=False):
    """Compute axis-angle from compact axis-angle representation.
    Reimplemented from pytransform3d library

    We assume active rotations.

    Parameters
    ----------
    rotvec : array-like, shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z).
    spherical : boolean
        if True, axis is reported on spherical coordinates
        instead of cartesian.

    Returns
    -------
    axis_angle : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle). The angle is
        constrained to [0, pi].
    custom axis_angle : array-like, shape (3,)
        Axis of rotation and rotation angle: (azimuth, inclination, angle).
        The angle is constrained to [0, pi]. The unit vector is in polar
        coordinates.
    """
    # check input
    rotvec = np.asarray(rotvec, dtype=np.float64)
    if rotvec.ndim != 1 or rotvec.shape[0] != 3:
        raise ValueError(
            "Expected axis and angle in array with shape (3,), "
            "got array-like object with shape %s" % (rotvec.shape,)
        )

    angle = np.linalg.norm(rotvec)

    if angle == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])

    axis = rotvec / angle
    axis_angle = np.hstack((axis, (angle,)))

    if spherical:
        x, y, z, _ = axis_angle
        _, azimuth, polar = cart2sph(x, y, z)
        return np.array([azimuth, polar, angle])
    else:
        return axis_angle


def plot_crystal_axes(ax, r, name=None, offset=(0, 0, 0), scale=1):
    """
    This function customizes a matplotlib axis (`ax`) to visualize
    crystal axes.

    Parameters
    ----------
    ax : object
        A matplotlib axis object to be modified.
    r : _type_
        A scipy.spatial.transform.Rotation object representing
        the crystallographic rotation.
    name : _type_, optional
        A string specifying the name of the crystal structure
        to be displayed in the center, by default None.
    offset : tuple, optional
        A tuple of three floats representing the offset for
        the axes and label placement, by default (0, 0, 0).
    scale : int, optional
        A float value to scale the length of the plotted
        axes, by default 1

    Returns
    -------
         None. The function modifies the input axis object
         (`ax`) in-place.
    """
    # Define a color list for crystallographic axes (colorblind-safe RGB)
    colors = ("#FF6666", "#005533", "#1199EE")

    # store the given offset location
    loc = np.array([offset, offset])

    # Loop through each axis (x, y, z) and its corresponding color
    for i, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis), colors)):

        axlabel = axis.axis_name # Get the name of the axis

        # Set the label name and the color for labels, lines and ticks.
        axis.set_label_text(axlabel)
        axis.label.set_color(c)
        axis.line.set_color(c)
        axis.set_tick_params(colors=c)
    
        # Create, locate and orientate the line segments
        line = np.zeros((2, 3))
        line[1, i] = scale
        line_rot = r.apply(line)    # rotate
        line_plot = line_rot + loc  # translate

        # plot line segments
        ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c)    

        # Calculate the position for the axis labels
        text_loc = line[1] * 1.2
        text_loc_rot = r.apply(text_loc)
        text_plot = text_loc_rot + loc[0]

        # plot the axis label text
        ax.text(*text_plot, axlabel.upper(), color=c, va="center", ha="center")

    # If a name is provided, add it to the center of the plot
    ax.text(
        *offset,
        name,
        color="k",
        va="center",
        ha="center",
        bbox={"fc": "w", "alpha": 0.8, "boxstyle": "circle"},
    )

    return None


# ============================================================================ #
# MINIMIZATION                                                                 #
# ============================================================================ #

def objective_function(euler_ang, measurements, params):
    """
    Objective function to minimize the difference between
    measured and theoretical T values
    """
    # extract variables
    Ta, Tb, Tc = params
    T_measured = measurements[:, 0]
    azimuths = np.deg2rad(measurements[:, 1])
    polar = np.deg2rad(measurements[:, 2])
    e1, e2, e3 = euler_ang

    # convert from spherical to cartesian coordinates
    x, y, z = sph2cart(r=T_measured, azimuth=azimuths, polar=polar)
    
    # apply rotation to measures using Eules angles (Bunge convention)
    # Note that the order of euler angles are inverted and the sign changed
    x2, y2, z2 = rotate(coordinates=(x, y, z), euler_ang=(-e3, -e2, -e1))
    
    # convert back to spherical coordinates
    T_measured, azimuths, polar = cart2sph(x2.ravel(), y2.ravel(), z2.ravel())
    
    # estimate theoretical T values
    T_theoretical = Tvalues(trans=(Ta, Tb, Tc), azimuth=azimuths, polar=polar)

    return np.sum(np.abs(T_measured - T_theoretical))**2


def find_orientation(measurements, params, num_guesses=20, silent=True, tolerance=None):
    """
    Given a set of points in 3D space, determine if they fall on the surface
    defined by the function T. If the points do not fall on the surface,
    apply a rotation to the points and check again until the points fall on
    the surface. Return the Euler angles that rotate the points to the surface.
    This used a gradient-based method but using multiple initial guesses
    chosen randomly and uniformly throughout Euler space.


    Parameters
    ----------
    measurements : numpy array
        The measurements, where each tuple contains (T, azimuths, polar).
    params : tuple of size 3
        tuple containing the transmission values along a-axis (Ta),
        b-axis (Tb), and c-axis (Tc). -> (Ta, Tb, Tc)
    num_guesses : int
        Number of initial guesses to try.
    tolerance : float or None
        tolerance for determining if a point is on the surface
    """

    best_result = None
    best_objective_value = float('inf')
    bounds = [(0, 90), (0, 90), (0, 180)]

    # Generate initial guesses
    initial_guesses = np.around(np.random.uniform([0, 0, 0], [90, 90, 180], size=(num_guesses, 3)), 0)

    for euler_ang in initial_guesses:
        # Minimize
        result = minimize(fun=objective_function,
                          x0=euler_ang,
                          args=(measurements, params),
                          bounds=bounds,
                          tol=tolerance)

        # Update result if the current one is better
        if result.fun < best_objective_value:
            best_objective_value = result.fun
            best_result = result

    if silent:
        return best_result
    else:
        print(f'Calculated Orientation: {np.around(best_result.x, 1)}')
        return best_result


def find_orientation_diffevol(measurements, params, silent=True, **kwargs):
    """This is just a wrapper of the Scipy's differential_evolution
    algorithm

    http://en.wikipedia.org/wiki/Differential_evolution
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html


    Parameters
    ----------
    measurements : _type_
        _description_
    params : _type_
        _description_
    """

    bounds = [(0, 90), (0, 90), (0, 180)]

    # Perform global optimization using differential evolution
    result = differential_evolution(func=objective_function, 
                                    bounds=bounds,
                                    args=(measurements, params),
                                    **kwargs)
    
    if silent:
        return result
    else:
        print(f'Calculated Orientation: {np.around(result.x, 1)}')
        return result


def find_orientation_annealing(measurements, params, silent=True, **kwargs):
    """This is just a wrapper of the Scipy's dual_annealing
    algorithm

    Parameters
    ----------
    measurements : _type_
        _description_
    params : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    
    bounds = [(0, 90), (0, 90), (0, 180)]

    # Perform global optimization using dual annealing
    result = dual_annealing(func=objective_function,
                            bounds=bounds,
                            args=(measurements, params),
                            **kwargs)

    if silent:
        return result
    else:
        print(f'Calculated Orientation: {np.around(result.x, 1)}')
        return result


def find_orientation_bruteforce(measurements, params, step=3):
    """_summary_

    Parameters
    ----------
    measurements : _type_
        _description_
    params : _type_
        _description_
    step : int, optional
        _description_, by default 3

    Returns
    -------
    _type_
        _description_
    """

    euler = explore_Euler_space(step)
    diff = np.empty(euler.shape[0])

    for index, euler_ang in enumerate(euler):

        val = objective_function(euler_ang,
                                 measurements,
                                 params=params)
        diff[index] = val

    print(f'Calculated Orientation: {euler[diff.argmin()]}')
    print(f'diff = {diff.min()}')
    return euler[diff.argmin()]


def find_nearest(df, values):
    """find the index of the nearest value in
    a pandas dataframe

    Parameters
    ----------
    df : _type_
        _description_
    values : _type_
        _description_
    """

    indexes = []
    for value in values:
        indexes.append((np.abs(df - value)).idxmin())
    return indexes


if __name__ == '__main__':
    pass
else:
    print('module FTIR v.2024.4.11 imported')

# End of file
