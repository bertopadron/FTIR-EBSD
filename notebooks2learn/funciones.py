import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as r
from contourpy import contour_generator


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
        angle respect to the a-axis in radians
    polar : int or float between 0 and pi
        angle respect to the c-axis in radians

    Returns
    -------
    numpy array
        the calculated T values for any given orientation
    """

    # extract Tx values
    Ta, Tb, Tc = trans

    return Tb * np.cos(azimuth)**2 * np.sin(polar)**2 + \
           Ta * np.sin(azimuth)**2 * np.sin(polar)**2 + \
           Tc * np.cos(polar)**2


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

    return np.around(x, decimals=6), np.around(y, decimals=6), np.around(z, decimals=6)


def regular_S2_grid(n_squared=100, degrees=False):
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


def extract_XY_section(x, y, z):
    """ It uses the matplolib contour function to get the values
    and spherical coordinates of T values within the XY plane.
    The contour function uses he marching squares algorithm
    to fing the intersection at a defined level.

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
    angles = (90 - angles) % 360

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


def rotate(coordinates, euler_ang):
    """ Rotate points in 3D cartesian space using the Bunge convention
    in degrees with intrinsic rotation. This is just a wrapper for the
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
    # create a ndarray to vectorize the rotation operation
    coordinates = np.dstack(coordinates)

    # define a rotation in euler space (Bunge) for intrinsic rotations
    rotation = r.from_euler('zxz', [euler_ang[0], euler_ang[1], euler_ang[2]],
                            degrees=True)

    # apply rotation
    new_coordinates = coordinates @ rotation.as_matrix().T

    return new_coordinates[:, :, 0], new_coordinates[:, :, 1], new_coordinates[:, :, 2]


def explore_Euler_space(step=1):
    """Returns a Numpy array with different combinations
    of Euler angles in degrees to explore the Euler space
    based on a defined step size. It assumes a orthorhombic
    symmetry where angles range like this:

    phi1: 0-90
    theta: 0-180
    phi2: 0-90

    Parameters
    ----------
    step : int, optional
        _description_, by default 1
    """

    phi1 = np.arange(0, 90 + step, step)
    theta = np.arange(0, 180 + step, step)
    phi2 = np.arange(0, 90 + step, step)

    # Create a meshgrid of all possible combinations
    phi1, theta, phi2 = np.meshgrid(phi1, theta, phi2, indexing='ij')

    # Stack the angles along the third axis to create the final array
    array = np.stack((phi1, theta, phi2), axis=-1)

    # Reshape the array to size 3*n
    array = array.reshape(-1, 3)

    return array


def calc_unpol_absorbance(A_max, A_min):
    return -np.log10((10**-A_max + 10**-A_min) / 2)
