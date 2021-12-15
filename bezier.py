import numpy as np


# Points = numpy.array((nops, 2)) where
# 'nops' is a number of interpolation points

def input_points(fname: str) -> "Points | None":
    """provides to read points from a text file
    Argument:
    ---------
        fname   a string such that value of 'fname' + ".pnt" is a correct name
                of the file containing interpolation points
    Effect:
    -------
        the corresponding list of 2D points if the attempt of reading is
        successful or None
    Constraint:
    -----------
        each line in the input file is formatted as follows
        "float'\t'float'\t'float'\n'"
    """
    try:
        file = open(fname + ".pnt")
        ptext, nops = [], 0
        for line in file:
            ptext.append(line)
            nops += 1
        # nops contains number of points
        file.close()
        points = np.zeros((nops,2))
        for ic, item in enumerate(ptext):
            items = item.split()[: 2]
            point = np.array(list(map(float, items)))
            points[ic] = point
        return points
    except FileNotFoundError:
        print("File is not found. Check!")
        return None
    except ValueError:
        print("File contains invalid data. Check!")
        file.close()
        return None
        

def special_Bezier_interpolation(
    points: "numpy.array(nops, 2)"
) -> "numpy.array(4 * (nops - 1), 2)":
    """computes control points of a cubic Bezier-curve
    Argument:
    ---------
        points  contains an interpolation points list where 'nops' is the number
                of the points
    Effect:
    -------
        the corresponding list containing control points of the cubic Bezier-
        curve built by interpolation algorithm free for oscillation effect
    Constraint:
        nops > 3
    """
    nops = np.shape(points)[0]
    N = nops - 1
    Delta = np.zeros(N)
    for ic in range(N):
        Delta[ic] = np.linalg.norm(points[ic + 1] - points[ic])
    # distances-array has been formed
    M = np.eye(N - 1)
    M[0, 1] = Delta[1] / 2. / (Delta[0] +  Delta[1])
    M[N - 2, N - 3] = Delta[N - 2] / 2. / (Delta[N - 2] + Delta[N - 1])
    for ic in range(1, N - 2):
        M[ic][ic - 1] = Delta[ic] / 2. / (Delta[ic] + Delta[ic + 1])
        M[ic][ic] = 1.
        M[ic][ic + 1] = Delta[ic + 1] / 2. / (Delta[ic] + Delta[ic + 1])
    # the matrix of the system has been formed
    P = np.zeros((N - 1, 2))
    for ic in range(0, N - 1):
        P[ic] = (
            3. / Delta[ic] / (Delta[ic] + Delta[ic + 1]) * points[ic] -
            3. / Delta[ic] / Delta[ic + 1] * points[ic + 1] +
            3. / Delta[ic + 1] / (Delta[ic] + Delta[ic + 1]) * points[ic + 2]
        )
    # the right side of the system has been formed
    Q = np.linalg.solve(M, P)
    # the system has been solved
    Bezier = np.zeros((4 * N, 2))
    for ic in range(4 * N):
        iic, ind = divmod(ic, 4)
        if iic == 0:
            if ind == 0: 
                Bezier[ic] = points[iic]
            elif ind == 1:
                Bezier[ic] = (
                    (2. / 3.) * points[0] + (1. / 3.) * points[1] -
                    Delta[0] * Delta[0] / 18. * Q[0]
                )
            elif ind == 2:
                Bezier[ic] = (
                    (1. / 3.) * points[0] + (2. / 3.) * points[1] -
                    Delta[0] * Delta[0] / 9. * Q[0]
                )
            else:   # ind == 3
                Bezier[ic] = points[iic + 1]
        elif iic == N - 1:
            if ind == 0: 
                Bezier[ic] = points[iic]
            elif ind == 1:
                Bezier[ic] = (
                    (2. / 3.) * points[N - 1] + (1. / 3.) * points[N] -
                    Delta[N - 1] * Delta[N - 1] / 9. * Q[N - 2]
                )
            elif ind == 2:
                Bezier[ic] = (
                    (1. / 3.) * points[N - 1] + (2. / 3.) * points[N] -
                    Delta[N - 1] * Delta[N - 1] / 18. * Q[N - 2]
                )
            else:   # ind == 3
                Bezier[ic] = points[iic + 1]
        else:  # 0 < iic < N-1
            if ind == 0:
                Bezier[ic] = points[iic]
            elif ind == 1:
                Bezier[ic] = (
                    (2. / 3.) * points[iic] + (1. / 3.) * points[iic + 1] -
                    (1. / 9.) * Delta[iic] * Delta[iic] * Q[iic - 1] -
                    (1. / 18.) * Delta[iic] * Delta[iic] * Q[iic]
                )
            elif ind == 2:
                Bezier[ic] = (
                    (1. / 3.) * points[iic] + (2. / 3.) * points[iic + 1] -
                    (1. / 18.) * Delta[iic] * Delta[iic] * Q[iic - 1] -
                    (1. / 9.) * Delta[iic] * Delta[iic] * Q[iic]
                )
            else:   # ind == 3
                Bezier[ic] = points[iic + 1]
    # control points of the cubic Bezier-curve have been restored
    return Bezier
        
