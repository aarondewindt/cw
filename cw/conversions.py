"""Package containing conversion functions."""

import numpy as np
import math

# Unit conversions
slugs_ft3_to_kg_m3 = 515.378819
kg_m3_to_slugs_ft3 = 1 / slugs_ft3_to_kg_m3

slugs_ft2_to_kg_m2 = 1.35581795
kg_m2_to_slugs_ft2 = 1 / slugs_ft2_to_kg_m2

slugs_to_kg = 14.5939029
kg_to_slugs = 1 / slugs_to_kg

ft_to_meter = 0.3048
meter_to_ft = 1 / ft_to_meter

newton_to_lbs = 0.22480894244319
lbs_to_N = 1 / newton_to_lbs

fps_to_mps = 0.3048000
mps_to_fps = 1 / fps_to_mps

kts_to_mps = knot_to_mps = 0.514444
mps_to_kts = mps_to_knot = 1 / knot_to_mps

ft2_to_m2 = 0.092903
m2_to_ft2 = 1 / ft2_to_m2

deg_to_rad = math.pi / 180
rad_to_deg = 180 / math.pi

fps2_to_mps2 = 0.3048000
mps2_to_fps2 = 1 / fps2_to_mps2

psf_to_pa = 47.880258888889
pa_to_psf = 1 / psf_to_pa

lbf_to_n = 4.4482216
n_to_lbf = 1 / lbf_to_n


def q_to_euler(q):
    """Converts Quaternions to Euler angles.
    
    Parameters
    ----------
    q : array_like
        Array holding Quaternions.
       
    Returns
    -------
    phi : float
        `phi` angle in radians.
    theta :float
        `theta` angle in radians.
    psi : float
        `psi` angle in radians.
    """
    
    phi = np.arctan2(2*(q[0]*q[1]+q[2]*q[3]),(q[0]**2+q[3]**2-q[1]**2-q[2]**2))
    theta = np.arcsin(2*(q[0]*q[2]-q[1]*q[3]))
    psi = np.arctan2(2*(q[0]*q[3]+q[1]*q[2]),(q[0]**2+q[1]**2-q[2]**2-q[3]**2))
    
    return phi, theta, psi


def dcm_to_q(DCM):
    """Converts Direction Cosine Matrix to Quaternions.
    
    Parameters
    ----------
    DCM : array_like
        Direct Cosine Matrix.
        
    Returns
    -------
    q : numpy_array
        Quaternions.
        
    Raises
    ------
    ValueError
        If an invalid DCM is passed.
    """
    
    q = np.array([[1.],[1.],[1.],[1.]])

    r = []
    r.append(1 + DCM[0,0] + DCM[1,1] + DCM[2,2])
    r.append(1 + DCM[0,0] - DCM[1,1] - DCM[2,2])
    r.append(1 - DCM[0,0] + DCM[1,1] - DCM[2,2])
    r.append(1 - DCM[0,0] - DCM[1,1] + DCM[2,2])
    r = np.array(r)
    case = np.argmax(r)

    if case == 0:
        r = math.sqrt(r[case])
        q[0] = 0.5 * r
        q[1] = 0.5 * (DCM[1][2]-DCM[2][1])/r
        q[2] = 0.5 * (DCM[2][0]-DCM[0][2])/r
        q[3] = 0.5 * (DCM[0][1]-DCM[1][0])/r
        
    elif case == 1:
        r = math.sqrt(r[case])
        q[0] = 0.5 * (DCM[1][2]-DCM[2][1])/r
        q[1] = 0.5 * r
        q[2] = 0.5 * (DCM[0][1]+DCM[1][0])/r
        q[3] = 0.5 * (DCM[2][0]+DCM[0][2])/r

    elif case == 2:
        r = math.sqrt(r[case])
        q[0] = 0.5 * (DCM[2][0] - DCM[0][2])/r
        q[1] = 0.5 * (DCM[0][1] + DCM[1][0])/r
        q[2] = 0.5 * r
        q[3] = 0.5 * (DCM[1][2] + DCM[2][1])/r
        
    elif case == 3:
        r = math.sqrt(r[case])
        q[0] = 0.5 * (DCM[0][1] - DCM[1][0])/r
        q[1] = 0.5 * (DCM[2][0] + DCM[0][2])/r
        q[2] = 0.5 * (DCM[1][2] + DCM[2][1])/r
        q[3] = 0.5 * r
    else:
        raise ValueError("Invalid DCM")
    return q


def q_to_dcm(q):
    """Converts Quaternions to Direction Cosine Matrix.
    
    Parameters
    ----------
    q : array_like
        Quaternions.
    
    Returns
    -------
    DCM : numpy_array
        Direction Cosine Matrix.
    """
    
    DCM = np.array([[q[0,0]**2+q[1,0]**2-q[2,0]**2-q[3,0]**2, 2*q[1,0]*q[2,0]+2*q[0,0]*q[3,0], 2*q[1,0]*q[3,0]-2*q[0,0]*q[2,0]],
                    [2*q[1,0]*q[2,0]-2*q[0,0]*q[3,0], q[0,0]**2-q[1,0]**2+q[2,0]**2-q[3,0]**2, 2*q[2,0]*q[3,0]+2*q[0,0]*q[1,0]],
                    [2*q[1,0]*q[3,0]+2*q[0,0]*q[2,0], 2*q[2,0]*q[3,0]-2*q[0,0]*q[1,0], q[0,0]**2-q[1,0]**2-q[2,0]**2+q[3,0]**2]])
    return DCM


def omega_to_qdot(omega, quat, k=1.0):
    """Converts Rotational Rates (omega) to Quaternion rates.
    
    Parameters
    ----------
    omega : array_like
        Rotational Rates.
    quat : array_like
        Rotational Rate column vector
    k : float, optional
        Quaternion normalization (the default is 1.0).

    Returns
    -------
    q : numpy_array
        Quaternion Rates.
    """

    p = omega[0,0]
    q = omega[1,0]
    r = omega[2,0]

    e = k * (1-(quat[0,0]*quat[0,0]+quat[1,0]*quat[1,0]+quat[2,0]*quat[2,0]+quat[3,0]*quat[3,0]))

    qdot = 0.5*np.array(  [[  e*quat[0,0] -p*quat[1,0] -q*quat[2,0] -r*quat[3,0]],
                           [  p*quat[0,0] +e*quat[1,0] +r*quat[2,0] -q*quat[3,0]],
                           [  q*quat[0,0] -r*quat[1,0] +e*quat[2,0] +p*quat[3,0]],
                           [  r*quat[0,0] +q*quat[1,0] -p*quat[2,0] +e*quat[3,0]]])
    
    return qdot


def qdot_to_omega(q):
    """Not implemented"""
    raise NotImplementedError()


def dcm_to_euler(DCM, rtype='zyx', unit='rad'):
    """Converts Direction Cosine Matrix to Euler angles.   
    
    Parameters
    ----------
    DCM : array_like
        Direction Cosine Matrix.
    rtype : string, optional
        Type of Euler angle to convert to (the default is zyx).
    unit : string
        Unit of the angles, can be 'deg' or 'rad' (default is rad).
        
    Returns
    -------
    r0 : float  
    r1 : float
    r2 : float
    """

    rtype = rtype.lower()

    r0= 0
    r1= 0
    r2= 0

    if  rtype == 'zyx':
        r0, r1, r2 = _threeaxisrot(DCM[0,1], DCM[0,0], -(DCM[0,2]), DCM[1,2], DCM[2,2])
    elif rtype == 'zyz':
        r0, r1, r2 = _twoaxisrot(DCM[2,1],DCM[2,0], DCM[2,2], DCM[1,2], -DCM[0,2])
    elif rtype == 'zxy':
        r0, r1, r2 =  _threeaxisrot(-DCM[1,0], DCM[1,1], DCM[1,2], -DCM[0,2], DCM[2,2])
    elif rtype == 'zxz':
        r0, r1, r2 = _twoaxisrot(DCM[2,0], -DCM[2,1], DCM[2,2], DCM[0,2], DCM[1,2])
    elif rtype == 'yxz':
        r0, r1, r2 =  _threeaxisrot(DCM[2,0], DCM[2,2], -DCM[2,1], DCM[0,1], DCM[1,1])
    elif rtype == 'yxy':
        r0, r1, r2 = _twoaxisrot(DCM[1,0], DCM[1,2], DCM[1,1], DCM[0,1], -DCM[2,1])     
    elif rtype == 'yzx':
        r0, r1, r2 =  _threeaxisrot(-DCM[0,2], DCM[0,0], DCM[0,1], -DCM[2,1], DCM[1,1])
    elif rtype == 'yzy':
        r0, r1, r2 =  _twoaxisrot(DCM[1,2], -DCM[1,0], DCM[1,1], DCM[2,1], DCM[0,1])
    elif rtype == 'xyz':
        r0, r1, r2 =  _threeaxisrot(-DCM[2,1], DCM[2,2], DCM[2,0], -DCM[1,0], DCM[0,0])
    elif rtype == 'xyx':
        r0, r1, r2 =  _twoaxisrot(DCM[0,1], -DCM[0,2], DCM[0,0], DCM[1,0], DCM[2,0])
    elif rtype == 'xzy':
        r0, r1, r2 =  _threeaxisrot(DCM[1,2], DCM[1,1], -DCM[1,0], DCM[2,0], DCM[0,0])
    elif rtype == 'xzx':
        r0, r1, r2 =  _twoaxisrot(DCM[0,2], DCM[0,1], DCM[0,0], DCM[2,0], -DCM[1,0])

    if unit == 'rad':
        return r0, r1, r2
    else:
        return np.rad2deg(r0), np.rad2deg(r1), np.rad2deg(r2)


def _threeaxisrot(r11,r12,r21,r31,r32):
    r0 = np.arctan2( r11, r12 );
    r1 = np.arcsin( r21 );
    r2 = np.arctan2( r31, r32 );
    return r0, r1, r2
    

def _twoaxisrot(r11, r12, r21, r31, r32):
    r0 = np.arctan2( r11, r12 );
    r1 = np.arccos( r21 );
    r2 = np.arctan2( r31, r32 );
    return r0, r1, r2 


def euler_to_dcm(phi, theta, psi, rtype = "zyx", unit='rad'):
    """Converts Euler angles to Direction Cosine Matrix.
    
    Parameters
    ----------
    phi : float
        First rotation angle (Yaw angle in case of "zyx").
    theta : float
        Second rotation angle (Pitch angle in case of "zyx").
    psi : float
        Third rotation angle (Roll angle in case of "zyx").
    rtype : string, optional
        Rotational order (default is zyx).
    unit : string, optional
        Unit of the returned angles, rad or deg (default is rad).
        
    Returns
    -------
    DCM : numpy_array
        Direction Cosine Matrix.
        
    Raises
    ------
    ValueError
        If the passed unit is not 'deg' of 'rad'.
    ValueError
        If the passed rotational order is not supported.
    """

    if   unit.lower() == 'deg':
        angles = [phi/180*math.pi, theta/180*math.pi, psi/180*math.pi]
    elif unit.lower() == 'rad':
        angles = [phi, theta, psi]
    else:
        raise ValueError('Invalid unit')

    cang = np.cos(angles)
    sang = np.sin(angles)
    rtype = str.lower(rtype)

    if rtype == "zyx":
        return np.array([[cang[1]*cang[0],cang[1]*sang[0],-sang[1]], \
                         [sang[2]*sang[1]*cang[0] - cang[2]*sang[0],sang[2]*sang[1]*sang[0] + cang[2]*cang[0],sang[2]*cang[1]], \
                         [cang[2]*sang[1]*cang[0] + sang[2]*sang[0],cang[2]*sang[1]*sang[0] - sang[2]*cang[0],cang[2]*cang[1]]])
    elif rtype == "zyz":
        return np.array([[cang[0]*cang[2]*cang[1] - sang[0]*sang[2],sang[0]*cang[2]*cang[1] + cang[0]*sang[2],-sang[1]*cang[2]], \
                         [-cang[0]*cang[1]*sang[2] - sang[0]*cang[2],-sang[0]*cang[1]*sang[2] + cang[0]*cang[2],sang[1]*sang[2]], \
                         [cang[0]*sang[1],sang[0]*sang[1],cang[1]]])
    elif rtype == "zxy":
        return np.array([[cang[2]*cang[0] - sang[1]*sang[2]*sang[0],cang[2]*sang[0] + sang[1]*sang[2]*cang[0],-sang[2]*cang[1]], \
                         [-cang[1]*sang[0],cang[1]*cang[0],sang[1]], \
                         [sang[2]*cang[0] + sang[1]*cang[2]*sang[0],sang[2]*sang[0] - sang[1]*cang[2]*cang[0],cang[1]*cang[2]]])
    elif rtype == "zxz":
        return np.array([[-sang[0]*cang[1]*sang[2] + cang[0]*cang[2],cang[0]*cang[1]*sang[2] + sang[0]*cang[2],sang[1]*sang[2]], \
                         [-sang[0]*cang[2]*cang[1] - cang[0]*sang[2],cang[0]*cang[2]*cang[1] - sang[0]*sang[2],sang[1]*cang[2]], \
                         [sang[0]*sang[1],-cang[0]*sang[1],cang[1]]])
    elif rtype == "yxz":
        return np.array([[cang[0]*cang[2] + sang[1]*sang[0]*sang[2],cang[1]*sang[2],-sang[0]*cang[2] + sang[1]*cang[0]*sang[2]], \
                         [-cang[0]*sang[2] + sang[1]*sang[0]*cang[2],cang[1]*cang[2],sang[0]*sang[2] + sang[1]*cang[0]*cang[2]], \
                         [sang[0]*cang[1],-sang[1],cang[1]*cang[0]]])
    elif rtype == "yxy":
        return np.array([[-sang[0]*cang[1]*sang[2] + cang[0]*cang[2],sang[1]*sang[2],-cang[0]*cang[1]*sang[2] - sang[0]*cang[2]], \
                         [sang[0]*sang[1],cang[1],cang[0]*sang[1]], \
                         [sang[0]*cang[2]*cang[1] + cang[0]*sang[2],-sang[1]*cang[2],cang[0]*cang[2]*cang[1] - sang[0]*sang[2]]])
    elif rtype == "yzx":
        return np.array([[cang[0]*cang[1],sang[1],-sang[0]*cang[1]], \
                         [-cang[2]*cang[0]*sang[1] + sang[2]*sang[0],cang[1]*cang[2],cang[2]*sang[0]*sang[1] + sang[2]*cang[0]], \
                         [sang[2]*cang[0]*sang[1] + cang[2]*sang[0],-sang[2]*cang[1],-sang[2]*sang[0]*sang[1] + cang[2]*cang[0]]])
    elif rtype == "yzy":
        return np.array([[cang[0]*cang[2]*cang[1] - sang[0]*sang[2],sang[1]*cang[2],-sang[0]*cang[2]*cang[1] - cang[0]*sang[2]], \
                         [-cang[0]*sang[1],cang[1],sang[0]*sang[1]], \
                         [cang[0]*cang[1]*sang[2] + sang[0]*cang[2],sang[1]*sang[2],-sang[0]*cang[1]*sang[2] + cang[0]*cang[2]]])
    elif rtype == "xyz":
        return np.array([[cang[1]*cang[2],sang[0]*sang[1]*cang[2] + cang[0]*sang[2],-cang[0]*sang[1]*cang[2] + sang[0]*sang[2]], \
                         [-cang[1]*sang[2],-sang[0]*sang[1]*sang[2] + cang[0]*cang[2],cang[0]*sang[1]*sang[2] + sang[0]*cang[2]], \
                         [sang[1],-sang[0]*cang[1],cang[0]*cang[1]]])
    elif rtype == "xyx":
        return np.array([[cang[1],sang[0]*sang[1],-cang[0]*sang[1]], \
                         [sang[1]*sang[2],-sang[0]*cang[1]*sang[2] + cang[0]*cang[2],cang[0]*cang[1]*sang[2] + sang[0]*cang[2]], \
                         [sang[1]*cang[2],-sang[0]*cang[2]*cang[1] - cang[0]*sang[2],cang[0]*cang[2]*cang[1] - sang[0]*sang[2]]])
    elif rtype == "xzy":
        return np.array([[cang[2]*cang[1],cang[0]*cang[2]*sang[1] + sang[0]*sang[2],sang[0]*cang[2]*sang[1] - cang[0]*sang[2]], \
                         [-sang[1],cang[0]*cang[1],sang[0]*cang[1]], \
                         [sang[2]*cang[1],cang[0]*sang[1]*sang[2] - sang[0]*cang[2],sang[0]*sang[1]*sang[2] + cang[0]*cang[2]]])
    elif rtype == "xzx":
        return np.array([[cang[1],cang[0]*sang[1],sang[0]*sang[1]], \
                         [-sang[1]*cang[2],cang[0]*cang[2]*cang[1] - sang[0]*sang[2],sang[0]*cang[2]*cang[1] + cang[0]*sang[2]], \
                         [sang[1]*sang[2],-cang[0]*cang[1]*sang[2] - sang[0]*cang[2],-sang[0]*cang[1]*sang[2] + cang[0]*cang[2]]])
    else:
        raise ValueError('Invalid rotation order')


def flat_to_lla(x, lla0):
    """Estimate geodetic latitude, longitude, and altitude from flat Earth
    position.
    
    Parameters
    ----------
    x : array_like
        Position w.r.t. lla0
    lla0 : array_like
        Reference location of latitude longitude and altitude, for the origin
        of the estimation and the origin of the flat Earth coordinate system.
        
    Returns
    -------
    lla : numpy_array
        Latitude longitude and altitude.
    """

    # Data from WGS84
    a =  6378137.0
    f  = 1/298.257223563
    
    # Some speed optimization
    F = (2*f-f*f)
    rlat = lla0[0,0]
    srlat2 = math.sin(rlat)

    Rn = a/math.sqrt(1-F*srlat2)
    Rm = Rn*((1-F)/(1-F*srlat2))

    # Change in lla 
    dLat = x[0,0]*math.atan2(1,Rm)
    dLon = x[1,0]*math.atan2(1,Rn*math.cos(rlat))
    da = -x[2,0]

    return np.add(lla0, np.array([[dLat],[dLon],[da]]))


def ecef_to_wgs84(Xc):
    """Converts coordinates (x,y,z) in the ECEF frame to coordinates 
    (lat,lon,h) in the WGS84 frame [1]_ .
    
    Parameters
    ----------
    Xc : array_like
        Coordinates (x,y,z) in ECEF frame (m).
    
    Returns
    -------
    lla : numpy_array
        Lla (geodetic latitude (rad), geodetic longitude (rad), altitude 
        above WGS84 ellipsoid (m).

    References
    ----------
    .. [1] Fukushima, T. (2006) "Transformation from geocentric rectangular
        to geodetic coordinates accelerated by Halley's method"
    """
    
    x = Xc[0,0] ; y = Xc[1,0] ; z = Xc[2,0]

    # Data from WGS84
    a = 6378137.0
    finv = 298.257223563

    # Algorithm (Python port of Fortran code)
    f = 1/finv
    e2 = (2-f)*f
    ec2 = 1-e2
    ec = math.sqrt(ec2)
    b = a*ec
    c = a*e2

    s0 = abs(z)
    p2 = x*x+y*y
    if (p2 != 0):
        p = math.sqrt(p2)
        zc = ec*s0
        c0 = ec*p
        c02 = c0*c0
        c03 = c02*c0
        s02 = s0*s0
        s03 = s02*s0
        a02 = c02+s02
        a0 = math.sqrt(a02)
        a03 = a02*a0
        s1 = zc*a03+c*s03
        c1 = p*a03-c*c03
        cs0c0 =c*c0*s0
        b0 = 1.5*cs0c0*((p*s0-zc*c0)*a0-cs0c0)
        s1 = s1*a03-b0*s0
        cc = ec*(c1*a03-b0*c0)
        s12 = s1*s1
        cc2 = cc*cc

        lat = math.atan(s1/cc)
        h = (p*cc+s0*s1-a*math.sqrt(ec2*s12+cc2))/math.sqrt(s12+cc2)
    else:
        lat = math.pi/2
        h = s0-b
    lon = math.atan2(y,x)
    if (z < 0):
        lat = -lat
    return np.array([[lat],[lon],[h]])


def wgs84_to_ecef(lla ,unit="rad"):
    """Converts coordinates (latitude, longitude, altitude) in WGS84 to 
    coordinates (x,y,z) in the ECEF reference frame.

    Parameters
    ----------
    lla : array_like
        Geodetic latitude, geodetic longitude and altitude above the WGS84
        ellipsoid (m).
    unit : string, optional
        Unit of the latitude and longitude, deg or rad (default is rad).
        
    Returns
    -------
    x : numpy_array
        Coordinates in ECEF frame (m).
    """
    
    # Data from the WGS84 model 
    a = 6378137.0
    finv = 298.257223563

    # Data from array
    lat = lla[0,0]
    lon = lla[1,0]
    h = lla[2,0]

    # Unit conversion
    unit = unit.lower()
    if unit == "deg":
        lat = math.radians(lat)
        lon = math.radians(lon)

    # Intermediate stuff
    b = a*(1-1/finv)
    psi = math.atan(math.tan(lat)*b/a)          # Magically works at the singularities! 
    r = a*math.cos(psi)+h*math.cos(lat)

    x = r*math.cos(lon)
    y = r*math.sin(lon)
    z = b*math.sin(psi) + h*math.sin(lat)

    return np.array([[x],[y],[z]])


def ecef_to_spherical(Xc):
    x = Xc[0,0] ; y = Xc[1,0] ; z = Xc[2,0]
    r = math.sqrt(x**2+y**2+z**2)
    lat_gc = math.asin(z/r)
    lon_gc = math.atan2(y,x)
   
    return [r,lat_gc,lon_gc]


def hgeo_to_hpot(h_geo, Re=6371000.0):
    """Calculates the geopotential altitude from the geometric altitude.
    
    Parameters
    ----------
    h_geo : float
        Geometric altitude.
    Re : float, optional
        Earth's radius (default is 6371000.0).
        
    Returns
    -------
    hpot : float
        Geopotential altitude.
    """
    
    return Re/(Re+h_geo)*h_geo


def state_to_keplerian(Xi,Vi):
    """
    Calculates the keplerian orbital elements from a given state vector.
    Angular orbital elements are returned in degrees. Only to be used 
    with ECI position and ECI velocity!

    Algorithm as described in AE4878 - Mission Geometry and Orbit Design.
    
    Parameters
    ----------
    Xi : array_like
        Position in the inertial reference frame.
    Vi : array_like
        Position in the inertial reference frame.
        
    Returns
    -------
    
    """
    mu = 3.986004418e14

    # Vectors
    Xi_vec = np.reshape(Xi,(3,))
    Vi_vec = np.reshape(Vi,(3,))
    h_vec = np.cross(Xi_vec,Vi_vec)
    N_vec = np.cross(np.array([0,0,1]),h_vec)

    # Scalars
    Xi = np.linalg.norm(Xi_vec)
    Vi = np.linalg.norm(Vi_vec)
    h = np.linalg.norm(h_vec)
    N = np.linalg.norm(N_vec)

    # Unit vectors
    Xi_hat = Xi_vec/Xi
    N_hat = N_vec/N

    Nxy = math.sqrt(N_vec[0]**2+N_vec[1]**2) 
    a = 1/(2/Xi - (Vi_vec[0]**2+Vi_vec[1]**2+Vi_vec[2]**2)/mu)
    e_vec = np.cross(Vi_vec,h_vec)/mu - Xi_hat
    e = np.linalg.norm(e_vec) ; e_hat = e_vec/e
    i = math.degrees(math.acos(h_vec[2]/h))
    RAAN = math.degrees(math.atan2(N_vec[1]/Nxy,N_vec[0]/Nxy))

    if np.dot(np.cross(N_hat,e_vec),h_vec) > 0:
        sign = 1
    else:
        sign = -1
    AOP = math.degrees(sign*math.acos(np.dot(e_hat,N_hat)))
    
    if np.dot(np.cross(e_vec,Xi_vec),h_vec) > 0:
        sign = 1
    else:
        sign = -1

    # Apparently this may throw an error due to round-off error, so I'm putting in this check.
    dotprod = np.dot(Xi_hat,e_hat)
    if dotprod > 1.0:
        dotprod = 1.0
    elif dotprod < -1.0:
        dotprod = -1.0

    TA = math.degrees(sign*math.acos(dotprod))

    return [a,e,i,RAAN,AOP,TA]


def keplerian_to_state(a,e,inc,RAAN,AOP,TA):
    """Converts the keplerian orbital elements into a state vector in the ECI
    frame. Angular elements should be provided in degrees.

    Algorithm as described in AE4878 - Mission Geometry and Orbit Design.
    
    Parameters
    ----------
    a : float
        a
    e : float
        e
    inc : float
        inc
    RAAN : float
        RAAN
    AOP : float
        AOP
    TA : float
        TA
        
    Returns
    -------
    Xi : array_like
        Position in the inertial reference frame.
    Vi : array_like
        Position in the inertial reference frame.
    """
    mu = 3.986004418e14

    # Conversion to radians
    inc = math.radians(inc)
    RAAN = math.radians(RAAN)
    AOP = math.radians(AOP)
    TA = math.radians(TA)

    # Precomputing values
    cinc = math.cos(inc) ; sinc = math.sin(inc) 
    cRAAN = math.cos(RAAN) ; sRAAN = math.sin(RAAN)
    cAOP = math.cos(AOP) ; sAOP = math.sin(AOP)
    cTA = math.cos(TA) ; sTA = math.sin(TA)

    l1 = cRAAN*cAOP - sRAAN*sAOP*cinc
    l2 = -cRAAN*sAOP - sRAAN*cAOP*cinc
    m1 = sRAAN*cAOP + cRAAN*sAOP*cinc
    m2 = -sRAAN*sAOP + cRAAN*cAOP*cinc
    n1 = sAOP*sinc
    n2 = cAOP*sinc
    r = a*(1-e**2)/(1+e*cTA)
    ksi = r*cTA
    eta = r*sTA
    H = math.sqrt(mu*a*(1-e**2))
    #print([l1,l2,m1,m2,n1,n2])
    #print([ksi,eta])
    #print(H)

    # Calculating state vector
    x = l1*ksi+l2*eta
    y = m1*ksi+m2*eta
    z = n1*ksi+n2*eta
    Vx = mu/H*(-l1*sTA+l2*(e+cTA))
    Vy = mu/H*(-m1*sTA+m2*(e+cTA))
    Vz = mu/H*(-n1*sTA+n2*(e+cTA))

    Xi = np.array([[x],[y],[z]])
    Vi = np.array([[Vx],[Vy],[Vz]])

    return [Xi, Vi]


def haversine(lat1,lon1,lat2,lon2,radius=6371000.0):
    """
    Haversine function, used to calculate the distance between two points on the surface of a sphere. Good approximation
    for distances between two points on the surface of the Earth if the correct local curvature is used and the points are
    relatively close to one another.

    Parameters
    ----------
    lat1 : array_like
        Latitude of first point in radians
    lon1 : array_like
        Longitude of first point in radians
    lat2: array_like
        Latitude of second point in radians
    lon2 : array_like
        Longitude of second point in radians
    radius : float
        Local radius of curvature in meters

    Returns
    -------
    distance : float
        Distance between the two locations
    """
    distance = 2*radius*math.asin(math.sqrt(math.sin((lat2-lat1)/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2))
    return distance


def angle_to_rot_2d(angle: float) -> np.ndarray:
    """
    Calculates a 2d rotation matrix from an angle in radians.

    :param angle: Angle in radians.
    :return: 2d rotation matrix
    """
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def rot_to_angle_2d(rotation_matrix: np.ndarray) -> float:
    """
    Get the rotation angle of a 2d rotation matrix.

    :param rotation_matrix:
    :return: Angle of rotation in radians
    """
    return np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
