import numpy as np
# ============================================================================
# Spatial Derivative Utilities for Radial Functions
# ============================================================================

def second_derivative_central(u, dr):
    """
    Compute second derivative using central differences.
    
    For interior points: u_rr[i] = (u[i+1] - 2*u[i] + u[i-1]) / dr^2
    Boundary points are set to 0 (will be handled separately by BCs)
    
    Args:
        u: array of function values
        dr: radial grid spacing
    
    Returns:
        u_rr: second derivative array
    """
    u_rr = np.zeros_like(u)
    u_rr[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dr**2
    return u_rr


def first_derivative_central(u, dr):
    """
    Compute first derivative using central differences.
    
    For interior points: u_r[i] = (u[i+1] - u[i-1]) / (2*dr)
    Boundary points use forward/backward differences.
    
    Args:
        u: array of function values
        dr: radial grid spacing
    
    Returns:
        u_r: first derivative array
    """
    u_r = np.zeros_like(u)
    u_r[1:-1] = (u[2:] - u[:-2]) / (2*dr)
    u_r[0] = (u[1] - u[0]) / dr  # forward difference at r=0
    u_r[-1] = (u[-1] - u[-2]) / dr  # backward difference at r=R
    return u_r


def laplacian_polar(u, r, dr):
    """
    Compute Laplacian in polar coordinates: ∇²u = u_rr + (1/r)*u_r
    
    This is valid for radially symmetric functions where ∂²u/∂θ² = 0
    
    Args:
        u: array of radial density values
        r: array of radial coordinates
        dr: radial grid spacing
    
    Returns:
        laplacian: ∇²u computed at all points
    """
    u_rr = second_derivative_central(u, dr)
    u_r = first_derivative_central(u, dr)
    
    # Avoid division by zero at r=0 by using L'Hopital's rule: lim(u_r/r) = u_rr as r->0
    laplacian = np.zeros_like(u)
    laplacian[0] = 2 * u_rr[0]  # At r≈0, (1/r)*u_r ≈ u_rr by L'Hopital
    laplacian[1:] = u_rr[1:] + u_r[1:] / r[1:]
    
    return laplacian
