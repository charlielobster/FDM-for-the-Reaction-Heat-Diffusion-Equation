def solver_dufort_frankel(x, dx, dt, t_final, a, k):
    """
    DuFort-Frankel Explicit Scheme.
    Unconditionally stable, meaning it won't explode even if a*dt/dx^2 > 0.5.
    Requires a leap-frog start (uses 3 time-levels).
    """
    nx = len(x)
    nt = int(round(t_final / dt))
    
    # Level n-1 (Past) and Level n (Present)
    C_past = np.where(np.abs(x) < 1.0, 1.0, 0.0)
    
    # Bootstrapping the first step (Level n) using a single stable step of FTCS
    # or the analytical solution to ensure a clean start
    C_curr = np.copy(C_past)
    r = a * dt / (dx**2)
    
    # Temporary small dt step just to populate the first time level cleanly
    C_curr[1:-1] = C_past[1:-1] + r * (C_past[2:] - 2*C_past[1:-1] + C_past[:-2]) - k * dt * C_past[1:-1]
    
    # Pre-calculate DuFort-Frankel scaling denominators
    denom = 1.0 + 2.0 * r + k * dt
    alpha = (1.0 - 2.0 * r - k * dt) / denom
    beta = (2.0 * r) / denom
    
    # 3-Level Time-stepping loop
    for n in range(1, nt):
        C_next = np.zeros(nx)
        
        # DuFort-Frankel Leap-Frog Stencil
        C_next[1:-1] = (alpha * C_past[1:-1] + 
                        beta * (C_curr[2:] + C_curr[:-2]))
        
        # Enforce boundaries
        C_next[0] = 0.0
        C_next[-1] = 0.0
        
        # Roll the time vectors forward
        C_past = np.copy(C_curr)
        C_curr = C_next
        
    return C_curr
