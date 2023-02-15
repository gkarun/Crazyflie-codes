"""Single CF: takeoff, do homing, land."""

import numpy as np
import time as pytime
from pycrazyswarm import Crazyswarm
from matplotlib import rc
from scipy.integrate import solve_ivp


TAKEOFF_Z = 1.0
TAKEOFF_DURATION = 2.5
INIT_POS_DURATION = 5.0
GOTO_DURATION = 3
WAIT_DURATION = 10.0
initPos = (1.0, 1.0, 1.5)
target_pos = np.array((0,0,0.2))

V = 0.1
RC = 0.3

K_ALPHA = 1.1*V/RC
K_BETA = 1.1*V/RC

def dubins3D(t,x, u_alpha, u_beta):
    alpha = x[3];
    beta  = x[4];

    dx = V*np.cos(beta)*np.cos(alpha)
    dy = V*np.cos(beta)*np.sin(alpha)
    dz = V*np.sin(beta)

    dalpha = u_alpha
    dbeta = u_beta

    dx_dt = [dx, dy, dz, dalpha, dbeta]
    return dx_dt


def sgn(x):
    if x>0:
        return 1.0
    else :
        return -1.0

def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    cf.takeoff(targetHeight=TAKEOFF_Z, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + 1.0)
    cf.goTo(initPos, yaw=0.0, duration=INIT_POS_DURATION)
    timeHelper.sleep(INIT_POS_DURATION + 1.0)
    
    prev_pos = np.array((0, 0, 0))
    
    current_pos = cf.position();
    diff_pos = target_pos - current_pos
    # print(diff_pos)
    R = np.linalg.norm(diff_pos)
    theta = np.arctan2(diff_pos[1], diff_pos[0])
    phi   = np.arctan(diff_pos[2]/np.sqrt(diff_pos[0]**2 + diff_pos[1]**2))
    
    while R > RC:
        start_time = pytime.process_time()
        current_pos = cf.position();
        diff_pos = target_pos - current_pos
        R = np.linalg.norm(diff_pos)
        theta = np.arctan2(diff_pos[1], diff_pos[0])
        phi   = np.arctan(diff_pos[2]/np.sqrt(diff_pos[0]**2 + diff_pos[1]**2))
        dpos = current_pos - prev_pos
        if(np.linalg.norm(diff_pos) == 0):
            continue
        
        if R < RC:
            break

           
        alpha = np.arctan2( dpos[1], dpos[0] )
        beta = np.arctan( dpos[2]/np.sqrt(dpos[0]**2 + dpos[1]**2) )

        
        dyaw  = theta - alpha
        dyaw  = np.arctan2(np.sin(dyaw), np.cos(dyaw)) 
        dpitch = phi - beta
        #dpitch = np.arctan(np.sin(dpitch)/np.cos(dpitch))
        
        
        u_alpha = K_ALPHA*sgn(dyaw)
        u_beta  = K_BETA*sgn(dpitch)
        
        t_span = np.linspace(0, 3, 100)
        x0 = np.array([0, 0, 0, alpha, beta])
        sol = solve_ivp(lambda t, x: dubins3D(t,x,u_alpha,u_beta),[t_span[0],t_span[-1]],x0, t_eval = t_span, rtol = 1e-5)
        #x0 = np.array([current_pos[0], current_pos[1], current_pos[2], alpha, beta])
        #sol = solve_ivp(lambda t, x: dubins3D(t,x,u_alpha,u_beta),[t_span[0],t_span[-1]],x0, t_eval = t_span, rtol = 1e-5)
        goal_pos = np.array([sol.y[0,-1],sol.y[1,-1],sol.y[2,-1]])
        prev_pos = current_pos
        elapsed_time = pytime.process_time() - start_time
        print(elapsed_time)
        cf.goTo(goal_pos, yaw=0.0, duration=GOTO_DURATION, relative = True, groupMask = 0)
        #cf.goTo(goal_pos, yaw=0.0, duration=GOTO_DURATION, relative = False, groupMask = 0)
        timeHelper.sleep(GOTO_DURATION)
        
    print("Reached")
    timeHelper.sleep(WAIT_DURATION)
    cf.land(targetHeight=0.05, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + 1.0)


if __name__ == "__main__":
    main()
