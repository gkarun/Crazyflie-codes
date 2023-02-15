"""Single CF: takeoff, do homing, land."""

import numpy as np
import time as pytime
from pycrazyswarm import Crazyswarm
from matplotlib import rc
from scipy.integrate import solve_ivp
from threading import Thread, Lock
import time

TAKEOFF_Z = 0.5
TAKEOFF_DURATION = 2.5
INIT_POS_DURATION = 5.0
GOTO_DURATION = 0.3
WAIT_DURATION = 10.0
CMDPOS_PUB_INTERVAL = 0.1

landFlag = False

initPos = np.array((1.0, 1.0, 1.5))
target_pos = np.array((-1,-1,0.3))

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


class pubCmdPos(Thread):
   def __init__(self, threadID, cf):
      Thread.__init__(self)
      self.cf = cf
      self.setPoint = np.array((0, 0, 0))
      self.yaw = 0.0
      self.mutexSetPoint = Lock()
   def run(self):
      global landFlag
      #print("thread: Inside Run --------------")
      while not landFlag:
          self.mutexSetPoint.acquire() 
          self.cf.cmdPosition(self.setPoint, yaw=0.0)
          self.mutexSetPoint.release()
          time.sleep(CMDPOS_PUB_INTERVAL)
          #print("thread: after sleep landFlag = {}".format(landFlag))
      print("thread : after exiting loop")
      self.cf.land(targetHeight=0.05, duration=TAKEOFF_DURATION)
      time.sleep(TAKEOFF_DURATION + 1.0)
      return


def main():
    global landFlag
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]
    sendPos = pubCmdPos(0, cf)
    
    cf.takeoff(targetHeight=TAKEOFF_Z, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + 1.0)
    sendPos.mutexSetPoint.acquire() 
    sendPos.setPoint=initPos
    sendPos.mutexSetPoint.release()
    sendPos.start() 
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
        if(np.linalg.norm(dpos) == 0):
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
        
        t_span = np.linspace(0, 1, 100)
        #x0 = np.array([0, 0, 0, alpha, beta])
        #sol = solve_ivp(lambda t, x: dubins3D(t,x,u_alpha,u_beta),[t_span[0],t_span[-1]],x0, t_eval = t_span, rtol = 1e-5)
        x0 = np.array([current_pos[0], current_pos[1], current_pos[2], alpha, beta])
        sol = solve_ivp(lambda t, x: dubins3D(t,x,u_alpha,u_beta),[t_span[0],t_span[-1]],x0, t_eval = t_span, rtol = 1e-5)
        goal_pos = np.array([sol.y[0,-1],sol.y[1,-1],sol.y[2,-1]])
        sendPos.mutexSetPoint.acquire() 
        sendPos.setPoint=goal_pos
        sendPos.mutexSetPoint.release()
      
        prev_pos = current_pos
        elapsed_time = pytime.process_time() - start_time
        print(elapsed_time)
        #cf.goTo(goal_pos, yaw=0.0, duration=GOTO_DURATION, relative = False, groupMask = 0)
        timeHelper.sleep(GOTO_DURATION)

    print("Reached")
    timeHelper.sleep(WAIT_DURATION)
    landFlag = True
    print("before join")
    sendPos.join(WAIT_DURATION)
    print("after join")
    timeHelper.sleep(WAIT_DURATION)
    
if __name__ == "__main__":
    main()
