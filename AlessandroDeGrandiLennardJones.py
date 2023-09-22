import pygame
import random
import sys
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time
import numpy as np

#
#PRESS P for plot and pause
#PRESS D for debug
#

DELAY = 0
DEBUG = True
SAVE_SCREENSHOT = False

RED= (255,0,0)
BLUE = (0,0,255)
GRAY = (127,127,127)
WHITE = (255,255,255)
BLACK = (0,0,0)

GRID_SIZE = 30
SCREEN_SIZE=900
UNIT_SIZE=SCREEN_SIZE//GRID_SIZE

####
SIGMA = 1
RC = 2.5*SIGMA #cutoff radius
EPSILON = 1
MASS = 1
DT = 0.01

THERMOSTAT_ON = True
T=1 #temperature
IV=2*T
MAX_T=5

INIT_N=300

#ID=GRID_SIZE/math.sqrt(INIT_N*SIGMA) #initial distance between particles
#print(ID)
ID=1
#INIT_N=300

###

#init Cell lists
CL={}
#N_CELLS=GRID_SIZE
CELL_SIZE=int(RC)
N_CELLS = GRID_SIZE//CELL_SIZE
for x in range(N_CELLS):
    for y in range(N_CELLS):
        CL[(x,y)]={}

class Cell:
    
    #maybe faster with dictionary
    ids=0
    allCells={}
    
    #SIZE= SCREEN_SIZE//GRID_SIZE
    SIZE= UNIT_SIZE

    def __init__(self, posX, posY, velocityX=0, velocityY=0, color=BLUE):
        Cell.ids+=1
        self.id=Cell.ids
        
        self.posX=posX
        self.posY=posY 
        self.velocityX=velocityX
        self.velocityY=velocityY
        self.accelerationX=0
        self.accelerationY=0
        self.forceX=0
        self.forceY=0
        self.potential=0
        self.kinetic=0
        self.color=color

        Cell.allCells[self.id]=self
        CL[(posX//CELL_SIZE,posY//CELL_SIZE)][self.id]=self

    def calculate_forces(self):
        self.forceX=0
        self.forceY=0
        self.potential=0
        self.distances=[]
        for x in range(-1,2):
            for y in range(-1,2):
                for _,cell in CL[((self.posX//CELL_SIZE)+x)%N_CELLS,((self.posY//CELL_SIZE)+y)%N_CELLS].items():
                    if(cell.id!=self.id):
                        dx = self.posX-cell.posX
                        dy = self.posY-cell.posY
                        
                        #periodic boundary conditions
                        if(dx>GRID_SIZE/2):
                            dx=dx-GRID_SIZE
                        elif(dx<=-GRID_SIZE/2):
                            dx=dx+GRID_SIZE
                        if(dy>GRID_SIZE/2):
                            dy=dy-GRID_SIZE
                        elif(dy<=-GRID_SIZE/2):
                            dy=dy+GRID_SIZE
                        
                        r = math.sqrt(dx**2+dy**2)
                        
                        
                        #print(f'id1={self.id} posX={self.posX} posY={self.posY} id2={cell.id} posX={cell.posX} posY={cell.posY} r={r} dx={dx} dy={dy} ')
                        if(r<RC and r>SIGMA*(69/100)):
                            self.distances.append(r)
                            F = (48/(r**2))*((SIGMA/r)**12 - 0.5*(SIGMA/r)**6) 
                            self.forceX+=F*dx/r
                            self.forceY+=F*dy/r

                            #potential energy
                            self.potential+=4*((SIGMA/r)**12 - (SIGMA/r)**6) - 4*((SIGMA/RC)**12 - (SIGMA/RC)**6)
            
    def update_position(self):
       
        new_posX = (self.posX + self.velocityX*DT + 0.5*self.accelerationX*DT**2)%GRID_SIZE
        new_posY = (self.posY + self.velocityY*DT + 0.5*self.accelerationY*DT**2)%GRID_SIZE
    
        #update cell list
        if((self.posX//CELL_SIZE,self.posY//CELL_SIZE) != (new_posX//CELL_SIZE,new_posY//CELL_SIZE)):
            CL[(self.posX//CELL_SIZE,self.posY//CELL_SIZE)].pop(self.id)
            CL[(new_posX//CELL_SIZE,new_posY//CELL_SIZE)][self.id]=self

        self.posX=new_posX
        self.posY=new_posY
    
    def update_velocity_accelleration(self):
        
        new_accelerationX = self.forceX/MASS
        new_accelerationY = self.forceY/MASS
        self.velocityX+=(self.accelerationX+new_accelerationX)*DT*0.5
        self.velocityY+=(self.accelerationY+new_accelerationY)*DT*0.5
        #self.velocityX+=new_accelerationX*DT
        #self.velocityY+=new_accelerationY*DT
        self.accelerationX=new_accelerationX
        self.accelerationY=new_accelerationY

        v=math.sqrt(self.velocityX**2+self.velocityY**2)
        #print(f'id={self.id} vx={self.velocityX} vy={self.velocityY} v={v}')
        #kinetic energy
        self.kinetic=0.5*MASS*v**2

        v*=255/MAX_T
        if(v>255):
            v=255
        self.color=(v,0,255-v)

    def total_velocity():
        tot_v=0
        for _,cell in Cell.allCells.items():
            tot_v+=math.sqrt(cell.velocityX**2+cell.velocityY**2)
        return tot_v
    
    def total_temperaure():
        #return Cell.total_velocity()/INIT_N
        return Cell.total_kinetic()/(3/2*INIT_N)

    def total_momentum():
        tot_mx=0
        tot_my=0
        for _,cell in Cell.allCells.items():
            tot_mx+=cell.velocityX*MASS
            tot_my+=cell.velocityY*MASS
        tot_m=math.sqrt(tot_mx**2+tot_my**2)
        #print(tot_mx,tot_my)
        return tot_m
    
    def total_potential():
        tot_pot=0
        for _,cell in Cell.allCells.items():
            tot_pot+=cell.potential
        return tot_pot
    
    def total_kinetic():
        tot_kin=0
        for _,cell in Cell.allCells.items():
            tot_kin+=cell.kinetic
        return tot_kin
    
    def total_energy():
        return Cell.total_potential()+Cell.total_kinetic()
    
    def average_radial_distribution_function():
        BINS_SIZE=0.05
        BINS= np.arange(BINS_SIZE,RC,BINS_SIZE)
        N_BINS= BINS.size
        RADIAL_DISTRIBUTION=[0]*N_BINS
        for _,cell in Cell.allCells.items():
            for r in cell.distances:
                for i,bin in enumerate(BINS):
                    if r>=bin and r<bin+BINS_SIZE:
                        RADIAL_DISTRIBUTION[i]+=1 
        
        AVG_DENSITY=INIT_N/(GRID_SIZE**2)
        for i,bin in enumerate(BINS):
            #normalized average radial distribution function
            #RADIAL_DISTRIBUTION[i]/=INIT_N
            RADIAL_DISTRIBUTION[i]/=AVG_DENSITY*2*math.pi*bin*BINS_SIZE*INIT_N
            
        return  BINS,RADIAL_DISTRIBUTION
    
    def berendsen_thermostat():

        current_temperature = Cell.total_temperaure()
        # Calculate scaling factor
        #DT/tau=0.0025
        scaling_factor = np.sqrt(1 + (0.0025) * ((T / current_temperature) - 1))

        # Scale velocities
        for _,cell in Cell.allCells.items():
            cell.velocityX *= scaling_factor
            cell.velocityY *= scaling_factor

    def update_desired_temperature(new_T):
        global T
        T=new_T
                     
###########
pygame.init()
pygame.display.set_caption('Press P or D')
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE), 0)
#WHITE = (255,255,255)


#list of all coordinates (x,y) in grid

lattice = [(x,y) for x in np.arange(0,GRID_SIZE,ID) for y in np.arange(0,GRID_SIZE,ID)]
#for x,y in lattice:
#    Cell(x,y)
initial_positions = random.sample(lattice, INIT_N)
for x,y in initial_positions:
    Cell(x,y)

    
#initial velocities, momentum=0
meanx=0
meany=0
for _,cell in Cell.allCells.items():
    #cell.velocityX=random.gauss(0,IV)
    #cell.velocityY=random.gauss(0,IV)
    cell.velocityX=random.uniform(-IV,IV)
    cell.velocityY=random.uniform(-IV,IV)
    meanx+=cell.velocityX
    meany+=cell.velocityY

meanx/=INIT_N
meany/=INIT_N

for _,cell in Cell.allCells.items():
    cell.velocityX-=meanx
    cell.velocityY-=meany

#for _,cell in Cell.allCells.items():
#    cell.velocityX=0
#    cell.velocityY=0
#Cell.allCells[1].velocityX=20
#Cell.allCells[1].velocityY=20


n_iterations=0

#plt.ion()
#fig=plt.figure()
total_t=[]
total_m=[]
total_p=[]
total_k=[]
total_e=[]


#print("momentum: ", total_m[-1])
end=False
start_time= time.time()
tot_time=0
while(not end):
    iter_time = time.time()
    n_iterations+=1
    
    #draw stuff
    draw_time = time.time()
    screen.fill(0)
    
    for _,cell in Cell.allCells.items():
        #pygame.draw.rect(screen,cell.color,(cell.posX*cell.SIZE, cell.posY*Cell.SIZE, Cell.SIZE, Cell.SIZE))
        pygame.draw.circle(screen,cell.color,(cell.posX*UNIT_SIZE, cell.posY*UNIT_SIZE), Cell.SIZE//2, width=0)
    
    #DRAW CL
    for x,y in CL.keys():
        pygame.draw.rect(screen, GRAY, (x*CELL_SIZE*UNIT_SIZE, y*CELL_SIZE*UNIT_SIZE, CELL_SIZE*UNIT_SIZE, CELL_SIZE*UNIT_SIZE), width=1)

    ##
    for _,cell in Cell.allCells.items():
        cell.update_position()
    for _,cell in Cell.allCells.items():
        cell.calculate_forces() 
    for _,cell in Cell.allCells.items():
        cell.update_velocity_accelleration()

    #plot stuff
    total_t.append(Cell.total_temperaure())
    total_m.append(Cell.total_momentum())
    total_p.append(Cell.total_potential())
    total_k.append(Cell.total_kinetic())
    total_e.append(Cell.total_energy())
    print("momentum: ", total_m[-1],end='     \r')

    if(THERMOSTAT_ON):
        Cell.berendsen_thermostat()
        
    #events
    for event in pygame.event.get():
        if(event.type == pygame.QUIT):
            pygame.display.quit()
            pygame.quit()
            end=True
            #sys.exit()
        if(event.type == pygame.KEYDOWN):
            if(event.key == pygame.K_UP):           
                DELAY-=10                
            if(event.key == pygame.K_DOWN):            
                DELAY+=10
            if(event.key == pygame.K_SPACE):
                 pygame.image.save(screen, f"-Iteration={n_iterations}-N={INIT_N}-dt{DT}.jpg") 
            #if(event.key == pygame.K_d):
            #    DEBUG=not DEBUG
            if(event.key == pygame.K_p):
                #subplots
                fig,ax = plt.subplots(4,1,figsize=(8, 10))
                
                ax[0].plot(range(n_iterations),total_m,label='Total momentum')
                ax[1].plot(range(n_iterations),total_e,label='Total energy')
                ax[1].plot(range(n_iterations),total_p,label='Total potential')
                ax[1].plot(range(n_iterations),total_k,label='Total kinetic')
                bins,avg_radial_distribution=Cell.average_radial_distribution_function()
                ax[2].plot(bins,avg_radial_distribution,label='Radial distribution function')
                ax[3].plot(range(n_iterations),total_t,label='Total temperature')
                if(THERMOSTAT_ON):
                    slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
                    slider = Slider(slider_ax, 'Desired Temp', 0, MAX_T, valinit=T, valstep=0.1)
                    slider.on_changed(Cell.update_desired_temperature)
                ax[0].legend()
                ax[1].legend()
                ax[2].legend()
                ax[3].legend()
                plt.show()
                
        if(event.type == pygame.MOUSEBUTTONDOWN):
            mousePosX,mousePosY= pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1]
            Cell(mousePosX/GRID_SIZE, mousePosY/GRID_SIZE)
            #for _ in range(10):
                #Cell((mousePosX//Cell.SIZE)+EATING_DISTANCE*2, mousePosY//Cell.SIZE, Cell.RABBIT)

    iter_time=time.time()-iter_time
    tot_time= time.time()-start_time
    #print(f'Iteration:{n_iterations} N_WOLVES={len(Cell.wolves)} N_RABBITS={len(Cell.rabbits)} -TIMING: sim={tot_time/60:.2f}m iter={iter_time:.3f}s : draw={draw_time:.3f}s move={time_move:.3f}s eat={time_eat:.3f}s repr={time_replicate:.3f}s surv={time_survive:.3f}s      ', end='\r')
    
    if(not end):
        pygame.display.update()     
        pygame.time.delay(DELAY)

    







