#!/usr/bin/env python

# Import libraries
from pylab import *
import numpy as np
import random
import math
from scipy.optimize import curve_fit

# Parameters of code
start_radius=100	# Radius at which random walkers are released
out_of_bounds=120	# Radius at which random walkers are considered to not reach cluster in sufficient amount of time (thrown out)

############################## START OF FUNCTION DEFINITIONS ############################## 

# Determine radius of walkers from origin 
def radius(walker):
	return math.sqrt((walker[0])**2+(walker[1])**2)

# Determine if walker has hit perimter of cluster; add to the cluster if hit; redefine perimeter
def check_perimeter(condition,walker,perimeter,cluster,cluster_rad):
	for i, per in enumerate(perimeter):						# Check through all perimeter sites
		if walker[0]+105 == per[0] and walker[1]+105 == per[1]:			# Determine if walker coord == perimeter coord
			cluster[per[0]][per[1]]=1					# If so add to cluster
                        perimeter.pop(i)						# Remove the old perimeter point (which is now part of cluster)
			cluster_rad.append(r_walker)					# Store radius of walker from origin
                        
			# Create new perimeter sites based on new cluster element
			anew_site=np.zeros(2)						
                        anew_site[0]=per[0]+1
                        anew_site[1]=per[1]
                        if cluster[per[0]+1][per[1]] != 1:				# Not a perimeter site if cluster point occupies site
                                perimeter.append(anew_site)
                        bnew_site=np.zeros(2)
                        bnew_site[0]=per[0]-1
                        bnew_site[1]=per[1]
                        if cluster[per[0]-1][per[1]] != 1:				# Not a perimeter site if cluster point occupies site
                                perimeter.append(bnew_site)
                        cnew_site=np.zeros(2)
                        cnew_site[0]=per[0]
                        cnew_site[1]=per[1]+1
                        if cluster[per[0]][per[1]+1] != 1:				# Not a perimeter site if cluster point occupies site
                                perimeter.append(cnew_site)
                        dnew_site=np.zeros(2)
                        dnew_site[0]=per[0]
                        dnew_site[1]=per[1]-1
                        if cluster[per[0]][per[1]-1] != 1:				# Not a perimeter site if cluster point occupies site
                                perimeter.append(dnew_site)
                        condition='hit'							# Let main loop know walker has hit cluster, and needs new walker
                        break								# No longer need to cycle through perimeter
	return condition,cluster

# Determine random point on a circle start_radius distance away from origin 
def rand_point_on_circle():
	# Pick a random number to determine if x or y value is picked first
	x_or_y=random.uniform(0,1)
	if x_or_y <=0.5:
		x=round(random.uniform(-100,100),0)						# Generate random x value
		y=round(math.sqrt(start_radius**2.0-x**2.0))					# Generate corresponding y value which satisfies radius=start_radius
		
		# y could be positive or negative, determine sign at random
        	rand=random.uniform(0,1)
		if rand <= 0.5:
			y=-1*y
		elif rand > 0.5:
			y=1*y
	elif x_or_y > 0.5:
                y=round(random.uniform(-100,100),0)                                             # Generate random y value
                x=round(math.sqrt(start_radius**2.0-y**2.0))                                    # Generate corresponding x value which satisfies radius=start_radius

                # x could be positive or negative, determine sign at random
                rand=random.uniform(0,1)
                if rand <= 0.5:
                        x=-1*x
                elif rand > 0.5:
                        x=1*x


	# Create walker with these x and y values
        walker=np.zeros(2)
        walker[0]=x
        walker[1]=y
        return walker

# For part b) 
# Curve fit and function definition:						# mass = C*(radius)^df
def curvefit(radius,C,df):							# where, C is the proportionality constant and 
	return C+(np.log10(radius))*df						# df is the fractal dimension


############################## END OF FUNCTION DEFINITIONS ##############################

# Loop over algorithm 10 times to generate a good data set
while_count=1
#while (while_count <= 3):
while (while_count <= 10):
	# Initialize cluster 2D array; 0=cluster point; 1=cluster point
	#cluster=np.zeros((int(2*(start_radius)+3),int(2*(start_radius)+3)))
	#cluster[102][102]=1
        cluster=np.zeros((int(2*(start_radius)+10),int(2*(start_radius)+10)))
        cluster[105][105]=1

	# Initialize first four perimeter points
	perimeter=[]
	perimeter.append((105,104))
	perimeter.append((105,106))
	perimeter.append((104,105))
	perimeter.append((106,105))

	# Initialize cluster radius variable; tells the radius of a cluster point from origin as it is added
	cluster_rad=[]
	cluster_rad.append(0)


	# Loop over random walkers until the cluster radius, R, is start_radius length 
	R=0
	walker_count=0
	while R <= 100:
        	walker_count+=1
        	print "Walkers generated: ",walker_count
		# Generate random walker on circle
        	walker=rand_point_on_circle()
        	condition=''											# Initialize condition of walker ('hit' or not)
        	# Move walker until it hits cluster or is too far away to be considered a possible hit
		while condition != 'hit':							
			# Determine walker's single step direction 
                	rand_num=random.uniform(0,1)
                	if rand_num <=0.25:
                        	walker[0]+=1				#right
                	elif rand_num > 0.25 and rand_num <= 0.50:	
                        	walker[0]-=1				#left
                	elif rand_num >0.50 and rand_num <= 0.75:
                        	walker[1]+=1				#up
                	elif rand_num >0.75:
                        	walker[1]-=1				#down
        		# Determine radius of walker from origin
			r_walker=radius(walker)
			# If walker is close to the largest brance of the cluster then check to the perimeter sites, otherwise continue (saves time)
			if r_walker <= max(cluster_rad)+5:
				condition,cluster=check_perimeter(condition,walker,perimeter,cluster,cluster_rad)
				if condition == 'hit':
					R=radius(walker)
                        		print "RADIUS=",R
                        		break
			# Check if walker is out of bounds
			if r_walker >= out_of_bounds:
                        	break					# Break loop to generate new walker


	# Plot cluster
	#np.savetxt("cluster_data.txt",cluster)
	plt.figure()
	cmap=matplotlib.colors.ListedColormap(['white','blue'])
	bounds=[-.5,.5,1.5]
	norm = matplotlib.colors.BoundaryNorm(bounds,cmap.N)
	img= plt.imshow(cluster,interpolation='nearest',cmap=cmap,norm=norm)
	plt.xlabel("Horizontal Position [Arbitrary units]")
	plt.ylabel("Vertical Position [Arbitrary units]")
	plt.title("Cluster from DLA Method")
	savefig("DLA_crystal_final_%i.pdf" %(while_count))
	#plt.show()

	# Part b): Fractal dimensions:
	mass_radius=np.arange(5,105,5)					# Array for radius at which mass is calculated
	mass_count=[0]*len(mass_radius)					# Array to count the number of walkers inside the bounds of each radius array element 
	if while_count==1:
		mass_count_avg=[0]*len(mass_radius)			# Setting up the counts for the average mass values
	for i in range(len(mass_radius)):
    		for j in range(len(cluster_rad)):
        		if cluster_rad[j]<=mass_radius[i]:
            			mass_count[i]=mass_count[i]+1
        	mass_count_avg[i]+=mass_count[i]
        	
        # Curve fit:
        mass=curvefit(mass_radius,1,1.5)				# Calculated values of mass from curve fit equation
	log_mass=np.log10(mass_count)					# Taking log to the base 10 for mass
	popt,pcov=curve_fit(curvefit,mass_radius,log_mass)
	print "Constant,Fractal dimension:",popt
	log_radius=np.log10(mass_radius)				# Taking log to the base 10 for radius
    	
    	mass_analytic=curvefit(mass_radius,popt[0],popt[1])		# Analytically calculated mass from the fit parameters obtained to get the "fit curve"
									# already in log form as given by curvefit function
	# Plotting fractal dimension relation:
	plt.figure()
	plt.plot(log_radius,log_mass,'r*',label="Raw data")
	plt.plot(log_radius,mass_analytic,'k-',label="Fit curve")
	plt.legend()
	plt.xlabel("Radius of cluster")
	plt.ylabel("Number of walkers within radius, mass")
	plt.title("Mass distribution of DLA cluster on a log-log plot")
	plt.savefig("Fractal_dimension_final_%i.pdf" %(while_count))
	#plt.show()

	print "mass_count:"
	print mass_count
	print "mass_count_avg"
	print mass_count_avg		
        while_count+=1    
        
mass_count_avg[:]=[i/10 for i in mass_count_avg]			# Getting average mass over 10 clusters
print "Final mass_count_avg"
print mass_count_avg

# Average Curve fit and function definition:
mass_avg=curvefit(mass_radius,1,1.5)
log_mass_avg=np.log10(mass_count_avg)
popt_avg,pcov_avg=curve_fit(curvefit,mass_radius,log_mass_avg)
print "Constant,Avegrage fractal dimension:",popt_avg
log_radius=np.log10(mass_radius)

mass_analytic=curvefit(mass_radius,popt_avg[0],popt_avg[1])

# Plotting fractal dimension relation:
plt.figure()
plt.plot(log_radius,log_mass_avg,'r*',label="Raw data")
plt.plot(log_radius,mass_analytic,'k-',label="Fit curve")
plt.legend()
plt.xlabel("Radius of cluster")
plt.ylabel("Number of walkers within radius, mass")
plt.title("Fractal dimensionality of DLA cluster averaged over 10 clusters (log-log)")
plt.savefig("Fractal_dimension_final_avg.pdf")
#plt.show()

	

