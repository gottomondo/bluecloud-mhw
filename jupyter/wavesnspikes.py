### Identify Marine Heatspikes and Heatwaves ###
# Based on code from https://github.com/ecjoliver/marineHeatWaves #

import scipy.ndimage as ndimage
import numpy as np

def wns(sst,pc90): # Input: time series of SST and 90th percentile
	MHS=sst>pc90 # Define spike 

	# Filter out MHS with short duration
	events,n_events=ndimage.label(MHS)
	start=[]
	end=[]
	for ev in range(1,n_events+1):
		event_duration = (events == ev).sum()
		if event_duration < 5: 
			continue

		init=int(np.argwhere(events==ev)[0])
		fin=int(np.argwhere(events==ev)[-1])
		start.append(init)
		end.append(fin)

	# Join MHWs which have short gaps between #
	maxgap=2
	gaps=np.array(start[1:])-np.array(end[0:-1])-1
	if len(gaps)>0:
		while gaps.min()<=maxgap:
			ev=np.where(gaps<=maxgap)[0][0]
			end[ev]=end[ev+1]
			del start[ev+1]
			del end[ev+1]
			gaps=np.array(start[1:])-np.array(end[0:-1])-1
			if len(gaps)==0:
				break

	# Save MHW array
	n_events=len(start)
	MHW=np.zeros((MHS.size))
	for ev in range(n_events):
		event_duration=end[ev]-start[ev]+1
		init=start[ev]
		for i in range(event_duration):
			MHW[init+i]=1

	return MHS, MHW
