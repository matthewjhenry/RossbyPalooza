# packages to import
import ClimateUtils as clim
import matplotlib.pyplot as plt
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units

########## parameters #################
delta_time = 10*60*60*24 #10 days in s
rho_w = 1000 #kg/m3
c_w = 4000 #J/(kg.K)
h_w = 10 #m
# ISR = 400 #W/m2
albedo_surf = 0.2
psurf = 1000 #hPa
delta_RH_s = 0.001
n_press = 31 #number of pressure levels
# days_max = 10 # number of days for integration
grav = 10. # m/s2
mix_ratio_param = 0.622 # no units
r_droplet = 1e-5 # m
#######################################

######### functions ###################
# function that returns cloud liquid water content
def liq_water_cont(p,svp,lcl_p_ind,p_top_ind):
	# regular p grid : dp is the same for all p
    dp = p[0]-p[1]
    W = np.zeros_like(p)
	# Water liq content is an integration
    W_sum = 0
    for i in range(len(p)-1,-1,-1):
			# above cloud
            if i>p_top_ind:
                W[i]=0
			#in cloud
            if i<=p_top_ind and i>=lcl_p_ind:
                W_sum += (mix_ratio_param/grav)*(svp[i]/p[i])*dp
                W[i] = W_sum
			#under cloud
            if i<lcl_p_ind:
                W[i]=0
    return W

# function that returns the column's albedo
def albedo_column(p,lcl_p_ind,W, flag):
    tau_SW = (3./2.)*(W[lcl_p_ind] / (rho_w * r_droplet))
    albedo_cloud = (0.15*tau_SW)/(1+0.15*tau_SW)
    if flag == True:
		# if tau_SW very big, return max albedo
        if tau_SW > 1e5:
            return 0.85
        else:
            albedo_col = albedo_cloud + np.power(1-albedo_cloud,2) * albedo_surf
            # alebdo of column can't be 1!
            if albedo_col>0.85:
                return 0.85
            else:
                return albedo_col
    else:
        return albedo_surf

# function that returns the index of the tau=1 level
def tau_1_level(W,GH_flag):
    # if tau << 1, OLR = sigma T_s**4
    tau_LW = W*1e2
	# use GH_flag = False to turn off LW feedback
    if GH_flag == True:
		# is total tau_LW smaller than 1
		# OLR = sigma Ts^4
		# return index of surface, i.e. 0
        if np.all(tau_LW<1.):
            return 0
		# else return index of tau=1
        else:
            return np.nanargmin(np.abs(tau_LW-1.))
    else:
        return 0
######################################
plot_T = []
plot_p_LCL = []
plot_T_LCL = []
plot_p_top = []
plot_albedo = []
plot_tau_1 = []

for Ts in [100* units('K'),500* units('K')]:
    #sweep ISR values
    ISR_array = [300,400,500,600,800,1000]
    # ISR_array = [300]

    # store values in these arrays
    T_array = []
    p_LCL_array = []
    T_LCL_array = []
    p_top_array = []
    albedo_array = []
    tau_1_array = []

    # set to True to turn on SW and LW fb respectively
    albedo_flag = False
    GH_flag = False

    for ISR in ISR_array:
	    ######## Variables ##########
	    RH_s = 0.5
	    ps = psurf * units('hPa')
	    p = np.linspace(psurf,0, n_press) * units('hPa')
	    #############################

	    T = []
	    T.append(500.)

	    N=100
	    # for days in range(days_max):
	    while np.abs(N)>1:
		# use MetPy package to calculate LCL level and T profile
		lcl_p, lcl_T = mpcalc.lcl(ps,Ts,mpcalc.dewpoint_rh(Ts,RH_s))
		T_parcel = mpcalc.parcel_profile(p,Ts,mpcalc.dewpoint_rh(Ts,RH_s))
		svp = mpcalc.saturation_vapor_pressure(T_parcel)
		p = p / units('hPa')
		Ts = Ts / units('K')
		T_parcel = T_parcel / units('K')
		svp = svp / units('hPa')
		
		p_top = p[np.nanargmin(np.abs(svp-0.5))]
		p_top_ind = np.nanargmin(np.abs(svp-0.5))
		lcl_p_ind = np.nanargmin(np.abs(p-lcl_p / units('hPa')))
		
		# use function to calculate liquid Water content as a function of z
		W = liq_water_cont(p,svp,lcl_p_ind,p_top_ind)
		# calculate net radiation
		ASR = ISR * (1-albedo_column(p,lcl_p_ind,W,albedo_flag))
		OLR = clim.sigma * T_parcel[tau_1_level(W,GH_flag)]**4
		N = ASR - OLR
		# calculate change in surface temperature
		dTs = N/(c_w*rho_w*h_w)*delta_time
		Ts += dTs
		T.append(Ts)
		Ts = Ts * units('K')
		p = p * units('hPa')
		RH_s += delta_RH_s
		
	    # store results
	    T_array.append(Ts / units('K'))
	    p_LCL_array.append(lcl_p / units('hPa'))
	    T_LCL_array.append(lcl_T / units('K'))
	    p_top_array.append(p_top)
	    albedo_array.append(albedo_column(p,lcl_p_ind,W, albedo_flag))
	    tau_1_array.append(p[tau_1_level(W,GH_flag)])
    plot_T.append(T_array)
    plot_p_LCL.append(p_LCL_array)
    plot_T_LCL.append(T_LCL_array)
    plot_p_top.append(p_top_array)
    plot_albedo.append(albedo_array)
    plot_tau_1.append(tau_1_array)


# plot hysteresis curve
plt.plot(ISR_array, plot_T[0],'b',label='T(0)=100K')
plt.plot(ISR_array, plot_T[1],'r',label='T(0)=500K')
plt.xlabel('Incident Solar Radiation (W/m2)')
plt.ylabel('K')
plt.title('Equilibrium Surface Temperature for different initial surface temperature')
plt.legend(loc='best')
plt.grid()
plt.show()
