
import numpy as np
import infotheory
import ffann                #Controller
import invpend              #Task 1
import cartpole             #Task 2
import leggedwalker         #Task 3
import matplotlib.pyplot as plt
import sys

dir = str(sys.argv[1])
start = int(sys.argv[2])
finish = int(sys.argv[3])
reps = finish-start

# ANN Params
nI = 3+4+3
nH1 = 5
nH2 = 5
nO = 1+1+3 #output activation needs to account for 3 outputs in leggedwalker
WeightRange = 15.0
BiasRange = 15.0

# Task Params
duration_IP = 10
stepsize_IP = 0.05
duration_CP = 10 #50
stepsize_CP = 0.05
duration_LW = 220 #220.0
stepsize_LW = 0.1
time_IP = np.arange(0.0,duration_IP,stepsize_IP)
time_CP = np.arange(0.0,duration_CP,stepsize_CP)
time_LW = np.arange(0.0,duration_LW,stepsize_LW)

MaxFit = 0.627 #Leggedwalker

# Fitness initialization ranges
#Inverted Pendulum
trials_theta_IP = 10
trials_thetadot_IP = 10
total_trials_IP = trials_theta_IP*trials_thetadot_IP
theta_range_IP = np.linspace(-np.pi, np.pi, num=trials_theta_IP)
thetadot_range_IP = np.linspace(-1.0,1.0, num=trials_thetadot_IP)

#Cartpole
trials_theta_CP = 10
trials_thetadot_CP = 10
trials_x_CP = 2
trials_xdot_CP = 2
total_trials_CP = trials_theta_CP*trials_thetadot_CP*trials_x_CP*trials_xdot_CP
theta_range_CP = np.linspace(-0.05, 0.05, num=trials_theta_CP)
thetadot_range_CP = np.linspace(-0.05, 0.05, num=trials_thetadot_CP)
x_range_CP = np.linspace(-0.05, 0.05, num=trials_x_CP)
xdot_range_CP = np.linspace(-0.05, 0.05, num=trials_xdot_CP)

#Legged walker
trials_theta = 10
theta_range_LW = np.linspace(-np.pi/6, np.pi/6, num=trials_theta)
trials_omega_LW = 10
omega_range_LW = np.linspace(-1.0, 1.0, num=trials_omega_LW)
total_trials_LW = trials_theta * trials_omega_LW

# Fitness function
def analysis(genotype):
    # Common setup
    nn = ffann.ANN(nI,nH1,nH2,nO)
    nn.setParameters(genotype,WeightRange,BiasRange)
    fitness = np.zeros(3)

    # Task 1
    body = invpend.InvPendulum()
    nn_state_ip = np.zeros((total_trials_IP*len(time_IP),nI+nH1+nH2+nO))
    total_steps = len(theta_range_IP) * len(thetadot_range_IP) * len(time_IP)
    fit_IP = np.zeros((len(theta_range_IP),len(thetadot_range_IP)))
    i=0
    k=0
    for theta in theta_range_IP:
        j=0
        for theta_dot in thetadot_range_IP:
            body.theta = theta
            body.theta_dot = theta_dot
            f = 0.0
            for t in time_IP:
                nn.step(np.concatenate((body.state(),np.zeros(4),np.zeros(3)))) #arrays for inputs for each task
                nn_state_ip[k] = nn.states()
                k += 1
                f += body.step(stepsize_IP, np.array([nn.output()[0]]))
            fit_IP[i][j] = ((f/duration_IP)+7.65)/7
            j += 1
        i += 1
    fitness[0] = np.mean(fit_IP)

    # Task 2
    body = cartpole.Cartpole()
    nn_state_cp = np.zeros((total_trials_CP*len(time_CP),nI+nH1+nH2+nO))
    total_steps = len(theta_range_CP) * len(thetadot_range_CP) * len(x_range_CP) * len(xdot_range_CP) * len(time_CP)
    fit_CP = np.zeros((len(theta_range_CP),len(thetadot_range_CP)))
    i = 0
    k = 0
    for theta in theta_range_CP:
        j = 0
        for theta_dot in thetadot_range_CP:
            f_cumulative = 0
            for x in x_range_CP:
                for x_dot in xdot_range_CP:
                    body.theta = theta
                    body.theta_dot = theta_dot
                    body.x = x
                    body.x_dot = x_dot
                    f = 0.0
                    for t in time_CP:
                        nn.step(np.concatenate((np.zeros(3),body.state(),np.zeros(3))))
                        nn_state_cp[k] = nn.states()
                        k += 1
                        f_temp,d = body.step(stepsize_CP, np.array([nn.output()[1]]))
                        f += f_temp
                    f_cumulative += f/duration_CP
            fit_CP[i][j] = f_cumulative/(len(x_range_CP)*len(xdot_range_CP))
            j += 1
        i += 1
    fitness[1] = np.mean(fit_CP)

    #Task 3
    body = leggedwalker.LeggedAgent(0.0,0.0)
    nn_state_lw = np.zeros((total_trials_LW*len(time_LW),nI+nH1+nH2+nO))
    total_steps = len(theta_range_LW) * len(omega_range_LW) * len(time_LW)
    fit_LW = np.zeros((len(theta_range_LW),len(omega_range_LW)))
    i = 0
    k = 0
    for theta in theta_range_LW:
        j = 0
        for omega in omega_range_LW:
            body.reset()
            body.angle = theta
            body.omega = omega
            for t in time_LW:
                nn.step(np.concatenate((np.zeros(3),np.zeros(4),body.state())))
                nn_state_lw[k] = nn.states()
                k += 1
                body.step(stepsize_LW, np.array(nn.output()[2:5]))
            fit_LW[i][j] = (body.cx/duration_LW)/MaxFit
            j += 1
        i += 1
    fitness[2] = np.mean(fit_LW)
    return fitness,fit_IP,fit_CP,fit_LW,nn_state_ip,nn_state_cp,nn_state_lw

def lesions(genotype,actvalues):

    nn = ffann.ANN(nI,nH1,nH2,nO)

    # Task 1
    ip_fit = np.zeros(nH1+nH2)
    body = invpend.InvPendulum()
    nn.setParameters(genotype,WeightRange,BiasRange)
    index = 0
    for layer in [1,2]:
        for neuron in range(nH1):
            if layer == 1:
                n = neuron
            else:
                n = nH1 + neuron
            print("IP:",n)
            maxfit = 0.0
            for act in actvalues[:,0,n]:
                fit = 0.0
                for theta in theta_range_IP:
                    for theta_dot in thetadot_range_IP:
                        body.theta = theta
                        body.theta_dot = theta_dot
                        for t in time_IP:
                            nn.step_lesioned(np.concatenate((body.state(),np.zeros(4),np.zeros(3))),neuron,layer,act)
                            f = body.step(stepsize_IP, np.array([nn.output()[0]]))
                            fit += f
                fit = fit/(duration_IP*total_trials_IP)
                fit = (fit+7.65)/7
                if fit < 0.0:
                    fit = 0.0
                if fit < maxfit:
                    maxfit = fit
            ip_fit[index]=fit
            index += 1

    # Task 2
    cp_fit = np.zeros(nH1+nH2)
    body = cartpole.Cartpole()
    nn.setParameters(genotype,WeightRange,BiasRange)
    index = 0
    for layer in [1,2]:
        for neuron in range(nH1):
            if layer == 1:
                n = neuron
            else:
                n = nH1 + neuron
            print("CP:",n)
            maxfit = 0.0
            for act in actvalues[:,1,n]:
                fit = 0.0
                for theta in theta_range_CP:
                    for theta_dot in thetadot_range_CP:
                        for x in x_range_CP:
                            for x_dot in xdot_range_CP:
                                body.theta = theta
                                body.theta_dot = theta_dot
                                body.x = x
                                body.x_dot = x_dot
                                for t in time_CP:
                                    nn.step_lesioned(np.concatenate((np.zeros(3),body.state(),np.zeros(3))),neuron,layer,act)
                                    f,d = body.step(stepsize_CP, np.array([nn.output()[1]]))
                                    fit += f
                fit = fit/(duration_CP*total_trials_CP)
                if fit < 0.0:
                    fit = 0.0
                if fit < maxfit:
                    maxfit = fit
            cp_fit[index]=fit
            index += 1

    #Task 3
    lw_fit = np.zeros(nH1+nH2)
    body = leggedwalker.LeggedAgent(0.0,0.0)
    nn.setParameters(genotype,WeightRange,BiasRange)
    index = 0
    for layer in [1,2]:
        for neuron in range(nH1):
            if layer == 1:
                n = neuron
            else:
                n = nH1 + neuron
            print("LW:",n)
            maxfit = 0.0
            for act in actvalues[:,2,n]:
                fit = 0.0
                for theta in theta_range_LW:
                    for omega in omega_range_LW:
                        body.reset()
                        body.angle = theta
                        body.omega = omega
                        for t in time_LW:
                            nn.step_lesioned(np.concatenate((np.zeros(3),np.zeros(4),body.state())),neuron,layer,act)
                            body.step(stepsize_LW, np.array(nn.output()[2:5]))
                        fit += body.cx/duration_LW
                fit = (fit/total_trials_LW)/MaxFit
                if fit < 0.0:
                    fit = 0.0
                if fit < maxfit:
                    maxfit = fit
            lw_fit[index]=fit
            index += 1

    return ip_fit,cp_fit,lw_fit

def find_all_lesions(dir,ind):
    max = np.zeros((3,nH1+nH2))
    nn = np.load("./{}/state_IP_{}.npy".format(dir,ind))
    max[0] = np.max(nn[:,nI:nI+nH1+nH2],axis=0)
    nn = np.load("./{}/state_CP_{}.npy".format(dir,ind))
    max[1] = np.max(nn[:,nI:nI+nH1+nH2],axis=0)
    nn = np.load("./{}/state_LW_{}.npy".format(dir,ind))
    max[2] = np.max(nn[:,nI:nI+nH1+nH2],axis=0)

    steps = 10
    actvalues = np.linspace(0.0, max, num=steps)

    bi = np.load("./{}/best_individual_{}.npy".format(dir,ind))
    f = np.load("./{}/perf_{}.npy".format(dir,ind))

    ipp,cpp,lwp = lesions(bi,actvalues)

    ipp = ipp/f[0]
    cpp = cpp/f[1]
    lwp = lwp/f[2]

    np.save(dir+"/lesions_IP_"+str(ind)+".npy",ipp)
    np.save(dir+"/lesions_CP_"+str(ind)+".npy",cpp)
    np.save(dir+"/lesions_LW_"+str(ind)+".npy",lwp)

    # Stats on neurons for Ablations
    Threshold = 0.95
    count = np.zeros(8)
    for (ip_neuron, cp_neuron, lw_neuron) in zip(ipp,cpp,lwp):
        if ip_neuron > Threshold and cp_neuron > Threshold and lw_neuron > Threshold: # no task neurons
            count[0] += 1
        if ip_neuron <= Threshold and cp_neuron > Threshold and lw_neuron > Threshold: # ip task neurons
            count[1] += 1
        if ip_neuron > Threshold and cp_neuron <= Threshold and lw_neuron > Threshold: # cp task neurons
            count[2] += 1
        if ip_neuron > Threshold and cp_neuron > Threshold and lw_neuron <= Threshold: #lw task neurons
            count[3] += 1
        if ip_neuron <= Threshold and cp_neuron <= Threshold and lw_neuron > Threshold: # ip + cp task neurons
            count[4] += 1
        if ip_neuron <= Threshold and cp_neuron > Threshold and lw_neuron <= Threshold: # ip + lw task neurons
            count[5] += 1
        if ip_neuron > Threshold and cp_neuron <= Threshold and lw_neuron <= Threshold: # cp + lw task neurons
            count[6] += 1
        if ip_neuron <=  Threshold and cp_neuron <= Threshold and lw_neuron <= Threshold: # all  task neurons
            count[7] += 1

    np.save(dir+"/stats_"+str(ind)+".npy",count)

    # plt.plot(ipp,'ro')
    # plt.plot(cpp,'go')
    # plt.plot(lwp,'bo')
    # plt.xlabel("Interneuron")
    # plt.ylabel("Performance")
    # plt.title("Lesions")
    # plt.savefig(dir+"/lesions_"+str(ind)+".png")
    # plt.show()

def find_all_var(dir,ind):
    nI = 10
    nH = 10
    v = np.zeros((3,10))
    nn = np.load("./{}/state_IP_{}.npy".format(dir,ind))
    v[0] = np.var(nn[:,nI:nI+nH],axis=0)
    nn = np.load("./{}/state_CP_{}.npy".format(dir,ind))
    v[1] = np.var(nn[:,nI:nI+nH],axis=0)
    nn = np.load("./{}/state_LW_{}.npy".format(dir,ind))
    v[2] = np.var(nn[:,nI:nI+nH],axis=0)
    max = np.max(v,axis=0)
    norm_var = np.zeros((10,3))
    for i in range(10):
        if max[i] > 0.0:
            norm_var[i] = v.T[i]/max[i]
        else:
            norm_var[i] = 0.0
    norm_var = norm_var.T
    np.save("./{}/NormVar_IP_{}.npy".format(dir,ind), norm_var[0])
    np.save("./{}/NormVar_CP_{}.npy".format(dir,ind), norm_var[1])
    np.save("./{}/NormVar_LW_{}.npy".format(dir,ind), norm_var[2])

    # plt.plot(norm_var[0],'ro')
    # plt.plot(norm_var[1],'go')
    # plt.plot(norm_var[2],'bo')
    # plt.xlabel("Interneurons")
    # plt.ylabel("Normalized variance")
    # plt.title("Normalized variance")
    # plt.savefig(dir+"/NormVar_"+str(ind)+".png")
    # plt.show()

def calculate_mi(filename, nbins=50, nreps=3):

    dat = np.load(filename)
    print(dat.shape)
    nI = 10
    nH = 10

    # to iterate through all neurons
    neuron_inds = np.arange(nI,nI+nH)

    # setup for infotheory analysis
    mis = []
    mins = np.min(dat[:,:nI+nH], 0)
    maxs = np.max(dat[:,:nI+nH], 0)
    dims = nI+nH

    # add all data
    it = infotheory.InfoTools(dims, nreps)
    it.set_equal_interval_binning([nbins]*dims, mins, maxs)
    it.add_data(dat[:,:nI+nH])

    # estimate entropy
    var_ids = [0]*nI + [-1]*nH
    ent =  it.entropy(var_ids)

    # estimate mutual information
    for i in range(nI,nI+nH):
        print("\tNeuron # {}".format(i+1))
        var_ids[i] = 1
        mi = it.mutual_info(var_ids)
        var_ids[i] = -1
        mis.append(mi/ent)

    return mis

def find_all_mis(dir,ind):
    mi = np.zeros((3,10))
    mi[0] = calculate_mi("./{}/state_IP_{}.npy".format(dir,ind))
    mi[1] = calculate_mi("./{}/state_CP_{}.npy".format(dir,ind))
    mi[2] = calculate_mi("./{}/state_LW_{}.npy".format(dir,ind))
    max = np.max(mi,axis=0)
    norm_mi = np.zeros((10,3))
    for i in range(10):
        if max[i] > 0.0:
            norm_mi[i] = mi.T[i]/max[i]
        else:
            norm_mi[i] = 0.0
    norm_mi = norm_mi.T
    np.save("./{}/NormMI_IP_{}.npy".format(dir,ind), norm_mi[0])
    np.save("./{}/NormMI_CP_{}.npy".format(dir,ind), norm_mi[1])
    np.save("./{}/NormMI_LW_{}.npy".format(dir,ind), norm_mi[2])

    # plt.plot(mi[0],'ro')
    # plt.plot(mi[1],'go')
    # plt.plot(mi[2],'bo')
    # plt.xlabel("Interneurons")
    # plt.ylabel("Mutual information")
    # plt.title("Mutual Information")
    # plt.savefig(dir+"/NormMI_"+str(ind)+".png")
    # plt.show()

gens = len(np.load(dir+"/average_history_0.npy"))
gs=len(np.load(dir+"/best_individual_0.npy"))
af = np.zeros((reps,gens))
bf = np.zeros((reps,gens))
bi = np.zeros((reps,gs))

index = 0
count = 0
for i in range(start,finish):
    af[index] = np.load(dir+"/average_history_"+str(i)+".npy")
    bf[index] = np.load(dir+"/best_history_"+str(i)+".npy")
    bi[index] = np.load(dir+"/best_individual_"+str(i)+".npy")
    if bf[index][-1]>0.8:
        print(i)
        count += 1
        f,m1,m2,m3,ns1,ns2,ns3=analysis(bi[index])
        print(i,f)
        np.save(dir+"/perf_"+str(i)+".npy",f)

        np.save(dir+"/perfmap_IP_"+str(i)+".npy",m1)
        np.save(dir+"/perfmap_CP_"+str(i)+".npy",m2)
        np.save(dir+"/perfmap_LW_"+str(i)+".npy",m3)

        np.save(dir+"/state_IP_"+str(i)+".npy",ns1)
        np.save(dir+"/state_CP_"+str(i)+".npy",ns2)
        np.save(dir+"/state_LW_"+str(i)+".npy",ns3)

        print(i,bf[index][-1],f)

        # plt.imshow(m1)
        # plt.colorbar()
        # plt.xlabel("Theta")
        # plt.ylabel("ThetaDot")
        # plt.title("Inverted Pendulum")
        # plt.savefig(dir+"/perfmap_IP_"+str(i)+".png")
        # plt.show()
        # plt.imshow(m2)
        # plt.colorbar()
        # plt.xlabel("Theta")
        # plt.ylabel("ThetaDot")
        # plt.title("Cart Pole")
        # plt.savefig(dir+"/perfmap_CP_"+str(i)+".png")
        # plt.show()
        # plt.imshow(m3)
        # plt.colorbar()
        # plt.xlabel("Theta")
        # plt.ylabel("ThetaDot")
        # plt.title("Legged Walker")
        # plt.savefig(dir+"/perfmap_LW_"+str(i)+".png")
        # plt.show()

        # find_all_lesions(dir,i)
        # find_all_var(dir,i)
        # find_all_mis(dir,i)
    index += 1

print("Ensemble count:", count)
print(bf[:,-1])
# plt.plot(af.T,'y')
# plt.plot(bf.T,'b')
# plt.xlabel("Generations")
# plt.ylabel("Fitness")
# plt.title("Evolution")
# plt.savefig(dir+"/evol.png")
# plt.show()
