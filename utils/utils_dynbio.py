from utils.utils_global import *
from scipy.integrate import odeint
from scipy.special import factorial
from scipy.stats import binom, poisson

text_path = 'assets/dynamical_models/text/'


###########################################
# week 1
@function_profiler
def plot_noise(kmrna, gmrna, kpro, gpro, Ngene, Ncell, NRepressor):
    t_final = 1000 # end of the simulation time
    ra=100 # association rate of a protein - fast enough
    kdiss=1  #dissociation constant of a protein
    rd=ra*kdiss #dissociation rate of a protein
    grep=0.05 # repressor degradation rate 
    krep=grep*NRepressor # repressor production rate


################ making arrays to record the data
    nrecord=1000 #number of data points recorded for plotting 
    t = np.linspace(0,t_final,num=nrecord) #This is unoccupye step that I take the record
    mRNA = [[[0 for j in range(nrecord)] for i in range(Ncell)] for ig in range (Ngene)]  # mRNA[0][0][0] is gene 1 mRNA in the cell 1 at time 0, mRNA[1][0][3] is gene 2 mRNA if Ngene=2 in the cell 1 at time 3
    protein =  [[[0 for j in range(nrecord)] for i in range(Ncell)] for ig in range (Ngene)]
    NR =  [[0 for j in range(nrecord)] for i in range(Ncell)] 

    mRNAinit=int(kmrna/(1.+NRepressor/kdiss)/gmrna)
    proinit=int(kpro*mRNAinit/gpro)
    mRNAnow = [[mRNAinit for i in range(Ncell)] for ig in range (Ngene)] 
    proteinnow =  [[proinit for i in range(Ncell)]  for ig in range (Ngene)]
    NRnow = [NRepressor for i in range(Ncell)]   #Free repressors 
    timenow = [0. for i in range(Ncell)]  
    unoccupy = [[1. for i in range(Ncell)] for ig in range (Ngene)]  # unoccupy[1][0] is gene 2 occupation in the cell 1 
    itime=[0 for i in range(Ncell)]  

    bind = [0 for ig in range(Ngene)] 
    unbind = [0 for ig in range(Ngene)] 
    mprod = [0 for ig in range(Ngene)] 
    mdeg = [0 for ig in range(Ngene)] 
    pprod = [0 for ig in range(Ngene)] 
    pdeg = [0 for ig in range(Ngene)] 
            
    while min(timenow) < t_final:
        for i in range(Ncell):
            rsum=krep
            repdeg=grep*NRnow[i]
            rsum+=repdeg
            for ig in range(Ngene):
                bind[ig]=unoccupy[ig][i]*NRnow[i]*ra
                unbind[ig]=(1-unoccupy[ig][i])*rd
                mprod[ig]=kmrna*unoccupy[ig][i]
                mdeg[ig]=mRNAnow[ig][i]*gmrna
                pprod[ig]=mRNAnow[ig][i]*kpro
                pdeg[ig]=proteinnow[ig][i]*gpro
                
                rsum+=mprod[ig]+mdeg[ig]+pprod[ig]+pdeg[ig]+bind[ig]+unbind[ig]
                
                #print(krep,repdeg)
                #print(mprod[ig],mdeg[ig],pprod[ig],pdeg[ig],bind[ig],unbind[ig])
                
            a = np.random.random(1)
            tau=-np.log(a)/rsum
            
            while (itime[i]<nrecord and t[itime[i]]>=timenow[i] and t[itime[i]]<timenow[i]+tau):
                NR[i][itime[i]]=NRnow[i]
                for ig in range(Ngene):
                    mRNA[ig][i][itime[i]]=mRNAnow[ig][i]
                    protein[ig][i][itime[i]]=proteinnow[ig][i]
                    #NR[i][itime[i]]+=1-unoccupy[ig][i]                
                itime[i]=itime[i]+1                   
            
            timenow[i]=timenow[i]+tau
          
            a=np.random.random(1)
            rsumnow=krep
            if a<rsumnow/rsum:
                NRnow[i]+=1
            else:
                rsumnow+=repdeg
                if a<rsumnow/rsum:
                    NRnow[i]-=1
                else:
                    ig=0
                    rsumnow+=mprod[ig]
                    if a<rsumnow/rsum:
                        mRNAnow[ig][i]+=1
                    else:
                        rsumnow+=mdeg[ig]
                        if a<rsumnow/rsum:
                            mRNAnow[ig][i]-=1
                        else:
                            rsumnow+=pprod[ig]
                            if a<rsumnow/rsum:
                                proteinnow[ig][i]+=1
                            else:
                                rsumnow+=pdeg[ig]
                                if a<rsumnow/rsum:
                                    proteinnow[ig][i]-=1
                                else:
                                    rsumnow+=bind[ig]
                                    if a<rsumnow/rsum:
                                        unoccupy[ig][i]=0
                                        NRnow[i]-=1
                                    else:
                                        rsumnow+=unbind[ig]
                                        if a<rsumnow/rsum:
                                            unoccupy[ig][i]=1
                                            NRnow[i]+=1
                                        else:
                                            if Ngene==2:
                                                ig=1
                                                rsumnow+=mprod[ig]
                                                if a<rsumnow/rsum:
                                                    mRNAnow[ig][i]+=1
                                                else:
                                                    rsumnow+=mdeg[ig]
                                                    if a<rsumnow/rsum:
                                                        mRNAnow[ig][i]-=1
                                                    else:
                                                        rsumnow+=pprod[ig]
                                                        if a<rsumnow/rsum:
                                                            proteinnow[ig][i]+=1
                                                        else:
                                                            rsumnow+=pdeg[ig]
                                                            if a<rsumnow/rsum:
                                                                proteinnow[ig][i]-=1
                                                            else:
                                                                rsumnow+=bind[ig]
                                                                if a<rsumnow/rsum:
                                                                    unoccupy[ig][i]=0
                                                                    NRnow[i]-=1
                                                                else:
                                                                    rsumnow+=unbind[ig]
                                                                    if a<rsumnow/rsum:
                                                                        unoccupy[ig][i]=1
                                                                        NRnow[i]+=1



    mRNAmean=np.mean(mRNA, axis=2)
    mRNAvar=np.var(mRNA, axis=2)
    promean=np.mean(protein, axis=2)
    provar=np.var(protein, axis=2)

    #print('mRNA average', np.mean(mRNAmean))
    #print('protein average', np.mean(promean))
    #print('protein variance', np.mean(provar))
    #print('protein standard deviation', np.sqrt(np.mean(provar)))
    #print('protein noise (Total noise)', np.sqrt(np.mean(provar))/np.mean(promean))
            
    mRNA_average = np.mean(mRNAmean)
    pro_average = np.mean(promean)
    pro_variacne = np.mean(provar)
    pro_std = np.sqrt(np.mean(provar))
    pro_noise = np.sqrt(np.mean(provar))/np.mean(promean)
    results = [mRNA_average, pro_average, pro_variacne, pro_std, pro_noise]
            

    if Ngene==2:
        noiseI2=0.
        noiseE2=0.
        for it in range (len(t)):
            for i in range (Ncell):
                noiseI2+=(protein[0][i][it]-protein[1][i][it])**2
                noiseE2+=(protein[0][i][it]*protein[1][i][it])
        
        proallmean=np.mean(promean, axis=1)
        noiseI2=noiseI2/(Ncell*len(t))
        noiseE2=noiseE2/(Ncell*len(t))
        noiseE2=noiseE2-proallmean[0]*proallmean[1]
        
        noiseI2=noiseI2/(2*proallmean[0]*proallmean[1])
        noiseE2=noiseE2/(proallmean[0]*proallmean[1])

        print('Extrinsic noise', np.sqrt(noiseE2))
        print('Intrinsic noise', np.sqrt(noiseI2))
        #print('Total noise', np.sqrt(noiseE2+noiseI2), np.sqrt(np.mean(provar))/np.mean(promean))
        results.append(np.sqrt(noiseE2))
        results.append(np.sqrt(noiseI2))
        

# plot results
    if Ngene==1:
        fig, axs = plt.subplots(3, 1, figsize=(15,15))
        axs[0].plot(t,NR[0][:],'g-',label=r'repressor')
        axs[0].set_title('cell 1, repressor')
        axs[0].set_ylabel('repressor ')
        axs[0].set_xlabel('time')


        ig=0
        axs[1].plot(t,mRNA[ig][0][:],'b-',label=r'mRNA')
        axs[1].set_title('cell 1, gene 1')
        axs[1].set_ylabel('mRNA')
        axs[1].set_xlabel('time')


        axs[2].plot(t,protein[ig][0][:],'r-',label=r'protein')
        axs[2].set_ylabel('protein')
        axs[2].set_xlabel('time')

        plt.show()


    if Ngene==2:
        fig, axs = plt.subplots(3, 2, figsize=(15,15))
        axs[0][0].plot(t,NR[0][:],'g-',label=r'repressor')
        axs[0][0].set_title('cell 1, repressor')
        axs[0][0].set_ylabel('repressor ')
        axs[0][0].set_xlabel('time')


        ig=0
        axs[1][0].plot(t,mRNA[ig][0][:],'b-',label=r'mRNA')
        axs[1][0].set_title('cell 1, gene 1')
        axs[1][0].set_ylabel('mRNA')
        axs[1][0].set_xlabel('time')


        axs[2][0].plot(t,protein[ig][0][:],'r-',label=r'protein')
        axs[2][0].set_ylabel('protein')
        axs[2][0].set_xlabel('time')

        ig=1
        axs[1][1].plot(t,mRNA[ig][0][:],'b-',label=r'mRNA')
        axs[1][1].set_title('cell 1, gene 2')
        axs[1][1].set_ylabel('mRNA')
        axs[1][1].set_xlabel('time')


        axs[2][1].plot(t,protein[ig][0][:],'r-',label=r'protein')
        axs[2][1].set_ylabel('protein')
        axs[2][1].set_xlabel('time')

        plt.show()    

    if Ncell==2: 
        if Ngene==1:
            fig, axs = plt.subplots(3, 1, figsize=(15,15))
            axs[0].plot(t,NR[1][:],'g-',label=r'repressor')
            axs[0].set_title('cell 2, repressor')
            axs[0].set_ylabel('repressor ')
            axs[0].set_xlabel('time')


            ig=0
            axs[1].plot(t,mRNA[ig][1][:],'b-',label=r'mRNA')
            axs[1].set_title('cell 2, gene 1')
            axs[1].set_ylabel('mRNA')
            axs[1].set_xlabel('time')


            axs[2].plot(t,protein[ig][1][:],'r-',label=r'protein')
            axs[2].set_ylabel('protein')
            axs[2].set_xlabel('time')

            plt.show()


        if Ngene==2:
            fig, axs = plt.subplots(3, 2, figsize=(15,15))
            axs[0][0].plot(t,NR[1][:],'g-',label=r'repressor')
            axs[0][0].set_title('cell 2, repressor')
            axs[0][0].set_ylabel('repressor ')
            axs[0][0].set_xlabel('time')


            ig=0
            axs[1][0].plot(t,mRNA[ig][1][:],'b-',label=r'mRNA')
            axs[1][0].set_title('cell 2, gene 1')
            axs[1][0].set_ylabel('mRNA')
            axs[1][0].set_xlabel('time')


            axs[2][0].plot(t,protein[ig][1][:],'r-',label=r'protein')
            axs[2][0].set_ylabel('protein')
            axs[2][0].set_xlabel('time')

            ig=1
            axs[1][1].plot(t,mRNA[ig][1][:],'b-',label=r'mRNA')
            axs[1][1].set_title('cell 2, gene 2')
            axs[1][1].set_ylabel('mRNA')
            axs[1][1].set_xlabel('time')


            axs[2][1].plot(t,protein[ig][1][:],'r-',label=r'protein')
            axs[2][1].set_ylabel('protein')
            axs[2][1].set_xlabel('time')


    return results, fig


# week 2
## plot function for Hill function
@function_profiler
def plot_hill_function(threshold, coeff, activation=True):
    x = np.linspace(0, 2, 1000)

    if activation:
        y = 1.0 / (1.0 + (x/threshold)**coeff)

        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.axhline(0.5, c='gray', ls='--')
        ax.axvline(threshold, c='gray', ls='--')
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 1)
        ax.set_title("Hill function for repression")
        ax.set_xlabel("Concentration of TF $c_\mathrm{TF}$")
        ax.set_ylabel("Value of Hill function")

    else :
        y = (x/threshold)**coeff / (1.0 + (x/threshold)**coeff)

        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.axhline(0.5, c='gray', ls='--')
        ax.axvline(threshold, c='gray', ls='--')
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 1)
        ax.set_title("Hill function for activation")
        ax.set_xlabel("Concentration of TF $c_\mathrm{TF}$")
        ax.set_ylabel("Value of Hill function")
    sns.despine()
    return fig

## Solving ODE
'''
def srna_simulation():
    def model(x, t, k_mRNA, g_mRNA, k_pro, g_pro, k_sRNA, g_sRNA, delta):
        # x[0] is the concentration of mRNA,m, and 
        # x[1] is the concentration of protein, p.
        # x[2] is the concentration of sRNA, s.
        dmdt = k_mRNA - g_mRNA*x[0] - delta*x[0]*x[2]
        dpdt = k_pro*x[0] - g_pro*x[1]
        dsdt = k_sRNA - g_sRNA*x[2] - delta*x[0]*x[2]
        dxdt = [dmdt, dpdt, dsdt]
        return dxdt
        
    # parameters
    k_mRNA  = 1.
    g_mRNA  = 5.
    k_pro   = 50.
    g_pro   = 1.
    k_sRNA   = 0.01
    g_sRNA   = 1.
    delta = 100.

    # simulation length
    t_final=10
    # data point
    t = np.linspace(0, t_final, 100)

    # initial state
    mRNA_initial    = 0.
    protein_initial = 0.
    sRNA_initial = 0.


    # initail state as array
    x_initial = [mRNA_initial, protein_initial, sRNA_initial]

n
# integrate the ordingary differential equation, it returns the array of solution in x_solution
    x_solution = odeint(model, x_initial, t, 
                        args=(k_mRNA, g_mRNA, k_pro, g_pro, k_sRNA, g_sRNA, delta))

n
# plot results
    fig, ax = plt.subplots()
    ax.plot(t, x_solution[:,0], ls = '-',label=r'mRNA')
    ax.plot(t, x_solution[:,1], ls = '--',label=r'protein')
    ax.plot(t, x_solution[:,2], ls = '--',label=r'sRNA')
    ax.xlabel('time')
    ax.ylabel('response')
    ax.legend(frameon=False)
    sns.despine()
    return fig
'''
##################
# week3
@function_profiler
def binomial(N, p, k):
    return factorial(N) * p**k * (1.0-p)**(N-k) / (factorial(N-k)*factorial(k)) 

@function_profiler
def plot_binomial(N1, p1, N2, p2):
    k_max = 50
    k = np.arange(np.max([N1, N2])+1)
    #y1 = binomial(N1, p1, k)
    #y2 = binomial(N2, p2, k)
    y1 = binom.pmf(k, N1, p1)
    y2 = binom.pmf(k, N2, p2)

    fig, ax = plt.subplots()
    ax.plot(k, y1, marker='o', markersize=2, 
            label="$N_1$={}, $p_1$={:.2f}".format(N1, p1))
    ax.plot(k, y2, marker='o', markersize=2, 
            label="$N_2$={}, $p_2$={:.2f}".format(N2, p2))
    ax.set_xlim(0, k_max)
    ax.set_ylim(0, 1)
    ax.set_title("Binomial distribution")
    ax.set_xlabel("$k$")
    ax.set_ylabel("$P_N(k)$")
    ax.legend(frameon=False)
    return fig, ax

@function_profiler
def plot_poisson(m1, m2):
    k_max = 50
    k = np.arange(k_max+1)
    #y = np.exp(-m) * np.power(m, k) / factorial(k) 
    y1 = poisson.pmf(k, m1)
    y2 = poisson.pmf(k, m2)

    fig, ax = plt.subplots()
    ax.plot(k, y1, marker='o', markersize=2, label="$m_1$={}".format(m1))
    ax.plot(k, y2, marker='o', markersize=2, label="$m_2$={}".format(m2))
    ax.set_xlim(0, k_max)
    ax.set_ylim(0, 0.4)
    ax.set_title("Poisson distribution")
    ax.set_xlabel("$k$")
    ax.set_ylabel("$P_p(k)$")
    ax.legend(frameon=False)
    return fig

@function_profiler
def plot_binomial_poisson(N, m):
    p = m/N
    k_max = 100
    k = np.arange(k_max+1)
    y_binomial = binom.pmf(k, N, p)
    y_poisson = poisson.pmf(k, m)

    fig, ax = plt.subplots()
    ax.plot(k, y_binomial, marker='o', markersize=2, 
            label="Binomial, $N$={}, $p$={:.2f}".format(N, p))
    ax.plot(k, y_poisson,  marker='o', markersize=2, 
            label="Poisson, $m=Np$={:.2f}".format(m))
    ax.set_xlim(0, k_max)
    ax.set_ylim(0, 0.4)
    ax.set_title("Binomial vs Poisson distribution")
    ax.set_xlabel("$k$")
    ax.set_ylabel("$P(k)$")
    ax.legend(frameon=False)
    return fig


################
# Week 4
@function_profiler
def plot_michaelis_menten1(lambda_max, K_S):
    x = np.linspace(0, 10.0, 100)
    y = lambda_max * x / (K_S + x)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.axhline(lambda_max, c='k', ls='--')
    ax.hlines(lambda_max*0.5, 0, K_S, colors='k', ls='--')
    ax.vlines(K_S, 0, lambda_max*0.5, colors='k', ls='--')
    ax.text(0.05, lambda_max*1.05, '$\lambda_\mathrm{max}$', )
    ax.text(K_S+0.05, lambda_max*0.5**1.05, '$\lambda_\mathrm{max}/2$', )
    ax.text(K_S+0.05, 0.02, '$K_S$', )
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.set_title('$\lambda = \lambda_\mathrm{max} \\frac{S}{K_S + S}$')
    ax.set_xlabel('$S$ concentration of the limiting nutrient')
    ax.set_ylabel('$\lambda$ growth rate')
    return fig, ax

################
# Week 5
@function_profiler
def plot_solve_regulation(H, gamma_P, positive=True):
    x = np.linspace(0, 4, 100)
    y1 = gamma_P*x
    fig, ax = plt.subplots()
    if positive:
        y0 = x**H/(1+x**H)
        ax.plot(x, y0, label="$y = \\frac{P^H}{1+P^H}$")
    else:
        y0 = 1/(1+x**H)
        ax.plot(x, y0, label="$y = \\frac{1}{1+P^H}$")
    ax.plot(x, y1, label="$y = \Gamma_\mathrm{p} P$")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Protein concentration $P$')
    ax.set_ylabel('y')
    ax.legend(frameon=False)
    sns.despine()
    return fig, ax

