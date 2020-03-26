
import multiprocessing
import gc, os
import psutil
import time
import numpy as np

def bootstrapSamp(counts_runA, cov_tot):
    np.random.seed()
    pb_runA = counts_runA/np.sum(counts_runA)
    _q = np.random.choice(range(0,len(counts_runA)),size=cov_tot,replace=True,p=pb_runA.astype(np.float))
    sim = np.zeros(len(pb_runA))
    for idx in _q:
        sim[idx] += 1
    return sim

def compute_FFC(base_run, pcr_run, include_nan=False):
    a = base_run/np.sum(base_run)
    b = pcr_run/np.sum(pcr_run)
    old_settings = np.seterr(divide='ignore',invalid='ignore')
    ampratio = b/a
    if include_nan == False:
        ffc = ampratio[np.isfinite(ampratio)]
    else:
        ffc = ampratio
    _=np.seterr(**old_settings)  # reset to default
    return ffc

## PCR sim
def binormSim(n, pamp, cyc=10):
    # simulate the number of molecules after "c" cycle
    # n: number of template molecules
    # p: probability of success PCR
    for i in range(cyc):
        n += np.random.binomial(n, pamp, 1)
    return n.item(0)

def sim_lowCopy(x):
    nseqs = 7373
    pamp = 0.95
    SEQCov = 200*nseqs
    mu = 100000000
    CV = 0.32
    s_normal = np.random.normal(mu, mu*CV, nseqs)
    syn_pool = np.array([round(c) if c>0 else 1 for c in s_normal])
    ffcSTDs_sim, CVs_sim = [], []
    avg_copyNs = [113, 65, 32, 16, 8]
    for copyN in avg_copyNs:
        prePCR_counts = bootstrapSamp(syn_pool, copyN*nseqs)
        simPCR_counts1 = np.array([binormSim(int(count), pamp, 18) for count in prePCR_counts]).astype(float)
        bsSEQ_counts = bootstrapSamp(simPCR_counts1, SEQCov)
        FFC = compute_FFC(syn_pool, bsSEQ_counts)
        ffcSTDs_sim.append(np.std(FFC))
        CVs_sim.append(np.std(bsSEQ_counts)/np.mean(bsSEQ_counts))
    return (CVs_sim)


p = multiprocessing.Pool(40)
try:
    time_start = time.time()
    all_sim = p.map(sim_lowCopy, range(100))
    time_end = time.time()
    time_hrs = (time_end - time_start)/3600
    print ("--- %s hrs ---" % time_hrs)
finally:
    p.close()
    p.join()
    np.save("./sim/sim_lowCopy_190204_100times_CVs.npy", all_sim)
    print "sim_lowCopy done!"
