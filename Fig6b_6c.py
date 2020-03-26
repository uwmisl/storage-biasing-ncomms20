%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pysam
import numpy as np
import seaborn as sns
import pandas as pd
import seaborn as sns

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
def binormSim(n, p, cyc=10):
    # simulate the number of molecules after "c" cycle
    # n: number of template molecules
    # p: probability of success PCR
    for i in range(cyc):
        n += np.random.binomial(n, p, 1)
    return n.item(0)



# simulate CV of synthesized oligos vs sequencing coverage
nseqs = pow(10,4)
p = 0.95
dropout_list = []
store_copyN = 100
for SEQCov_mean in [5,10,30,50,100,1000]:
  for CV in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
      sigma = store_copyN*CV
      s_normal = np.random.normal(store_copyN, sigma, nseqs)
      store_pool = np.array([round(c) if c>=0 else 0 for c in s_normal])
      simPCR_counts1 = np.array([binormSim(int(count), p, 20) for count in store_pool]).astype(float)
      bsSEQ_counts = bootstrapSamp(simPCR_counts1, SEQCov_mean*nseqs)
      missing_counts = np.count_nonzero(bsSEQ_counts==0)
      dropoutRate = 100.0*missing_counts/nseqs
      dropout_list.append((SEQCov_mean, CV, dropoutRate))

df_dpRate = pd.DataFrame(np.array(dropout_list), columns = ["Sequencing reads", "Oligo pool CV", "dropoutRate"])
df_dpRate["dropoutRate"] = df_dpRate["dropoutRate"].astype(float)
df_dpRate["dropoutRate"] = df_dpRate["dropoutRate"].apply(lambda x: round(x,1))
df_dpRate["Sequencing reads"] = df_dpRate["Sequencing reads"].astype(int)
df_dpRate["Oligo pool CV"] = df_dpRate["Oligo pool CV"].astype(float)

df_dpRate_T = df_dpRate.pivot("Sequencing reads", "Oligo pool CV","dropoutRate")
sns.set_context("notebook", font_scale=2.0, rc={"lines.linewidth": 2.5})
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(df_dpRate_T, annot=True, linewidths=.5, ax=ax)
plt.savefig("./figs/Fig_dropout_rate_copyVSseq_copyN=100.pdf", bbox_inches="tight", dpi=300, fmt="pdf")







# simulate storeage copy number vs sequencing coverage
nseqs = pow(10,4)
CV = 0.4
p = 0.95
dropout_list = []
for store_copyN in [5, 10, 30, 50, 100, 1000]:
  for SEQCov_mean in [5, 10, 30, 50, 100, 1000]:
      sigma = store_copyN*CV
      s_normal = np.random.normal(store_copyN, sigma, nseqs)
      store_pool = np.array([round(c) if c>=0 else 0 for c in s_normal])
      simPCR_counts1 = np.array([binormSim(int(count), p, 20) for count in store_pool]).astype(float)
      bsSEQ_counts = bootstrapSamp(simPCR_counts1, SEQCov_mean*nseqs)
      missing_counts = np.count_nonzero(bsSEQ_counts==0)
      dropoutRate = 100.0*missing_counts/nseqs
      dropout_list.append((store_copyN, SEQCov_mean, dropoutRate))

df_dpRate = pd.DataFrame(np.array(dropout_list), columns = ["Copy # per seq", "Sequencing reads", "dropoutRate"])
df_dpRate["dropoutRate"] = df_dpRate["dropoutRate"].astype(float)
df_dpRate["dropoutRate"] = df_dpRate["dropoutRate"].apply(lambda x: round(x,1))
df_dpRate["Copy # per seq"] = df_dpRate["Copy # per seq"].astype(int)
df_dpRate["Sequencing reads"] = df_dpRate["Sequencing reads"].astype(int)
df_dpRate_T = df_dpRate.pivot("Sequencing reads", "Copy # per seq", "dropoutRate")
sns.set_context("notebook", font_scale=2.0, rc={"lines.linewidth": 2.5})
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(df_dpRate_T, annot=True, linewidths=.5, ax=ax)
