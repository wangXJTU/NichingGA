from init_sys import init_sys
from fitness import fitness
from evolution import GA, DE
from swarm import PSO, ABC, WOA, GWO, HHO
from niching import nichingswarm, niching
from single_point import SA
from time import time
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import csv


'''
solution: num_srv dimensionality, each is 0~num_bs-1, 
            representing the base/edge station deploying server
'''

MAX_ITE = 100

num_user = 1000
num_bs = 1000
num_srv = 600
load = 0.5

max_srv = 1000  # max service rate of servers
max_req = max_srv * load  # max request rate of BS
cloud_delay = 0.05
result_file = "opt-results-" + str(num_bs / 1000) + "kBS-" + str(num_srv) + "ES-" + str(load) + "load" + ".csv"
converge_file = "converge.csv"

rfile = open(result_file, "a")
confile = open(converge_file, "w", newline='')
csv_writer = csv.writer(confile)
print("Method", "Delay (s)", "Time (min)", sep=',', file=rfile)
labels = []
History = []
#  service rates of servers; request rates of BSs
init = init_sys(num_bs, num_srv, max_req, max_srv)
srv_rate, req_rate = init.gen_sys()
perf = fitness(srv_rate, req_rate, cloud_delay)

ga = GA(srv_rate, req_rate, cloud_delay, dim=num_srv, high=num_bs, ite=MAX_ITE)
start_time = time()
delay = ga.ga()
end_time = time()
print("GA", delay)
print("GA", delay, (end_time - start_time) / 60, sep=',', file=rfile)
labels.append("GA")
History.append(ga.history_fit)

de = DE(srv_rate, req_rate, cloud_delay, dim=num_srv, high=num_bs, ite=MAX_ITE)
start_time = time()
delay = de.de()
end_time = time()
print("DE", delay)
print("DE", delay, (end_time - start_time) / 60, sep=',', file=rfile)
labels.append("DE")
History.append(de.history_fit)

sa = SA(srv_rate, req_rate, cloud_delay, dim=num_srv, high=num_bs)
start_time = time()
delay = sa.sa()
end_time = time()
print("SA", delay)
print("SA", delay, (end_time - start_time) / 60, sep=',', file=rfile)
labels.append("SA")
History.append(sa.history_fit)

# C. Pandey, V. Tiwari, S. Pattanaik and D. Sinha Roy,
# "A Strategic Metaheuristic Edge Server Placement Scheme for Energy Saving in Smart City,"
# 2023 International Conference on Artificial Intelligence and Smart Communication (AISC),
# Greater Noida, India, 2023, pp. 288-292, doi: 10.1109/AISC56616.2023.10084941.
pso = PSO(srv_rate, req_rate, cloud_delay, dim=num_srv, high=num_bs, ite=MAX_ITE)
start_time = time()
delay = pso.pso()
end_time = time()
print("PSO", delay)
print("PSO", delay, (end_time - start_time) / 60, sep=',', file=rfile)
labels.append("PSO")
History.append(pso.history_fit)

# Bing Zhou, Bei Lu and Zhigang Zhang,
# “Placement of Edge Servers in Mobile Cloud Computing using Artificial Bee Colony Algorithm”
# International Journal of Advanced Computer Science and Applications(IJACSA), 14(2), 2023.
# http://dx.doi.org/10.14569/IJACSA.2023.0140273
abc = ABC(srv_rate, req_rate, cloud_delay, dim=num_srv, high=num_bs, ite=MAX_ITE)
start_time = time()
delay = abc.abc()
end_time = time()
print("ABC", delay)
print("ABC", delay, (end_time - start_time) / 60, sep=',', file=rfile)
labels.append("ABC")
History.append(abc.history_fit)

#  Moorthy, R.S., Arikumar, K.S., Prathiba, B.S.B. (2023).
#  An Improved Whale Optimization Algorithm for Optimal Placement of Edge Server.
#  In: the 4th International Conference on Advances in Distributed Computing and Machine Learning (ICADCML 2023),
#  Rourkela, India, pp. 89-100. https://doi.org/10.1007/978-981-99-1203-2_8
woa = WOA(srv_rate, req_rate, cloud_delay, dim=num_srv, high=num_bs, ite=MAX_ITE)
start_time = time()
delay = woa.woa_dim()
end_time = time()
print("WOA", delay)
print("WOA", delay, (end_time - start_time) / 60, sep=',', file=rfile)
labels.append("WOA")
History.append(woa.history_fit)

hho = HHO(srv_rate, req_rate, cloud_delay, dim=num_srv, high=num_bs, ite=MAX_ITE)
start_time = time()
delay = hho.hho()
end_time = time()
print("HHO", delay)
print("HHO", delay, (end_time - start_time) / 60, sep=',', file=rfile)
labels.append("HHO")
History.append(hho.history_fit)

gwo = GWO(srv_rate, req_rate, cloud_delay, dim=num_srv, high=num_bs, ite=MAX_ITE)
start_time = time()
delay = gwo.gwo()
end_time = time()
print("GWO", delay)
print("GWO", delay, (end_time - start_time) / 60, sep=',', file=rfile)
labels.append("GWO")
History.append(gwo.history_fit)

pso = PSO(srv_rate, req_rate, cloud_delay, dim=num_srv, high=num_bs, ite=MAX_ITE)
start_time = time()
delay = pso.psom()
end_time = time()
print("PSOM", delay)
print("PSOM", delay, (end_time - start_time) / 60, sep=',', file=rfile)
labels.append("PSOM")
History.append(pso.history_fit)

pso = PSO(srv_rate, req_rate, cloud_delay, dim=num_srv, high=num_bs, ite=MAX_ITE)
start_time = time()
delay = pso.pso_sa()
end_time = time()
print("PSOSA", delay)
print("PSOSA", delay, (end_time - start_time) / 60, sep=',', file=rfile)
labels.append("PSOSA")
History.append(pso.history_fit)

#  ToSN 2023 – 华南理工
#  Xinglin Zhang, Jinyi Zhang, Chaoqun Peng, Xiumin Wang,
#  Multimodal Optimization of Edge Server Placement Considering System Response Time,
#  ACM Transactions on Sensor Networks, 2023, 19(1): 13, 20 pages,
#  https://doi.org/10.1145/3534649
niche = nichingswarm(srv_rate, req_rate, cloud_delay, dim=num_srv, high=num_bs, ite=MAX_ITE)
start_time = time()
delay = niche.nPSO()
end_time = time()
print("nPSO", delay)
print("nPSO", delay, (end_time - start_time) / 60, sep=',', file=rfile)
labels.append("nPSO")
History.append(niche.history_fit)

niche = niching(srv_rate, req_rate, cloud_delay, dim=num_srv, high=num_bs, ite=MAX_ITE)
start_time = time()
delay = niche.GA()
end_time = time()
print("NichingGA", delay)
print("NichingGA", delay, (end_time - start_time) / 60, sep=',', file=rfile)
labels.append("NichingGA")
History.append(niche.history_fit)

rfile.close()
csv_writer.writerow(labels)
csv_writer.writerows(History)

confile.close()
'''
params = {
    'axes.labelsize': '35',  #轴上字
    'xtick.labelsize': '27',  #轴图例
    'ytick.labelsize': '27',  #轴图例
    'lines.linewidth': 0.1,  #线宽
    'legend.fontsize': '27',  #图例大小
    'figure.figsize': '12, 9'  # set figure size,长12，宽9
}
pylab.rcParams.update(params)  #set figure parameter

line_styles = ['ro-', 'b^-', 'gs-', 'yv-', 'ch-', 'mD-', 'ro--', 'b^--', 'gs--', 'yv--', 'ch--',
               'mD--']  #set line style
for i in range(len(labels)):
    xx = [i for i in range(len(History[i]))]
    plt.plot(xx, History[i], line_styles[i], label=labels[i], markersize=20)
fig1 = plt.figure(1)
axes = plt.subplot(111)
#axes.set_xticks([i * 10 for i in range(11)])
axes.set_yticks([i * 10 for i in range(5)])
plt.ylabel('Fitness')   # set ystick label
plt.xlabel('Iterative time')

plt.savefig('plot_test1.png',dpi=1000,bbox_inches='tight')
# plt.show()
'''