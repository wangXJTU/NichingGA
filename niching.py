from fitness import fitness
import numpy as np
from sklearn.cluster import KMeans


#  Jaccard distance between two individuals with dim dimensions
def Jaccard(ind1, ind2, dim):
    inter_n = np.sum(ind1 == ind2)
    return inter_n / (dim * 2 - inter_n)


#  calculate similarities between any two individuals of pop
def cal_sim_mat(pop, num_pop, dim, sim_func=Jaccard):
    sim_mat = np.ones(shape=(num_pop, num_pop))
    for i in range(num_pop):
        for j in range(i):
            sim_mat[j][i] = sim_mat[i][j] = sim_func(pop[i], pop[j], dim)
    return sim_mat


def crossing(code1, code2, dim):
    cross_points = np.random.randint(2, size=dim, dtype=bool)
    off1 = code1 * cross_points + code2 * ~cross_points
    off2 = code1 * ~cross_points + code2 * cross_points
    return off1, off2


def mutating(code, dim, low, high):
    points = np.random.randint(2, size=dim, dtype=bool)
    mutated = np.random.randint(low, high, size=dim)
    off = code * points + mutated * ~points
    return off


class nichingswarm(fitness):
    def __init__(self, srv_rate, req_rate, cloud_delay=0.1,
                 num_pop=100, dim=20, w_low=0.4, w_high=1.2, a1=2.0, a2=2.0, ite=100, low=0, high=1,
                 sim_thres=0.5, sim_func=Jaccard):
        super(nichingswarm, self).__init__(srv_rate, req_rate, cloud_delay)
        self.num_pop = num_pop
        self.dim = dim
        # linearly decreasing weight
        self.w_low = w_low
        self.w_high = w_high
        self.a1 = a1
        self.a2 = a2
        self.low = low
        self.high = high - 1
        # 0 ~ high-1
        self.pop = np.random.randint(low, high, size=(num_pop, dim))
        self.vel = np.random.rand(num_pop, dim) * high / 2
        # fitness, 1 / average delay, greater is better
        self.fits = np.zeros(num_pop)
        self.ite = ite
        self.pb = None
        self.pb_fit = None
        #  niche best
        self.nb = None
        self.nb_fit = None
        self.gb = None
        self.gb_fit = None
        self.group = None
        self.num_group = 0
        # matrix
        self.sim_mat = None
        self.sim_thres = sim_thres
        self.sim_func = sim_func
        self.history_fit = None

    '''
    Xinglin Zhang, Jinyi Zhang, Chaoqun Peng, and Xiumin Wang. 2022. 
    Multimodal Optimization of Edge Server Placement Considering System Response Time. 
    ACM Transactions on Sensor Networks, Vol. 19, No. 1, Article 13 (December 2022), 20 pages.
    https://doi.org/10.1145/3534649
    '''

    #  requiring self.sim_mat with similarities between all individuals
    #  providing self.group and self.num_group
    def grouping(self):
        self.group = np.zeros(shape=(self.num_pop, self.num_pop), dtype=bool)
        self.num_group = 0
        for i in range(self.num_pop):
            # largest average similarity of the particle to group, and the group index
            largest_sim = 0
            sim_g = 0
            for g in range(self.num_group):
                #  average similarity of ith particle to gth group
                avg_sim = np.average(self.sim_mat[i][self.group[g]])
                if largest_sim < avg_sim:
                    largest_sim = avg_sim
                    sim_g = g
            #  add ith to sim_gth group
            if largest_sim > self.sim_thres:
                self.group[sim_g][i] = True
            #  add ith to a new group
            else:
                self.group[self.num_group][i] = True
                self.num_group = self.num_group + 1

    def nPSO(self, grouping=None):
        if grouping is None:
            grouping = self.grouping
        self.pop = np.random.randint(self.low, self.high, size=(self.num_pop, self.dim))
        self.vel = np.random.rand(self.num_pop, self.dim) * self.high / 2
        for i in range(self.num_pop):
            self.fits[i] = 1 / self.fn(self.pop[i])
        self.pb = np.copy(self.pop)
        self.pb_fit = np.copy(self.fits)
        best_idx = np.argmax(self.fits)
        self.gb = np.copy(self.pop[best_idx])
        self.gb_fit = self.fits[best_idx]
        self.history_fit = []
        self.history_fit.append(self.gb_fit)

        #
        w_step = (self.w_high - self.w_low) / self.ite
        weight = self.w_high
        for ite in range(self.ite):
            weight = weight - w_step

            r1 = np.random.rand(self.num_pop, self.dim)
            r2 = np.random.rand(self.num_pop, self.dim)
            self.sim_mat = cal_sim_mat(self.pop, self.num_pop, self.dim, self.sim_func)
            grouping()
            self.nb = np.zeros(shape=(self.num_pop, self.dim))
            for g in range(self.num_group):
                # find the best particle in ith group (niche best)
                ng = np.argmax(self.group[g] * self.pb_fit)
                self.nb[self.group[g]] = np.copy(self.pb[ng])

            self.vel = (self.vel * weight + self.a1 * r1 * (self.pb - self.pop)
                        + self.a2 * r2 * (self.nb - self.pop))

            self.pop = (self.pop + self.vel.astype(np.intc)) % (self.high - self.low) + self.low

            for i in range(self.num_pop):
                self.fits[i] = 1 / self.fn(self.pop[i])
                if self.pb_fit[i] < self.fits[i]:
                    self.pb[i] = np.copy(self.pop[i])
                    self.pb_fit[i] = self.fits[i]
            best_idx = np.argmax(self.pb_fit)
            self.gb = np.copy(self.pb[best_idx])
            self.gb_fit = self.pb_fit[best_idx]
            self.history_fit.append(self.gb_fit)
        return 1 / self.gb_fit




class niching(fitness):
    def __init__(self, srv_rate, req_rate, cloud_delay=0.1,
                 size_pop=100, dim=20, ite=100, low=0, high=2,
                 n_niche=5, ratio=0.1):
        super(niching, self).__init__(srv_rate, req_rate, cloud_delay)
        self.dim = dim
        self.low = low
        self.high = high
        self.ite = ite
        self.n_niche = n_niche
        #  n_niche population
        self.niche = None
        self.fits = None
        self.size_pop = int(size_pop / n_niche)
        # consisting of Top 10% individuals (positions) of every niches
        self.ratio = ratio
        self.elites = None
        self.n_elite = int(self.ratio * self.size_pop) * self.n_niche
        self.fit_elite = None
        self.best = None
        self.best_fit = 0
        self.history_fit = None

    def sort(self, pop, fits, size):
        for i in range(size):
            tmp_fit = fits[i]
            idx = i
            for j in range(i + 1, size):
                if tmp_fit < fits[j]:
                    tmp_fit = fits[j]
                    idx = j
            if idx != i:
                pop[i], pop[idx] = pop[idx], pop[i]
                fits[i], fits[idx] = fits[idx], fits[i]

    def set_elites(self):
        n_elite_each_pop = int(self.ratio * self.size_pop)
        self.elites = np.zeros(shape=(self.n_elite, self.dim), dtype=int)
        self.fit_elite = np.zeros(shape=self.n_elite)
        for i in range(self.n_niche):
            self.sort(self.niche[i], self.fits[i], self.size_pop)
            self.elites[i * n_elite_each_pop:(i + 1) * n_elite_each_pop] = np.copy(self.niche[i][0:n_elite_each_pop])
            self.fit_elite[i * n_elite_each_pop:(i + 1) * n_elite_each_pop] = np.copy(self.fits[i][0:n_elite_each_pop])

    def init_pop(self):
        self.niche = np.random.randint(self.low, self.high, size=(self.n_niche, self.size_pop, self.dim))
        self.fits = np.zeros(shape=(self.n_niche, self.size_pop))
        for i in range(self.n_niche):
            for j in range(self.size_pop):
                self.fits[i][j] = 1 / self.fn(self.niche[i][j])
        self.set_elites()

        best_idx = np.argmax(self.fit_elite)
        self.best_fit = self.fit_elite[best_idx]
        self.best = np.copy(self.elites[best_idx])
        self.history_fit = []
        self.history_fit.append(self.best_fit)

    def uniform_cross(self, code1, code2):
        cross_points = np.random.randint(2, size=self.dim, dtype=bool)
        off1 = code1 * cross_points + code2 * ~cross_points
        off2 = code1 * ~cross_points + code2 * cross_points
        return off1, off2

    def uniform_mutation(self, code):
        points = np.random.randint(2, size=self.dim, dtype=bool)
        mutated = np.random.randint(self.low, self.high, size=self.dim)
        off = code * points + mutated * ~points
        return off

    #  roulette wheel selection
    def roulette(self, fits):
        sum_fn = np.sum(fits)
        rv = np.random.uniform() * sum_fn
        cal_fn = fits[0]
        i = 0
        while rv > cal_fn:
            i = i + 1
            cal_fn = cal_fn + fits[i]
        return i - 1

    def GA(self):
        self.init_pop()
        prob_mu = 0.1
        prob_cross = 0.8
        num_offs = 7
        for ite in range(self.ite):
            for i in range(self.n_niche):
                for j in range(self.size_pop):
                    offs = np.zeros(shape=(num_offs, self.dim), dtype=int)
                    off_fits = np.zeros(num_offs)
                    flag = False
                    # crossing with the same niche, another niche and the elites
                    if np.random.rand() < prob_cross:
                        while True:
                            an_idx = np.random.randint(self.size_pop)
                            if an_idx != j:
                                break
                        offs[0], offs[1] = self.uniform_cross(self.niche[i][j], self.niche[i][an_idx])
                        off_fits[0] = 1 / self.fn(offs[0])
                        off_fits[1] = 1 / self.fn(offs[1])
                        flag = True
                    if np.random.rand() < prob_cross:
                        while True:
                            an_niche = np.random.randint(self.n_niche)
                            if an_niche != i:
                                break
                        an_idx = np.random.randint(self.size_pop)
                        offs[2], offs[3] = self.uniform_cross(self.niche[i][j], self.niche[an_niche][an_idx])
                        off_fits[2] = 1 / self.fn(offs[2])
                        off_fits[3] = 1 / self.fn(offs[3])
                        flag = True

                    if np.random.rand() < prob_cross:
                        an_idx = np.random.randint(self.n_elite)
                        offs[4], offs[5] = self.uniform_cross(self.niche[i][j], self.elites[an_idx])
                        off_fits[4] = 1 / self.fn(offs[4])
                        off_fits[5] = 1 / self.fn(offs[5])
                        flag = True
                    # mutating
                    if np.random.rand() < prob_mu:
                        offs[6] = self.uniform_mutation(self.niche[i][j])
                        off_fits[6] = 1 / self.fn(offs[6])
                        flag = True
                if flag:
                    sel_idx = np.argmax(off_fits)
                    # sel_idx = self.roulette(off_fits)
                    self.niche[i][j] = np.copy(offs[sel_idx])
                    self.fits[i][j] = off_fits[sel_idx]
                    for io in range(num_offs):
                        if self.best_fit < off_fits[io]:
                            self.best_fit = off_fits[io]
                            self.best = np.copy(offs[io])

                self.sort(self.niche[i], self.fits[i], self.size_pop)
            self.set_elites()
            self.history_fit.append(self.best_fit)

        return 1 / self.best_fit
