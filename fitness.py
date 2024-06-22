import numpy as np

class fitness:
    def __init__(self, srv_rate, req_rate, cloud_delay=0.1):
        self.srv_rate = srv_rate
        self.req_rate = req_rate
        self.num_srv = srv_rate.shape[0]
        self.num_bs = req_rate.shape[0]
        self.cloud_delay = cloud_delay

    # fitness function: avg_delay_greedy
    def fn(self, solution):
        delay = np.zeros(self.num_bs, dtype=int) + self.cloud_delay
        srv_bs = np.zeros(self.num_bs, dtype=int)
        for i in range(self.num_srv):
            bs_idx = solution[i]
            srv_bs[bs_idx] = srv_bs[bs_idx] + self.srv_rate[i]
        for i in range(self.num_bs):
            #  1/(mu-lamda) by queue theory
            bs_delay = 1 / (srv_bs[i] - self.req_rate[i])
            if self.cloud_delay > bs_delay > 0:
                delay[i] = bs_delay
        return sum(delay * self.req_rate) / sum(self.req_rate)
