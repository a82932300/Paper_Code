class DiffeqSolver(nn.Module):
    def __init__(self, input_dim, ode_func, method, latents,
                 odeint_rtol=1e-4, odeint_atol=1e-5, device=torch.device("cpu")):
        super(DiffeqSolver, self).__init__()

        self.ode_method = method
        self.latents = latents
        self.device = device
        self.ode_func = ode_func

        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

    def forward(self, first_point, time_steps_to_predict, backwards=False):
        """
		# Decode the trajectory through ODE Solver
		"""
        # n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]

        pred_y = odeint(self.ode_func, first_point, time_steps_to_predict,
                        rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)
        # pred_y = pred_y.permute(1, 0, 2, 3, 4)  # => [b, t, c, h0, w0]

        return pred_y

    def sample_traj_from_prior(self, starting_point_enc, time_steps_to_predict, n_traj_samples=1):
        """
		# Decode the trajectory through ODE Solver using samples from the prior

		time_steps_to_predict: time steps at which we want to sample the new trajectory
		"""
        func = self.ode_func.sample_next_point_from_prior

        pred_y = odeint(func, starting_point_enc, time_steps_to_predict,
                        rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)
        pred_y = pred_y.permute(1, 2, 0, 3)
        return pred_y


#####################################################################################################

class ODEFunc(nn.Module):
    def __init__(self, input_dim, latent_dim, ode_func_net, device=torch.device("cpu")):

        super(ODEFunc, self).__init__()

        self.input_dim = input_dim
        self.device = device
        self.latent_dim = latent_dim
        self.gradient_net = ode_func_net

        self.U = nn.Parameter(torch.zeros(size=(latent_dim, node_num)))
        nn.init.xavier_uniform_(self.U.data, gain=1.414)  # 初始化

    def forward(self, t_local, states, backwards=False):
        z = states[0]
        adj = states[-1]

        dz = self.get_ode_gradient_nn(t_local, z)
        da = dz @ (self.U)
        a = z @ (self.U)
        dh = da @ z + a @ dz
        return tuple([dz, da, dh, adj])

    def get_ode_gradient_nn(self, t_local, y):
        output = self.gradient_net(y)
        return output

    def sample_next_point_from_prior(self, t_local, y):
        return self.get_ode_gradient_nn(t_local, y)


class main(nn.Module):
    def __init__(self, inputdim, hiddendim):
        super(main, self).__init__()

        self.hiddendim = hiddendim

        self.linear = nn.Linear(inputdim, hiddendim)

        self.global_lin_o = nn.Linear(hiddendim, hiddendim, bias=True)
        self.global_lin_d = nn.Linear(hiddendim, hiddendim, bias=True)
        self.global_lin_his = nn.Linear(node_num, hiddendim, bias=True)

        self.local_lin_o = nn.Linear(hiddendim, hiddendim, bias=True)
        self.local_lin_d = nn.Linear(hiddendim, hiddendim, bias=True)
        # self.local_lin_his = nn.Linear(node_num, hiddendim, bias=True)
        # self.output_gat = GraphAttentionLayer(hiddendim,128,0.5,0.3)

        self.outputlin = nn.Linear(hiddendim, node_num)
        self.outputlin_2 = nn.Linear(hiddendim, node_num)

        self.linear_w = nn.Linear(hiddendim, hiddendim)

        # self.odefunc = nn.Linear(hiddendim,hiddendim)

        layers = []

        layers.append(nn.Linear(hiddendim, hiddendim))
        layers.append(nn.ELU())
        self.odefunc = nn.Sequential(*layers)

        self.rec_ode_func = ODEFunc(hiddendim, hiddendim, self.odefunc)
        self.z0_diffeq_solver = DiffeqSolver(hiddendim, self.rec_ode_func, method="euler", latents=hiddendim,
                                             odeint_rtol=1e-3, odeint_atol=1e-4)
        self.act = nn.ELU()

    def forward(self, adj_set, adj_w):
        # x = self.linear(adj_w)

        # x = self.act(x)

        t = torch.tensor([0, 1, 2]).float()

        h = torch.zeros(node_num, self.hiddendim)

        a = torch.zeros(node_num, node_num)
        k = torch.zeros(node_num, self.hiddendim)

        g_adj_set = []

        for i in range(len(adj_set)):
            re = adj_w[i]

            re = self.global_lin_his(re)

            # adj_i = self.gadj(adj_set[i],x[i])
            global_f = re.sum(0) / node_num
            delta_global = global_f - (h.sum(0) / node_num)

            global_f = self.global_lin_o(global_f) + self.global_lin_d(delta_global)
            global_f = F.sigmoid(global_f)
            # A = adj_i + adj_i @ adj_i

            delta_local = re - h
            local_f = self.local_lin_o(re) + self.local_lin_d(delta_local)
            local_f = F.tanh(local_f)

            h = global_f * local_f

            z = h


            stats = self.z0_diffeq_solver(tuple([z, a, k, adj_set[i]]), t)
            stats = tuple([stats[0][-1], stats[1][-1], stats[2][-1], stats[3][-1]])

            z = stats[0]
            a = stats[1]
            k = stats[2]
            h = z

            g_adj_set.append(h + k)
        g_adj_set = torch.stack(g_adj_set)
        g_adj_set = self.act(self.outputlin(g_adj_set))
        return g_adj_set[-1], g_adj_set
