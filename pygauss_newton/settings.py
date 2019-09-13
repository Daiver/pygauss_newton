class Settings:
    def __init__(self,
                 n_max_iterations=50,
                 damping_constant_absolute=0.0,
                 loss_stop_threshold=1e-10,
                 grad_norm_stop_threshold=1e-10,
                 step_norm_stop_threshold=1e-10,
                 verbose=True):
        self.n_max_iterations = n_max_iterations
        self.damping_constant_absolute = damping_constant_absolute

        self.loss_stop_threshold = loss_stop_threshold
        self.grad_norm_stop_threshold = grad_norm_stop_threshold
        self.step_norm_stop_threshold = step_norm_stop_threshold

        self.verbose = verbose
