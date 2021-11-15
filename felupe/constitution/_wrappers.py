class MatadiMaterial:
    def __init__(self, material, threads=1):
        "Wrap a MatADi Materials into a FElupe material."
        self.material = material
        self.threads = threads

    def function(self, *args):
        fun = self.material.function(args, threads=self.threads)
        if len(fun) == 1:
            return fun[0]
        else:
            return fun

    def gradient(self, *args):
        grad = self.material.gradient(args, threads=self.threads)
        if len(grad) == 1:
            return grad[0]
        else:
            return grad

    def hessian(self, *args):
        hess = self.material.hessian(args, threads=self.threads)
        if len(hess) == 1:
            return hess[0]
        else:
            return hess
