from matplotlib import pyplot as plt
import sympy as sp
import numpy as np
from scipy.optimize import least_squares

def sym_mat(matrix):
    matout = (matrix + matrix.T)/2
    return matout

def frob_inner(mat1,mat2):
    matout = sp.Trace(mat1.T * mat2)
    return matout

class BiaxialTest:
    def __init__(self):

        self.lam = sp.symbols('lam')             # Stretch
        self.F_biax = sp.Matrix([[self.lam, 0., 0.],  # Biaxial deformation gradient
                            [0., self.lam, 0.],
                            [0., 0., 1/self.lam**2]])

        self.data_remote_fiber = np.loadtxt('raw_data/base_fiber.csv', delimiter=',')
        self.data_remote_fiber[:,0] -= np.min(self.data_remote_fiber[:,0])
        self.data_remote_crossfiber = np.loadtxt('raw_data/base_crossfiber.csv', delimiter=',')
        self.data_remote_crossfiber[:,0] -= np.min(self.data_remote_crossfiber[:,0])
        self.data_infarct_fiber = np.loadtxt('raw_data/infarct_fiber.csv', delimiter=',')
        self.data_infarct_fiber[:,0] -= np.min(self.data_infarct_fiber[:,0])
        self.data_infarct_crossfiber = np.loadtxt('raw_data/infarct_crossfiber.csv', delimiter=',')
        self.data_infarct_crossfiber[:,0] -= np.min(self.data_infarct_crossfiber[:,0])


    def get_material_param_names(self, material):
        if material == 'fung':
            return ['C', 'bf', 'bt', 'bfs']
        elif material == 'ho':
            return ['a', 'b', 'af', 'bf']
        elif material == 'ho_linear':
            return ['a', 'af']
        elif material == 'de_linear':
            return ['fib_scale', 'iso_scale']
        else:
            raise ValueError('Material not recognized')
        

    def fung_mat(self, params):
        assert len(params) == 4, 'Fung material requires 4 parameters'

        F = sp.MatrixSymbol('F', 3, 3)  # Deformation gradient
        E = 0.5*(F.T*F - sp.Identity(3))    # Green-Lagrange strain tensor

        C_f, bf, bt, bfs = params
        Q = bf*E[0,0]**2 + bt*(E[1,1]**2 + E[2,2]**2 + E[1,2]**2 + E[2,1]**2) + bfs*(E[0,1]**2 + E[1,0]**2 + E[0,2]**2 + E[2,0]**2)
        W = 0.5*C_f*(sp.exp(Q) - 1)
        P = sp.Matrix([[W.diff(F[0,0]), W.diff(F[0,1]), W.diff(F[0,2])],
                        [W.diff(F[1,0]), W.diff(F[1,1]), W.diff(F[1,2])],
                        [W.diff(F[2,0]), W.diff(F[2,1]), W.diff(F[2,2])]])

        P_biax = P.subs(F, self.F_biax)
        sigma_biax = P_biax*self.F_biax.T

        return sigma_biax


    def ho_mat(self, params):
        assert len(params) == 4, 'HO material requires 4 parameters'

        F = sp.MatrixSymbol('F', 3, 3)
        f0 = sp.MatrixSymbol('f0', 3, 1)
        f0 = sp.Matrix([1.,0.,0.])

        C = F.T*F
        I_C = C[0,0] + C[1,1] + C[2,2]
        I_Cf = f0.T*C*f0
        I_Cf = I_Cf[0,0]

        a, b, af, bf = params
        W = a/(2*b)*(sp.exp(b*(I_C-3))-1) + af/(2*bf)*(sp.exp(bf*(I_Cf - 1)**2)-1)
        P = sp.Matrix([[W.diff(F[0,0]), W.diff(F[0,1]), W.diff(F[0,2])],
                    [W.diff(F[1,0]), W.diff(F[1,1]), W.diff(F[1,2])],
                    [W.diff(F[2,0]), W.diff(F[2,1]), W.diff(F[2,2])]])

        P_biax = P.subs(F, self.F_biax)
        sigma_biax = P_biax*self.F_biax.T

        return sigma_biax


    def ho_linear_mat(self, params):
        assert len(params) == 2, 'HO linear material requires 2 parameters'

        F = sp.MatrixSymbol('F', 3, 3)
        f0 = sp.Matrix([1.,0.,0.])

        C = F.T*F
        I_C = C[0,0] + C[1,1] + C[2,2]
        I_Cf = f0.T*C*f0
        I_Cf = I_Cf[0,0]

        a, af = params
        b, bf = 5, 5
        W = a/(2*b)*(sp.exp(b*(I_C-3))-1) + af/(2*bf)*(sp.exp(bf*(I_Cf - 1)**2)-1)
        P = sp.Matrix([[W.diff(F[0,0]), W.diff(F[0,1]), W.diff(F[0,2])],
                    [W.diff(F[1,0]), W.diff(F[1,1]), W.diff(F[1,2])],
                    [W.diff(F[2,0]), W.diff(F[2,1]), W.diff(F[2,2])]])

        P_biax = P.subs(F, self.F_biax)
        sigma_biax = P_biax*self.F_biax.T

        return sigma_biax


    def de_mat_linear(self, params):
        assert len(params) == 2, 'DE linear material requires 2 parameters'

        fib_scale, iso_scale = params

        b1 = 10.02
        b2 = 1.158

        bff = fib_scale + iso_scale
        bss = iso_scale
        bnn = iso_scale

        bfs = 6.175
        bfn = 3.520
        bsn = 2.895

        F = sp.MatrixSymbol('F', 3, 3)
        f0 = sp.Matrix([1.,0.,0.])
        s0 = sp.Matrix([0.,1.,0.])
        n0 = sp.Matrix([0.,0.,1.])

        C = F.T*F
        I_C = C[0,0] + C[1,1] + C[2,2]

        I_ff = f0.T*C*f0
        I_ff = I_ff[0,0]
        I_ss = s0.T*C*s0
        I_ss = I_ss[0,0]
        I_nn = n0.T*C*n0
        I_nn = I_nn[0,0]

        I_fs = frob_inner(C,sym_mat(f0*s0.T))
        I_fn = frob_inner(C,sym_mat(f0*n0.T))
        I_sn = frob_inner(C,sym_mat(s0*n0.T))

        W1 = sp.exp(b1*(I_C - 3))
        W2 = sp.exp(b2*(I_fs**2 + I_fn**2 + I_sn**2))

        S_sum1_term1 = bff*(W1*I_ff - 1)*(f0*f0.T)
        S_sum1_term2 = bss*(W1*I_ss - 1)*(s0*s0.T)
        S_sum1_term3 = bnn*(W1*I_nn - 1)*(n0*n0.T)

        S_sum2_term1 = bfs*I_fs*sym_mat(f0*s0.T)
        S_sum2_term2 = bfn*I_fn*sym_mat(f0*n0.T)
        S_sum2_term3 = bsn*I_sn*sym_mat(s0*n0.T)

        S_sum1 = S_sum1_term1 + S_sum1_term2 + S_sum1_term3
        S_sum2 = W2*(S_sum2_term1 + S_sum2_term2 + S_sum2_term3)

        S = S_sum1 + S_sum2

        P = F*S
        P_biax = P.subs(F, self.F_biax)
        sigma_biax = P_biax*self.F_biax.T

        return sigma_biax


    def add_plane_stress_contribution(self, sigma_biax):
        p = sp.symbols('p')
        s33 = sigma_biax[2,2] + p
        p_biax = sp.solve(s33, p)[0]

        sigma_biax = sigma_biax + p_biax*sp.eye(3)
        return sigma_biax


    def get_biaxial_func(self, material, params, planestress=False):
        if material == 'fung':
            sigma_biax = self.fung_mat(params)
        elif material == 'ho':
            sigma_biax = self.ho_mat(params)
        elif material == 'ho_linear':
            sigma_biax = self.ho_linear_mat(params)
        elif material == 'de_linear':
            sigma_biax = self.de_mat_linear(params)

        if planestress:
            sigma_biax = self.add_plane_stress_contribution(sigma_biax)

        func_11 = sp.lambdify(self.lam, sigma_biax[0,0])
        func_22 = sp.lambdify(self.lam, sigma_biax[1,1])

        def biaxial_func(l):
            s_11 = func_11(l)
            s_22 = func_22(l)
            return np.vstack([s_11, s_22])

        return biaxial_func


    def error_respect_fung_func(self, ho_params, *fung_params):
        x = np.linspace(1., 1.15)
        ho_func = self.get_biaxial_func('ho', ho_params, planestress=True)
        fung_func = self.get_biaxial_func('fung', fung_params)

        error = ho_func(x) - fung_func(x)

        return error.flatten()


    def error_respect_data(self, params, material, data='remote'):
        if data == 'remote':
            xf, yf = self.data_remote_fiber.T
            xcf, ycf = self.data_remote_crossfiber.T
        elif data == 'infarct':
            xf, yf = self.data_infarct_fiber.T
            xcf, ycf = self.data_infarct_crossfiber.T

        # Transforming to stretch
        xf = np.sqrt(2*xf + 1)
        xcf = np.sqrt(2*xcf + 1)

        mat_func = self.get_biaxial_func(material, params, planestress=True)
        mat_f, _ = mat_func(xf)
        _, mat_cf = mat_func(xcf)

        error_f = mat_f - yf
        error_cf = mat_cf - ycf

        return np.concatenate([error_f, error_cf])



if __name__ == '__main__':

    material = 'de_linear'    # ho_linear, ho, fung, de_linear

    # Initialize class
    biaxial_test = BiaxialTest()
    param_names = biaxial_test.get_material_param_names(material)

    # Fit remote data
    sol = least_squares(biaxial_test.error_respect_data, np.ones(2), args=(material, 'remote'), bounds=(0.,np.inf))
    params_remote = sol.x
    remote_func = biaxial_test.get_biaxial_func(material, params_remote, planestress=True)

    print('Remote params:', [name + ': ' + str(param) for name, param in zip(param_names, params_remote)])

    # Fit infarct data
    sol = least_squares(biaxial_test.error_respect_data, np.ones(2), args=(material, 'infarct'), bounds=(0.,np.inf))
    params_infarct = sol.x
    infarct_func = biaxial_test.get_biaxial_func(material, params_infarct, planestress=True)
    print('Infarct params:', [name + ': ' + str(param) for name, param in zip(param_names, params_infarct)])

    print('Scalings:', params_infarct/params_remote)

    # Plots
    x = np.linspace(1., 1.15)

    plt.figure(1,clear=True)
    plt.plot(*biaxial_test.data_remote_fiber.T, 'bo', label='data remote')
    plt.plot(*biaxial_test.data_infarct_fiber.T, 'bo', label='data infarct')
    plt.plot(0.5*(x**2-1), remote_func(x)[0,:], 'b--', label='fit remote')
    plt.plot(0.5*(x**2-1), infarct_func(x)[0,:], 'b--', label='fit infarct')
    plt.grid(True)
    plt.xlim([0,0.1])
    plt.ylim([0,25])
    plt.title('Fiber')
    plt.xlabel('Strain [-]')
    plt.ylabel('Stress [kPa]')
    plt.savefig(f'{material}_fit_fiber.png', bbox_inches='tight', dpi=180)


    plt.figure(2,clear=True)
    plt.plot(*biaxial_test.data_remote_crossfiber.T, 'ro', label='data remote')
    plt.plot(*biaxial_test.data_infarct_crossfiber.T, 'ro', label='data infarct')
    plt.plot(0.5*(x**2-1), remote_func(x)[1,:], 'r--', label='fit remote')
    plt.plot(0.5*(x**2-1), infarct_func(x)[1,:], 'r--', label='fit infarct')
    plt.title('Crossfiber')
    plt.grid(True)
    plt.xlim([0,0.1])
    plt.ylim([0,25])
    plt.xlabel('Strain [-]')
    plt.ylabel('Stress [kPa]')
    plt.savefig(f'{material}_fit_remote.png', bbox_inches='tight', dpi=180)