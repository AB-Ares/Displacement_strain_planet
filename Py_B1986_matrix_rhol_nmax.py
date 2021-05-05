import re
import numpy as np
from sympy import linsolve, lambdify, symbols
from sympy.parsing.sympy_parser import parse_expr
import pyshtools as pysh

#################################################################
################### written by Adrien Broquet ###################
#################################################################

# This file contains several functions that solve the Banerdt
# et al. (1986) thin shell model under different assumptions.

######## Note that a few corrections are made to the model:
# 1) in eq (5), Banerdt forgot a ((R - c) / R) factor for the drho
#   term and miscalculated a pre-factor for the drhom_lm term.
# 2) in eq. (A24), we added correction from Beuthe (2008) on the displacement equations

######## A few additions:
# 1) We account for a potential difference between the density of the surface (rhol) and crust (rhoc)
# 2) The density anomaly is not always at the base of the crust, and is between top_drho and base_drho.
# 3) We added to option to add a downward continuation filter as in Wieczorek & Phillips (1998)

# The user can specify and additional equation and array
# in the matrix to be solved. This can be useful when playing
# around with the system of equations. The add_equation format
# requires all parameters to be on the same side.
# An example constraint is to assume that the surface elevation
# equals the flexure (H_lm - w_lm = 0), in which case
# add_equation = 'H_lm - w_lm'
# Another is that the surface elevation minus flexure equals a given
# thickness value (H_lm - w_lm - Thick_lm) = 0.
# add_equation = 'H_lm - w_lm - Thick_lm' and
# add_array = 'Thick_lm'

def Thin_shell_matrix_rhol(g0, R, c, Te, rhom, rhoc, rhol, rhobar,
   lmax_calc, E, v, base_drho = 50e3, top_drho = 0, filter = None,
   filter_half = 50, H_lm = None, drhom_lm = None,
   dc_lm = None, w_lm = None, omega_lm = None, q_lm = None,
   G_lm = None, Gc_lm = None, add_equation = None,
   add_array = None, quiet = False, remove_equation = None,
   w_corr = None, wdc_corr = None, H_corr = None,
   lambdify_func = None, first_inv = True, drho_corr = None):
   
   # Declare all possible input arrays.
   input_arrays = np.array([w_lm, Gc_lm, q_lm, omega_lm,
   dc_lm, drhom_lm, G_lm, H_lm], dtype = object)
   input_constraints = np.array(['w_lm','Gc_lm','q_lm',
   'omega_lm','dc_lm', 'drhom_lm','G_lm','H_lm'])
   
   equation_order = np.array(['G_lm', 'Gc_lm', 'q_lm',
      'w_lm', 'omega_lm'])
   
   # Perform some initial checks with input arrays to determine
   # what are the unknown and if a sufficient number of arrays
   # has been input.

   # Number of input arrays
   num_array_test = np.array([type(arr) for arr in \
   input_arrays]) != type(None)
   # Size of input arrays
   size_array_test = np.array([np.size(arr) for arr in \
   input_arrays])
   # Input arrays
   constraint_test = input_constraints[num_array_test]
   # Other arrays
   not_constraint = input_constraints[~num_array_test]
   
   if lmax_calc < 0:
      raise ValueError(
      "lmax_calc must be greater or equal to 0. " + \
      "Input value was {:s}."
      .format(repr(lmax_calc)))
      
   if lmax_calc > np.sqrt(np.max(size_array_test)/2)-1:
      raise ValueError(
      "lmax_calc must be less or equal to {:s}. Input " + \
      "value was {:s}."
      .format(repr(np.sqrt(np.max(size_array_test)/2)-1), \
      repr(lmax_calc+1)))
     
   # The system is a total of 5 equations relating 8 unknowns.
   # If an additional equation is given, 2 arrays must be input
   # to find a solution.
   if add_equation is not None:
      if remove_equation is None:
         if np.sum(num_array_test) != 2:
            raise ValueError(
            "Must input 2 arrays between %s." \
            %(input_constraints) + "\nNumber of input" + \
            " arrays was {:s}. "\
            .format(repr(np.sum(num_array_test))) + \
            "Input arrays are %s." %(constraint_test))
      else:
         if np.sum(num_array_test) != 3:
            raise ValueError(
            "Must input 3 arrays between %s." \
            %(input_constraints) + "\nNumber of input" + \
            " arrays was {:s}. "\
            .format(repr(np.sum(num_array_test))) + \
            "Input arrays are %s." %(constraint_test))
      if "=" in add_equation:
         raise ValueError(
         "All terms of the added equation must be" + \
         " on the same side, and there is no need to specify = 0, the input" + \
         "equation is %s. " %(add_equation))
         
      if all(sym not in add_equation for sym in input_constraints):
         raise ValueError(
         "The input equation must relate any of the 8" + \
         " unknown arrays that are %s." \
         %(input_constraints) + \
         "\nThe input equation is %s." %(add_equation))
   else:
      if remove_equation is None:
      # If no additional equation is given, 3 arrays must be
      # input to find a solution.
         if np.sum(num_array_test) != 3:
            raise ValueError(
            "Must input 3 arrays between %s." \
            %(input_constraints) \
            + "\nNumber of input arrays was {:s}. " \
            .format(repr(np.sum(num_array_test))) + \
            "Input arrays are %s." %(constraint_test))
      else:
         if np.sum(num_array_test) != 4:
            raise ValueError(
            "Must input 4 arrays between %s." \
            %(input_constraints) \
            + "\nNumber of input arrays was {:s}. " \
            .format(repr(np.sum(num_array_test))) + \
            "Input arrays are %s." %(constraint_test))
   
   if quiet is False:
      print("Input arrays are %s." %(constraint_test))
      print("Solving for %s." %(not_constraint))
      if filter is not None:
         print("Minimum %s filter" %('curvature' if \
            filter == 'Mc' else 'amplitude'))
      if first_inv is True:
         print("First inversion, storing lambdify results")
      else:
         print("Using stored solutions with new inputs")
   # Allocate arrays to be used for outputs.
   shape = (2,lmax_calc+1,lmax_calc+1)
   if first_inv:
      lambdify_func = np.zeros((lmax_calc+1), dtype=object)
   if w_lm is None:
      w_lm = np.zeros(shape)
   if Gc_lm is None:
      Gc_lm = np.zeros(shape)
   if q_lm is None:
      q_lm = np.zeros(shape)
   if omega_lm is None:
      omega_lm = np.zeros(shape)
   if drhom_lm is None:
      drhom_lm = np.zeros(shape)
   if G_lm is None:
      G_lm = np.zeros(shape)
   if H_lm is None:
      H_lm = np.zeros(shape)
   if dc_lm is None:
      dc_lm = np.zeros(shape)
   if wdc_corr is None:
      wdc_corr = np.zeros(shape)
      H_corr = np.zeros(shape)
      w_corr = np.zeros(shape)
      drho_corr = np.zeros(shape)
   A_lm = np.zeros(shape)
      
   if Te == 0: #Avoid numerical problems with infinite values
      Te = 1
      print('!! Elastic thickness set to 1 to avoid numerical problems !!')
      
   # Precompute some constants.
   M = base_drho - top_drho #Thickness of the density anomaly
   Re = R - 0.5 * Te # Midpoint of the elastic shell.
   ETERE2 = E * Te * Re**2
   Re4 = Re**4
   drho = rhom - rhoc
   drhol = rhoc - rhol
   if Te == 0: # Avoids error printing when dividing by zero.
      psi = np.inf
   else:
      psi = 12. * Re**2 / Te**2
   D = E * Te**3 / (12. * (1. - v**2)) # Shell's rigidity.
   v1v = v / (1. - v)
   RCR = (R - c) / R
   
   gmoho =  g0 * (1.+ (((R-c)/R)**3 - 1) * rhoc/rhobar) / ((R-c)/R)**2
   if top_drho <= c:
      gdrho = g0 * (1.+ (((R-top_drho)/R)**3 - 1) * rhoc/rhobar) / ((R-top_drho)/R)**2
   else:
      gdrho = g0 * (1.+ (((R-top_drho)/R)**3 - 1) * rhom/rhobar) / ((R-top_drho)/R)**2
   
   # Store symbolized array names with sympy. Order is 
   # important.
   # These will be denoted e.g. 'H_lm1' for H_lm.
   add_constraints = ''
   if add_array is not None:
      add_constraints = ' add_array1'
   add_constraints += ' wdc_corr1 w_corr1 H_corr1 drho_corr1'
      
   a_symb_uknwn = symbols(" ".join([symb + '1 ' for symb in not_constraint]))
   a_symb_knwn = symbols(" ".join([symb + '1 ' for symb in constraint_test])+add_constraints)
   args_symb = (constraint_test, not_constraint, a_symb_uknwn)
   
   if remove_equation is not None and quiet is False:
      print("Removing equation for %s." %(remove_equation))
   if add_equation is not None:
      if quiet is False:
         if add_array is None:
            print("Adding an additional equation where %s." %(add_equation))
         else:
            print("Adding an additional equation and array where %s." %(add_equation))
      # Reformat added equation for sympy
      for string in input_constraints:
         add_equation = re.sub(r'(\b{}\b)'.format(string), \
         '%s' %(string)+'1', add_equation)
      if add_array is not None:
         add_equation = re.sub(r'(\b{}\b)'.format('add_array'), \
         '%s' %('add_array')+'1', add_equation)
      add_equation = parse_expr(add_equation)
    
   # Solve matrix over all degrees.
   for l in range(1, lmax_calc+1): # Ignore degree 0 from calculations
      Lapla = float(-l * (l + 1)) # Laplacian identity.
      
      # Degree-dependent from Banerdt correction after Beuthe (2008).
      if l == 1:
         alpha = 1.e-20
         gamma = 1.e-20
         beta = 1.e-20
         eta = 1.e-20
         # No displacement for degree-1.
      else:
         alpha = -Re4 * (Lapla + 1. - v) / ((D / \
         ( 1. + 1. / psi)) * (Lapla**3 + 4. * \
         Lapla**2 + 4. * Lapla) + Re**2*(E*Te) * (Lapla + 2.))
         
         gamma = (Lapla * Re4 * ((1./(1.+psi)) * \
         (Lapla + 2.) - 1 - v)) / ((D / (1. + 1. / psi)) * \
         (Lapla**3 + 4. * Lapla**2 + 4. * Lapla) + \
         Re**2*(E * Te) * (Lapla + 2.))
         
         zeta = (1./(1.+psi)) * (1./(1.-v**2)) * (Lapla + 1. + v) * (Lapla + 2.)
         beta = zeta * alpha + alpha + Re**2/(E*Te)
         eta = zeta * gamma + gamma - Re**2/(E*Te*(1.+psi)) * (Lapla - psi * (1. + v))

      if first_inv is True:
         # Degree & radius -dependent constants for potential
         # upward continuation
         Rl3 = R / float(l+3)
         rhobconst = 3. / (rhobar * float(2*l+1))
         RCRl = RCR**l
         RCRl1 = RCR**(l+1)
         RCRl2 = RCR**(l+2)
         
         if filter is None:
            DCfilter_mohoD = 1.
            DCfilter_mohoU = 1.
         else:
            DCfilter_mohoD = DownContFilter(l, filter_half, R, R - c, type = filter)
            DCfilter_mohoU = DownContFilter(l, filter_half, R - c, R, type = filter)
         
         if (R - top_drho) <= (R - c):
            RtRCl = ((R-top_drho)/(R-c))**l
         else:
            RtRCl = ((R-c)/(R-top_drho))**(l+1)
         if (R - base_drho) <= (R - c):
            RbRCl = ((R-base_drho)/(R-c))**l
         else:
            RbRCl = ((R-c)/(R-base_drho))**(l+1)
         
         RtRCl *= (R-top_drho)**3/((R-c)*R**2)
         RbRCl *= (R-base_drho)**3/((R-c)*R**2)
            
         RtRl3 = ((R-top_drho)/R)**(l+3)
         RbRl3 = ((R-base_drho)/R)**(l+3)
         
         # Symbolic definition.
         w_lm1, Gc_lm1, q_lm1, omega_lm1, dc_lm1, drhom_lm1, \
         G_lm1, H_lm1 = symbols(' w_lm1 Gc_lm1 q_lm1' + \
         ' omega_lm1 dc_lm1 drhom_lm1 G_lm1 H_lm1 ')
         
         if add_array is not None:
            add_array1 = symbols(' add_array1 ')
            
         wdc_corr1, w_corr1, H_corr1, drho_corr1 = \
            symbols(' wdc_corr1 w_corr1 H_corr1 drho_corr1')
         
         # System of equations from Banerdt et al. (1986).
         Eqns = [
         -G_lm1 + rhobconst * (rhol * H_lm1 + drhol * w_lm1 + \
         drho * (w_lm1 - dc_lm1) * RCRl2 / DCfilter_mohoD + \
         drhom_lm1 * Rl3 * (RtRl3 - RbRl3)) + \
         rhol * H_corr1 + drhol * w_corr1 + \
         drho * wdc_corr1 * RCRl, # Corrections
         
         -Gc_lm1 + rhobconst * ((rhol * H_lm1 + drhol * w_lm1) * \
         RCRl1 + drho * (w_lm1 - dc_lm1) * RCR**3 + \
         drhom_lm1 * Rl3 * (RtRCl-RbRCl)) + \
         (rhol * H_corr1 + drhol * w_corr1) * RCRl1 + \
         drho * wdc_corr1 * RCR**3, # Corrections
         
         -q_lm1 + g0 * (rhol * (H_lm1 - G_lm1) + drhol * w_lm1) \
         + gmoho * drho * (w_lm1 - dc_lm1 - Gc_lm1) + \
         gdrho * drhom_lm1 * M,
         
         -w_lm1 + alpha * q_lm1 + gamma * omega_lm1,
         
         -omega_lm1 + v1v * rhol * g0 * Te * H_lm1 / R - \
         (drhol * g0 * v1v * Te + rhoc * gmoho * (v1v * Te - c) \
         - rhom * gmoho * (Te - c)) * w_lm1 / R - v1v * drho * \
         gmoho * (Te - c) * dc_lm1 / R - 0.5 * v1v * drhom_lm1 * \
         gdrho * (Te - top_drho) * np.min([M,Te-c]) / R + \
         drho_corr1, # Correction for density variation
         ]
         
         if add_equation is not None:
            Eqns.insert(len(Eqns), add_equation)
         
         if remove_equation is not None:
            for item in [remove_equation]:
               Eqns.pop(int(np.where(equation_order == item)[0]))
         
         # Rearange system of equations using sympy.
         sol = linsolve(Eqns, a_symb_uknwn+a_symb_knwn)
         
         # Vectorize the linsolve function.
         linsolve_vector = lambdify(a_symb_uknwn+a_symb_knwn, \
         list(sol))
         if first_inv: # Store matrix solution for potential
                       # Reutilisation later
            lambdify_func[l] = linsolve_vector
      else:
         linsolve_vector = lambdify_func[l]
         
      # Depending on the input arrays, pass a symbol or the input
      # values.
      H_lm1 = constraint_test_symb('H_lm', H_lm[:,l,:l+1], *args_symb)
      G_lm1 = constraint_test_symb('G_lm', G_lm[:,l,:l+1], *args_symb)
      Gc_lm1 = constraint_test_symb('Gc_lm', Gc_lm[:,l,:l+1], *args_symb)
      q_lm1 = constraint_test_symb('q_lm', q_lm[:,l,:l+1], *args_symb)
      omega_lm1 = constraint_test_symb('omega_lm', omega_lm[:,l,:l+1], *args_symb)
      dc_lm1 = constraint_test_symb('dc_lm', dc_lm[:,l,:l+1], *args_symb)
      drhom_lm1 = constraint_test_symb('drhom_lm', drhom_lm[:,l,:l+1], *args_symb)
      w_lm1 = constraint_test_symb('w_lm', w_lm[:,l,:l+1], *args_symb)

      # Results.
      if add_array is not None:
         outs = np.concatenate(np.array(linsolve_vector(w_lm1 = w_lm1, \
         Gc_lm1 = Gc_lm1, G_lm1 = G_lm1, H_lm1 = H_lm1, \
         q_lm1 = q_lm1, omega_lm1 = omega_lm1, dc_lm1 = dc_lm1, \
         drhom_lm1 = drhom_lm1, \
         add_array1 = add_array[:,l,:l+1], \
         wdc_corr1 = wdc_corr[:,l,:l+1], H_corr1 = H_corr[:,l,:l+1], \
         w_corr1 = w_corr[:,l,:l+1], drho_corr1 = drho_corr[:,l,:l+1]), \
         dtype=object))
      else:
         outs = np.concatenate(np.array(linsolve_vector(w_lm1 = w_lm1, \
         Gc_lm1 = Gc_lm1, G_lm1 = G_lm1, H_lm1 = H_lm1, \
         q_lm1 = q_lm1, omega_lm1 = omega_lm1, dc_lm1 = dc_lm1, \
         drhom_lm1 = drhom_lm1, \
         wdc_corr1 = wdc_corr[:,l,:l+1], H_corr1 = H_corr[:,l,:l+1], \
         w_corr1 = w_corr[:,l,:l+1], drho_corr1 = drho_corr[:,l,:l+1]), \
         dtype=object))
      
      # Determine how symbols are listed in the outputs because
      # solutions order depends on the input symbol order,
      # which depends on the user inputs.
      a_symbs = np.array(a_symb_uknwn+a_symb_knwn).astype('str')
      idx_w_lm = int(np.where(a_symbs == 'w_lm1')[0])
      idx_G_lm = int(np.where(a_symbs == 'G_lm1')[0])
      idx_Gc_lm = int(np.where(a_symbs == 'Gc_lm1')[0])
      idx_H_lm = int(np.where(a_symbs == 'H_lm1')[0])
      idx_omega_lm = int(np.where(a_symbs == 'omega_lm1')[0])
      idx_drhom_lm = int(np.where(a_symbs == 'drhom_lm1')[0])
      idx_dc_lm = int(np.where(a_symbs == 'dc_lm1')[0])
      idx_q_lm = int(np.where(a_symbs == 'q_lm1')[0])
      
      # Write solutions
      w_lm[:,l,:l+1] = outs[idx_w_lm]
      Gc_lm[:,l,:l+1] = outs[idx_Gc_lm]
      q_lm[:,l,:l+1] = outs[idx_q_lm]
      omega_lm[:,l,:l+1] = outs[idx_omega_lm]
      dc_lm[:,l,:l+1] = outs[idx_dc_lm]
      drhom_lm[:,l,:l+1] = outs[idx_drhom_lm]
      G_lm[:,l,:l+1] = outs[idx_G_lm]
      H_lm[:,l,:l+1] = outs[idx_H_lm]
      
      # Tangential displacement
      A_lm[:,l,:l+1] = beta * q_lm[:,l,:l+1] + eta * \
      omega_lm[:,l,:l+1]
      
   return w_lm, A_lm, w_lm - dc_lm, dc_lm, drhom_lm, omega_lm, \
   q_lm, Gc_lm, G_lm, H_lm, lambdify_func
   
def constraint_test_symb(str_symb, arr, constraint_test, not_constraint, arr_symb):
#  This function finds the indice of the input symbol in the solution arrays
   if str_symb in constraint_test:
      out = arr
   else:
      out = \
      arr_symb[int(np.where(not_constraint==str_symb)[0])]
      
   return out

def DownContFilter(l, half, R_ref, D_relief, type = 'Mc'):
#  This function will compute the minimum amplitude ('Ma') or curvature ('Mc') downward continuation
#  filter of Wieczorek and Phillips 1998 for degree l, where the filter is
#  assumed to be equal to 0.5 at degree half.
   if (half == 0):
      DownContFilter = 1.
   else:
      if type == 'Mc':
         tmp = 1./(float(half * half+half) * (float(2*half+1) * (R_ref/D_relief)**half)**2)
         DownContFilter = 1. + tmp * float(l*l+l) * (float(2*l+1) * (R_ref/D_relief)**l)**2
      elif type == 'Ma':
         tmp = 1./(float(2.*half+1.) * (R_ref/D_relief)**half)**2
         DownContFilter = 1. + tmp * (float(2*l+1) * (R_ref/D_relief)**l)**2
      else:
         raise ValueError(
         "Error in DownContFilter, filter type must be either 'Ma' or 'Mc' " + \
         "Input value was {:s}."
         .format(repr(type)))
   DownContFilter = 1. / DownContFilter
   
   return DownContFilter
