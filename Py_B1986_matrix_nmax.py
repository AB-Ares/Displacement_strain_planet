import numpy as np
import pyshtools as pysh
from Py_B1986_matrix_rhol_nmax import *

#################################################################
################### written by Adrien Broquet ###################
#################################################################
   
def corr_nmax_drho(dr_lm, drho_lm, shape_grid, rho_grid, lmax_calc, mass, \
   nmax, drho, R, c = 0, density_var = False, filter = False, filter_half = None):
   
   # This routine estimates the delta between the finite-amplitude correction with and
   # without density variations and the mass-sheet case computed in the thin-shell code.
   # The difference will be iteratively added to the thin-shell code.
   
   # Finite-amplitude correction.
   MS_lm_nmax = pysh.SHCoeffs.from_zeros(lmax_calc).coeffs
   # This is the computation in Thin_shell_matrix_rhol.
   for l in range(1,lmax_calc+1):
      MS_lm_nmax[:,l,:l+1] = drho * dr_lm[:,l,:l+1] / (2 * l + 1)
   MS_lm_nmax *=  4. * np.pi / mass
   
   if nmax != 1:
      # This is the correct calculation with finite-amplitude
      FA_lm_nmax, D = pysh.gravmag.CilmPlusRhoHDH(shape_grid, nmax, mass,
        rho_grid, lmax = lmax_calc)
      MS_lm_nmax *=  D**2
   else:
      FA_lm_nmax = MS_lm_nmax

   # Density contrast in the relief correction.
   if density_var:
      FA_lm_drho, D = pysh.gravmag.CilmPlusRhoHDH(shape_grid, 1, mass,
        rho_grid, lmax = lmax_calc)
      MS_lm_drho = MS_lm_nmax
      if filter:
         for l in range(1,lmax_calc+1):
            MS_lm_drho[:,l,:l+1] /= DownContFilter(l, filter_half, R, R - c, type = filter)
            FA_lm_drho[:,l,:l+1] /= DownContFilter(l, filter_half, R, R - c, type = filter)
      if nmax == 1:
         MS_lm_drho *= D**2
      
      # Divide because the thin-shell code multiplies by density contrast,
      # to correct for finite-amplitude. Here we also correct for density variations,
      # so the correction is already scaled by the density contrast.
      return R * (FA_lm_drho - MS_lm_drho + FA_lm_nmax - MS_lm_nmax) / drho
   else:
   
      return R * (FA_lm_nmax - MS_lm_nmax)
   
def Thin_shell_matrix_nmax(g0, R, c, Te, rhom, rhoc, rhol, rhobar, \
   lmax_calc, E, v, mass, filter_half = 50, nmax = 5, \
   H_lm = None, drhom_lm = None, dc_lm = None, w_lm = None, \
   omega_lm = None, q_lm = None, G_lm = None, Gc_lm = None, \
   filter = None, C_lm = None, add_equation = None, add_array = None, \
   quiet = True, remove_equation = None, base_drho = 150e3, top_drho = 50e3, \
   delta_max = 5, iter_max = 250):
   
   # This function will solve the Banerdt system of equations (see
   # Py_B1986_matrix) as a function of the given inputs and using
   # the finite-amplitude correction of Wieczorek & Phillips
   # (1998) when nmax > 1.
   
   # Density variations in the surface or moho reliefs are also accounted
   # for any nmax, and when top_drho == 0 or base_drho == c.
   
   args_param_m = (g0, R, c, Te, rhom, rhoc, rhol, rhobar, lmax_calc, E, v)
   args_param_lm = dict(H_lm = H_lm, drhom_lm = drhom_lm, dc_lm = dc_lm, w_lm = w_lm, \
   omega_lm = omega_lm, q_lm = q_lm, G_lm = G_lm, Gc_lm = Gc_lm, \
   base_drho = base_drho, top_drho = top_drho, \
   filter = filter, filter_half = filter_half, add_array = add_array, \
   remove_equation = remove_equation, add_equation = add_equation, quiet = quiet)
   
   lmaxgrid = 4 * lmax_calc
   args_grid = dict(grid = 'DH2', lmax = lmaxgrid, extend = False, lmax_calc = lmax_calc)
   
   # Precompute some sums that will be used later for checks
   sum_dc = np.sum(dc_lm)
   sum_w = np.sum(w_lm)
   sum_drho = np.sum(drhom_lm)
   
   # Density contrast not at topography or moho and no finite-amplitude correctio, return
   if nmax == 1 and top_drho != 0 and base_drho != c:
      if quiet is False:
         print('Returning without finite-amplitude corrections')
      w_lm_o, A_lm_o, moho_relief_lm_o, dc_lm_o, drhom_lm_o, omega_lm_o, \
      q_lm_o, Gc_lm_o, G_lm_o, H_lm_o, lambdify_func_o = Thin_shell_matrix_rhol(*args_param_m, \
      **args_param_lm)
      
      if quiet is False:
         print('Set the interfaces degree-0 coefficients')
      w_lm_o[0,0,0] = R
      dc_lm_o[0,0,0] = c
      moho_relief_lm_o[0,0,0] = R - c
      H_lm_o[0,0,0] = R
      
      return w_lm_o, A_lm_o, moho_relief_lm_o, dc_lm_o, drhom_lm_o, omega_lm_o, \
      q_lm_o, Gc_lm_o, G_lm_o, H_lm_o, lambdify_func_o
      
   else:
   # Correct for density contrast in surface or moho relief, and/or finite-amplitude correction
      density_var_H, density_var_dc, density_var = False, False, False
      if drhom_lm is None or sum_drho != 0:
         density_var = True # Variations in density
         if quiet is False:
            print('Iterate to account for density' + \
            ' variations %s' %('and finite-amplitude correction, nmax is %i' %(nmax) \
            if nmax > 1 else ''))
         if top_drho == 0:
         # Correct for density variations in the surface relief
            density_var_H = True
         if base_drho == c:
         # Correct for density variations in the moho relief
            density_var_dc = True
      else:
         if quiet is False:
            print('Iterate for finite-amplitude correction, nmax is %i' %(nmax))

      # If only finite-amplitude correction, density
      # contrast is multipled in the thin-shell code
      # we set it to 1. This will be changed later if required.
      ones = np.ones((2*(lmaxgrid+1),2*(2*(lmaxgrid+1))))
      H_drho_grid, w_drho_grid, wdc_drho_grid = ones, ones, ones
      drho_H, drho_wdc, drho_w = 1., 1., 1.
      
      if drhom_lm is not None and sum_drho != 0:
         rho_grid = pysh.SHCoeffs.from_array(drhom_lm).expand(**args_grid).data
         rhoc = drhom_lm[0,0,0]
         rhol = drhom_lm[0,0,0]
         args_param_m = (g0, R, c, Te, rhom, rhoc, rhol, rhobar, lmax_calc, E, v)
         if quiet is False:
            print("rhol and rhoc are set to the mean input density variations %.2f kg m-3" %(rhoc))
         
      # Geoid correction due to density variation
      # and or finite-amplitude corrections.
      # Moho relief
      delta_wdc_geoid = pysh.SHCoeffs.from_zeros(lmax_calc).coeffs
      # Deflected topography relief
      delta_w_geoid = pysh.SHCoeffs.from_zeros(lmax_calc).coeffs
      # Surface topography relief
      delta_H_geoid = pysh.SHCoeffs.from_zeros(lmax_calc).coeffs
      # Tangential load potential corrections due to density variations
      # at the reliefs
      drho_corr = pysh.SHCoeffs.from_zeros(lmax_calc).coeffs
      
      precomp_H_grid, precomp_w_grid, precomp_dc_grid = False, False, False
      # Precompute grids
      if H_lm is not None:
         precomp_H_grid = True
         H_lm[0,0,0] = R
         H_grid = pysh.SHCoeffs.from_array(H_lm).expand(**args_grid).data
      if w_lm is not None and rhoc != rhol:
         precomp_w_grid = True
         if sum_w == 0:
            w_grid = ones * R
         else:
            w_lm[0,0,0] = R
            w_grid = pysh.SHCoeffs.from_array(w_lm).expand(**args_grid).data
      if w_lm is not None and dc_lm is not None:
         precomp_dc_grid = True
         wdc_lm = w_lm - dc_lm
         if sum_w == 0 and sum_dc == 0:
            wdc_grid = ones * R - c
         else:
            wdc_lm[0,0,0] = R - c
            wdc_grid = pysh.SHCoeffs.from_array(wdc_lm).expand(**args_grid).data
      
      
      # Error messages if iteration not converging
      var_unit = 'km'
      var_relief = 'Moho relief'
      if sum_dc == 0 and sum_w == 0:
         var_relief = 'Grid density'
         var_unit = 'kg m-3'
      elif sum_dc == 0:
         var_relief = 'Flexure relief'
         
      lambdify_func_o = None
      first_inv = True
      delta = 1.e9
      iter = 0
      # Iterate until convergence
      # First guess is using the mass-sheet case
      while delta > delta_max:
         iter += 1
         w_lm_o, A_lm_o, moho_relief_lm_o, dc_lm_o, drhom_lm_o, omega_lm_o, \
         q_lm_o, Gc_lm_o, G_lm_o, H_lm_o, lambdify_func_o = Thin_shell_matrix_rhol(*args_param_m, \
         **args_param_lm, wdc_corr = delta_wdc_geoid, w_corr = delta_w_geoid, \
         H_corr = delta_H_geoid, first_inv = first_inv, \
         lambdify_func = lambdify_func_o, drho_corr = drho_corr)
         first_inv = False
         
         # Scheme proposed in Wieczorek+(2013) SOM eq 21, 22 to
         # speed up convergence delta(i+3) = (delta(i+2) + delta(i+1))/2..
         if iter % 3 == 0:
            delta_wdc_geoid = (delta_wdc_geoid_2 + delta_wdc_geoid_1) / 2.
            delta_H_geoid = (delta_H_geoid_2 + delta_H_geoid_1) / 2.
            delta_w_geoid = (delta_w_geoid_2 + delta_w_geoid_1) / 2.
            if quiet is False:
                print('Skipping iteration %s, with convergence'  %(iter) + \
                ' scheme')
            continue
         
         if density_var:
            rho_grid = pysh.SHCoeffs.from_array(drhom_lm_o).expand(**args_grid).data
            if drhom_lm is not None and sum_drho != 0:
               rhoc = drhom_lm_o[0,0,0]
               rhol = drhom_lm_o[0,0,0]
               args_param_m = (g0, R, c, Te, rhom, rhoc, rhol, rhobar, lmax_calc, E, v)
            gmoho = g0 * (1.+ (((R - c) / R)**3 - 1) * rhoc / rhobar) / ((R - c) / R)**2
            v1v = v / (1. - v)
            
            if density_var_H:
               drho_corr += v1v * drhom_lm_o * g0 * Te * H_lm_o / R
               drho_H = rhol
               if drhom_lm is not None and sum_drho != 0:
                  H_drho_grid = rho_grid
               else:
                  H_drho_grid = rho_grid + rhol
            if density_var_dc:
               drho_corr += v1v * drhom_lm_o * gmoho * (Te - c) * dc_lm_o / R
               drho_wdc = rhom - rhoc
               if drhom_lm is not None and sum_drho != 0:
                  wdc_drho_grid = rhom - rho_grid
               else:
                  wdc_drho_grid = rhom - (rhoc + rho_grid)
         
         H_lm_o[0,0,0] = R
         if not precomp_H_grid:
            H_grid = pysh.SHCoeffs.from_array(H_lm_o).expand(**args_grid).data
         delta_H_geoid = corr_nmax_drho(H_lm_o, drhom_lm_o, H_grid, H_drho_grid, lmax_calc, mass, nmax, drho_H, R, density_var = density_var_H)
         
         w_lm_o[0,0,0] = R
         if not precomp_w_grid:
            w_grid = pysh.SHCoeffs.from_array(w_lm_o).expand(**args_grid).data
         if rhoc != rhol:
            delta_w_geoid = corr_nmax_drho(w_lm_o, drhom_lm_o, w_grid, w_drho_grid, lmax_calc, mass, nmax, drho_w, R)
         
         wdc_lm_o = w_lm_o - dc_lm_o
         wdc_lm_o[0,0,0] = R - c
         if not precomp_dc_grid:
            wdc_grid = pysh.SHCoeffs.from_array(wdc_lm_o).expand(**args_grid).data
         delta_wdc_geoid = corr_nmax_drho(wdc_lm_o, drhom_lm_o, wdc_grid, wdc_drho_grid, lmax_calc, mass, nmax, drho_wdc, R, density_var = density_var_dc, filter = filter, filter_half = filter_half, c = c)
                  
         if iter != 1:
            if sum_dc == 0:
               if sum_w != 0:
                  delta = abs(grid_prev - w_grid).max()
                  if quiet is False:
                      print('Iteration %i, Delta (km) = %.3f' %(iter, delta / 1e3))
                      print('Maximum displacement (km) = %.2f' %(((w_grid-R)/1e3).max()))
                      print('Minimum displacement (km) = %.2f' %(((w_grid-R)/1e3).min()))
               else:
                  delta = abs(grid_prev - rho_grid).max() * 1e3
                  if quiet is False:
                      print('Iteration %i, Delta (kg m-3) = %.3f' %(iter, delta/1e3))
                      print('Maximum density (kg m-3) = %.2f' %(rho_grid.max()))
                      print('Minimum density (kg m-3) = %.2f' %(rho_grid.min()))
            else:
               delta = abs(grid_prev - (R - wdc_grid - c)).max()
               if quiet is False:
                   print('Iteration %i, Delta (km) = %.3f' %(iter, delta / 1e3))
                   crust_thick = (H_grid - wdc_grid) / 1e3
                   print('Maximum Crustal thickness (km) = %.2f' %(crust_thick.max()))
                   print('Minimum Crustal thickness (km) = %.2f' %(crust_thick.min()))
         
         # Speed up convergence scheme
         if iter % 2 == 0:
            delta_wdc_geoid_2 = delta_wdc_geoid
            delta_H_geoid_2 = delta_H_geoid
            delta_w_geoid_2 = delta_w_geoid
         else:
            delta_wdc_geoid_1 = delta_wdc_geoid
            delta_H_geoid_1 = delta_H_geoid
            delta_w_geoid_1 = delta_w_geoid
         
         if sum_dc == 0:
            if sum_w != 0:
               grid_prev = w_grid
            else:
               grid_prev = rho_grid
         else:
            grid_prev = R - wdc_grid - c
            
         if iter > iter_max:
            raise ValueError(
            "%s not converging, stopped at iteration %i, " %(var_relief, iter) + \
            "delta was %.4f (%s) and delta_max is %.4f (%s)."
            %(delta/1e3, var_unit, delta_max/1e3, var_unit))
            exit(1)
         if delta > 500e3 and iter != 1:
            raise ValueError(
            "%s not converging, stopped at iteration %i, " %(var_relief,iter) + \
            "delta was %.4f (%s) and delta_max is %.4f (%s). Try modifying nmax%s"  \
            %(delta/1e3, var_unit, delta_max/1e3, var_unit, \
            " or try filtering." if filter == 0 else "."))
            exit(1)
   
   if quiet is False:
      print('Set the interfaces degree-0 coefficients')
   w_lm_o[0,0,0] = R
   dc_lm_o[0,0,0] = c
   moho_relief_lm_o[0,0,0] = R - c
   H_lm_o[0,0,0] = R
   
   return w_lm_o, A_lm_o, moho_relief_lm_o, dc_lm_o, drhom_lm_o, omega_lm_o, \
   q_lm_o, Gc_lm_o, G_lm_o, H_lm_o, lambdify_func_o
